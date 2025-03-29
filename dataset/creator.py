import os
import h5py
import rasterio
import argparse
import concurrent.futures

import numpy as np

from pyproj import Transformer
from pathlib import Path
from scipy.ndimage import label as nd_label  # Explicit import to avoid shadowing
from rasterio.transform import xy

from dataset.utils import (
    get_image_and_polygons,
    create_partial_segment_mask,
    create_hybrid_sdt_boundary_labels,
)

from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger


def pad_image(image, window_size):
    h, w = image.shape[:2]
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def pad_label(label, window_size):
    h, w = label.shape[:2]
    channels = 1 if label.ndim == 2 else label.shape[2]
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)

    return np.pad(
        label,
        ((0, pad_h), (0, pad_w), (0, 0)) if label.ndim == 3 else ((0, pad_h), (0, pad_w)),
        mode="constant",
    )


def extract_patches(image, label, window_size, stride, buffer=32):
    pad = (window_size - stride) // 2
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    label = np.pad(label, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    h, w = image.shape[:2]
    patches = []

    for y in range(pad, h - window_size - pad + 1, stride):
        for x in range(pad, w - window_size - pad + 1, stride):
            img_patch = image[y:y+window_size, x:x+window_size]
            lbl_patch = label[y:y+window_size, x:x+window_size]
            
            buffer_mask = create_partial_segment_mask(lbl_patch[..., 0])
            
            _, num_trees = nd_label(lbl_patch[:, :, 0])

            patches.append((img_patch, lbl_patch, buffer_mask, num_trees))
    
    return patches


def process_image(image_path, label_path, conf):
    logger = get_logger()

    image_name = os.path.basename(image_path)

    try:
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs  # CRS of the raster
            geographic_crs = "EPSG:4326"  # Geographic CRS (WGS84)

            transformer = Transformer.from_crs(crs, geographic_crs, always_xy=True)

        img_arr, polygons = get_image_and_polygons(
            image_path,
            label_path,
            conf.nir_rgb_order,
            conf.normalize_channelwise,
            conf.normalize_imagewise,
        )

        binary_mask, centroid_mask, hybrid_channel = create_hybrid_sdt_boundary_labels(img_arr, polygons)
        combined_mask = np.stack([binary_mask, centroid_mask, hybrid_channel], axis=-1)

        patches = extract_patches(img_arr, combined_mask, conf.window_size, conf.stride)

        labeled_patches = []
        h, w = img_arr.shape[:2]
        pad = (conf.window_size - conf.stride) // 2
        num_patches_per_row = ((w - conf.window_size) // conf.stride) + 1

        for idx, (img_patch, lbl_patch, buffer_mask, num_trees) in enumerate(patches):
            patch_row = idx // num_patches_per_row
            patch_col = idx % num_patches_per_row
            
            pixel_x = patch_col * conf.stride
            pixel_y = patch_row * conf.stride
            
            center_x = pixel_x + conf.window_size // 2
            center_y = pixel_y + conf.window_size // 2
            
            raster_lon, raster_lat = xy(transform, center_y, center_x)
            centroid_lon, centroid_lat = transformer.transform(raster_lon, raster_lat)

            labeled_patches.append(
                (
                    img_patch,          # Image patch with buffer context
                    {
                        'mask': lbl_patch[..., 0],        # Binary segmentation
                        'centroid': lbl_patch[..., 1],    # Gaussian centroids  
                        'hybrid': lbl_patch[..., 2],      # SDT-boundary channel
                        'buffer_mask': buffer_mask        # Critical for training
                    },
                    num_trees,          # Precomputed tree count (buffer-masked)
                    image_name,
                    centroid_lat,
                    centroid_lon,
                    (pixel_x, pixel_y)  # Original coordinates
                )
            )

        return image_name, labeled_patches

    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return image_name, []


def write_to_hdf5(hdf5_file, data):
    logger = get_logger()

    with h5py.File(hdf5_file, "a") as hf:
        for image_name, labeled_patches in data:
            if not labeled_patches:
                logger.warning(f"No labeled patches for image: {image_name}")
                continue
                
            for idx, patch in enumerate(labeled_patches):
                (img_patch, label_dict, num_trees, 
                 _, centroid_lat, centroid_lon, 
                 (pixel_x, pixel_y)) = patch
                
                group_name = f"{image_name}_{idx}"
                patch_group = hf.create_group(group_name)
                
                patch_group.create_dataset("image", data=img_patch, 
                                         compression="gzip", dtype=np.float32)
                
                label_group = patch_group.create_group("labels")
                label_group.create_dataset("mask", data=label_dict['mask'], 
                                         compression="gzip", dtype=np.float32)
                label_group.create_dataset("centroid", data=label_dict['centroid'], 
                                         compression="gzip", dtype=np.float32)
                label_group.create_dataset("hybrid", data=label_dict['hybrid'], 
                                         compression="gzip", dtype=np.float32)
                label_group.create_dataset("buffer_mask", data=label_dict['buffer_mask'], 
                                         compression="gzip", dtype=np.uint8)
                
                patch_group.attrs["num_trees"] = int(num_trees)
                patch_group.attrs["source_image"] = str(image_name)
                patch_group.attrs["latitude"] = float(centroid_lat)
                patch_group.attrs["longitude"] = float(centroid_lon)
                patch_group.attrs["pixel_x"] = int(pixel_x)
                patch_group.attrs["pixel_y"] = int(pixel_y)
            

def convert_to_hdf5(
    conf,
    no_of_samples=None,
    num_workers=4,
    chunk_size=10,
):
    logger = get_logger()

    data_path = Path(conf.data_folder)
    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file
 
    assert not os.path.exists(hdf5_path), f"[ERROR] The HDF5 file '{hdf5_path}' already exists. Please provide a different file name or delete the existing file."

    image_list = []
    label_list = []

    image_list = list(data_path.rglob("*.tiff")) + list(data_path.rglob("*.tif")) + list(data_path.rglob("*.jp2"))

    for image_path in image_list[:]:
        label_path = Path(str(image_path).replace("/Images/", "/Geojsons/"))
        label_path = label_path.with_suffix(".geojson")

        if os.path.exists(label_path):
            label_list.append(label_path)
        else:
            image_list.remove(image_path)
            logger.warning(f"Labels not found for {image_path}")

    if no_of_samples is not None:
        image_list = image_list[:no_of_samples]
        label_list = label_list[:no_of_samples]

    chunk_count = len(image_list) // chunk_size + int(len(image_list) % chunk_size != 0)

    for chunk_idx in range(chunk_count):
        chunk_files = list(
            zip(
                image_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size],
                label_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            )
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_image,
                    image,
                    label,
                    conf
                ): image
                for image, label in chunk_files
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:  # Ensure result is not None
                        results.append(result)
                except Exception as e:
                    logger.error(f"File {file} generated an exception: {e}")

            write_to_hdf5(hdf5_path, results)

        files_left = max(0, len(image_list) - (chunk_idx + 1) * chunk_size)
        logger.info(f"Completed chunk {chunk_idx + 1}/{chunk_count}. {files_left} files left to process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial images dataset creation.")
    parser.add_argument("config",          type=str, help="Path to the configuration file",)
    parser.add_argument("--no-of-samples", type=int, default=None, help="If set N>1, creates samples from N Aerial Images, for testing purposes only",)
    parser.add_argument("--num-workers",   type=int, default=4, help="Number of workers for parallel processing",)
    parser.add_argument("--chunk-size",    type=int, default=10, help="Number of images to process in a single chunk",)
    parser.add_argument('--verbosity',     type=str, default='info', choices=['info', 'debug', 'warning'])
    args = parser.parse_args()

    conf = setup(args.config)

    _ = configure_logger(verbosity=args.verbosity)

    convert_to_hdf5(
        conf,
        no_of_samples=args.no_of_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )

'''
Usage:

- Local

export TREEMORT_DATA_PATH="/Users/anisr/Documents/dead_trees" 

python3 -m dataset.creator ${TREEMORT_REPO_PATH}/configs/data/finland.txt

- For testing only

python3 -m dataset.creator ${TREEMORT_REPO_PATH}/configs/data/finland.txt --no-of-samples 3

- HPC

Usage: ./submit_creator.sh <hpc_type> <data config file>

Examples:

sh ./scripts/submit_creator.sh lumi finland

'''
