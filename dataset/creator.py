import os
import h5py
import rasterio
import argparse
import configargparse
import concurrent.futures

import numpy as np

from pyproj import Transformer
from pathlib import Path
from rasterio.transform import xy

from dataset.preprocessutils import (
    get_image_and_polygons,
    create_label_mask_with_centroids,
)


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


def extract_patches(image, label, window_size, stride):
    image = pad_image(image, window_size)
    label = pad_label(label, window_size)
    h, w = image.shape[:2]

    patches = []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            image_patch = image[y : y + window_size, x : x + window_size]
            
            label_patch = label[y : y + window_size, x : x + window_size]
            label_patch[:, :, 0] = (label_patch[:, :, 0] > 0).astype(np.float32)
            label_patch[:, :, 1] = (label_patch[:, :, 1] > 0).astype(np.float32)

            patches.append((image_patch, label_patch))
    return patches


def process_image(image_path, label_path, conf):
    import rasterio
    import numpy as np

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

        label_mask, centroid_mask = create_label_mask_with_centroids(img_arr, polygons)
        combined_mask = np.stack([label_mask, centroid_mask], axis=-1)

        patches = extract_patches(img_arr, combined_mask, conf.window_size, conf.stride)

        labeled_patches = []
        for idx, patch in enumerate(patches):
            patch_row, patch_col = idx // (img_arr.shape[1] // conf.stride), idx % (img_arr.shape[1] // conf.stride)
            pixel_x, pixel_y = patch_col * conf.stride, patch_row * conf.stride
            pixel_centroid_x = pixel_x + conf.window_size // 2
            pixel_centroid_y = pixel_y + conf.window_size // 2

            raster_lon, raster_lat = xy(transform, pixel_centroid_y, pixel_centroid_x)
            centroid_lon, centroid_lat = transformer.transform(raster_lon, raster_lat)

            dead_tree_count = int(np.sum(patch[1][:, :, 1]))

            labeled_patches.append(
                (
                    patch[0],              # Image patch
                    patch[1],              # Combined label patch
                    dead_tree_count,       # Number of dead tree segments
                    image_name,            # Source image name
                    centroid_lat,          # Centroid latitude
                    centroid_lon           # Centroid longitude
                )
            )

        return image_name, labeled_patches

    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return image_name, []
    

def write_to_hdf5(hdf5_file, data):
    with h5py.File(hdf5_file, "a") as hf:
        if data:
            for file_stub, labeled_patches in data:
                if labeled_patches:
                    for idx, (image_patch, label_patch, dead_tree_count, filename, lat, lon) in enumerate(labeled_patches):
                        key = f"{file_stub}_{idx}"
                        
                        patch_group = hf.create_group(key)
                        
                        patch_group.create_dataset("image", data=image_patch, compression="gzip")
                        patch_group.create_dataset("label", data=label_patch, compression="gzip")
                        
                        patch_group.attrs["dead_tree_count"] = dead_tree_count
                        patch_group.attrs["source_image"] = filename
                        patch_group.attrs["latitude"] = lat
                        patch_group.attrs["longitude"] = lon
                else:
                    print(f"[WARNING] No labeled patches for file: {file_stub}")
        else:
            print("[WARNING] No data provided to write.")
            

def convert_to_hdf5(
    conf,
    no_of_samples=None,
    num_workers=4,
    chunk_size=10,
):
    data_path = Path(conf.data_folder)
    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file
    
    assert not os.path.exists(hdf5_path), f"[ERROR] The HDF5 file '{hdf5_path}' already exists. Please provide a different file name or delete the existing file."

    image_list = []
    label_list = []

    image_list = list(data_path.rglob("*.tiff")) + list(data_path.rglob("*.tif")) + list(data_path.rglob("*.jp2"))

    for image_path in image_list[:]:  # Using image_list[:] to make a copy for safe removal
        label_path = Path(str(image_path).replace("/Images/", "/Geojsons/"))
        label_path = label_path.with_suffix(".geojson")

        if os.path.exists(label_path):
            label_list.append(label_path)
        else:
            image_list.remove(image_path)
            print(f"[WARNING] Labels not found for {image_path}")

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
                    print(f"[ERROR] File {file} generated an exception: {e}")

            write_to_hdf5(hdf5_path, results)

        files_left = max(0, len(image_list) - (chunk_idx + 1) * chunk_size)
        print(f"[INFO] Completed chunk {chunk_idx + 1}/{chunk_count}. {files_left} files left to process.")


def parse_config(config_file_path):
    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("--data-folder",             type=str, required=True,    help="directory with aerial image and label data")
    parser.add("--hdf5-file",               type=str, required=True,    help="name of output hdf5 file")
    parser.add("--num-workers",             type=int, default=4,        help="number of workers for parallel processing")
    parser.add("--window-size",             type=int, default=256,      help="size of the window")
    parser.add("--stride",                  type=int, default=128,      help="stride for the window")
    parser.add("--nir-rgb-order",           type=int, nargs='+', default=[3, 0, 1, 2],   help="NIR, R, G, B order")
    parser.add("--normalize-imagewise",     action="store_true",        help="normalize imagewise")
    parser.add("--normalize-channelwise",   action="store_true",        help="normalize channelwise")

    conf, _ = parser.parse_known_args()

    return conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial images dataset creation.")
    parser.add_argument("config",           type=str,               help="Path to the configuration file",)
    parser.add_argument("--no-of-samples",  type=int, default=None, help="If set N>1, creates samples from N Aerial Images, for testing purposes only",)
    parser.add_argument("--num-workers",    type=int, default=4,    help="Number of workers for parallel processing",)
    parser.add_argument("--chunk-size",     type=int, default=10,   help="Number of images to process in a single chunk",)
    args = parser.parse_args()

    conf = parse_config(args.config)

    # conf.data_folder = "/Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm"

    convert_to_hdf5(
        conf,
        no_of_samples=args.no_of_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )

'''
Usage:

python3 -m dataset.creator ./configs/Finland_RGBNIR_25cm.txt

- For testing only

python3 -m dataset.creator ./configs/Finland_RGBNIR_25cm.txt --no-of-samples 3

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

sbatch --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/Finland_RGBNIR_25cm.txt" $TREEMORT_REPO_PATH/scripts/run_creator.sh

'''
