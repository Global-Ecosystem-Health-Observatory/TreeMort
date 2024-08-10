import os
import h5py
import argparse
import configargparse
import concurrent.futures

import numpy as np

from preprocessutils import (
    get_image_and_polygons_reorder,
    get_image_and_polygons_normalize,
    segmap_to_topo,
)


def pad_image(image, window_size):
    h, w = image.shape[:2]
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def pad_label(label, window_size):
    h, w = label.shape[:2]
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    return np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant")


def extract_patches(image, label, window_size, stride):
    image = pad_image(image, window_size)
    label = pad_label(label, window_size)

    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            image_patch = image[y : y + window_size, x : x + window_size]
            label_patch = label[y : y + window_size, x : x + window_size]

            # Convert label_patch to binary mask
            label_patch = (label_patch > 0).astype(np.float32)

            patches.append((image_patch, label_patch))

    return patches


def process_image(
    file,
    image_folder,
    label_folder,
    nir_r_g_b_order,
    normalize_channelwise,
    normalize_imagewise,
    window_size,
    stride,
):
    try:
        if file.endswith(("jp2", "tif", "tiff")):
            image_filepath = os.path.join(image_folder, file)
            geojson_filepath = os.path.join(label_folder, file.rsplit(".", 1)[0] + ".geojson")

            if nir_r_g_b_order is not None:
                exim_np, adjusted_polygons = get_image_and_polygons_reorder(
                    image_filepath,
                    geojson_filepath,
                    nir_r_g_b_order,
                    normalize_channelwise,
                    normalize_imagewise,
                )
            else:
                exim_np, adjusted_polygons = get_image_and_polygons_normalize(
                    image_filepath,
                    geojson_filepath,
                    normalize_channelwise,
                    normalize_imagewise,
                )

            topolabel = segmap_to_topo(exim_np, adjusted_polygons)

            patches = extract_patches(exim_np, topolabel, window_size, stride)

            # Create labeled patches with binary classification (1 if contains dead trees, else 0)
            labeled_patches = [(patch[0], patch[1], int(np.any(patch[1])), file) for patch in patches]

            return file, labeled_patches
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")
        return file, []


def write_to_hdf5(hdf5_file, data):
    with h5py.File(hdf5_file, "a") as hf:  # Open in append mode
        for file_stub, labeled_patches in data:
            for idx, (image_patch, label_patch, contains_dead_tree, filename) in enumerate(labeled_patches):
                key = f"{file_stub}_{idx}"
                hf.create_group(key)
                hf[key].create_dataset("image", data=image_patch, compression="gzip")
                hf[key].create_dataset("label", data=label_patch, compression="gzip")
                hf[key].attrs["contains_dead_tree"] = contains_dead_tree
                hf[key].attrs["source_image"] = filename

def convert_to_hdf5(
    conf,
    no_of_samples=None,
    num_workers=4,
    chunk_size=10,
):
    image_folder = os.path.join(conf.data_folder, conf.image_folder)
    label_folder = os.path.join(conf.data_folder, conf.label_folder)
    hdf5_file = os.path.join(conf.data_folder, conf.hdf5_file)
    
    assert not os.path.exists(hdf5_file), f"[ERROR] The HDF5 file '{hdf5_file}' already exists. Please provide a different file name or delete the existing file."

    file_list = (
        os.listdir(image_folder)[:no_of_samples]
        if no_of_samples
        else os.listdir(image_folder)
    )
    chunk_count = len(file_list) // chunk_size + int(len(file_list) % chunk_size != 0)

    for chunk_idx in range(chunk_count):
        chunk_files = file_list[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_image,
                    file,
                    image_folder,
                    label_folder,
                    conf.nir_rgb_order,
                    conf.normalize_channelwise,
                    conf.normalize_imagewise,
                    conf.window_size,
                    conf.stride,
                ): file
                for file in chunk_files
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[ERROR] File {file} generated an exception: {e}")

            write_to_hdf5(hdf5_file, results)

        files_left = max(0, len(file_list) - (chunk_idx + 1) * chunk_size)
        print(f"[INFO] Completed chunk {chunk_idx + 1}/{chunk_count}. {files_left} files left to process.")

def parse_config(config_file_path):
    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("--data-folder",             type=str, required=True,    help="directory with aerial image and label data")
    parser.add("--image-folder",            type=str, required=True,    help="name of image directory")
    parser.add("--label-folder",            type=str, required=True,    help="name of label directory")
    parser.add("--hdf5-file",               type=str, required=True,    help="name of output hdf5 file")
    parser.add("--num-workers",             type=int, default=4,        help="number of workers for parallel processing")
    parser.add("--window-size",             type=int, default=256,      help="size of the window")
    parser.add("--stride",                  type=int, default=128,      help="stride for the window")
    parser.add("--nir-rgb-order",           type=int, nargs='+', default=[3, 2, 1, 0],   help="NIR, R, G, B order")
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

    # Load configuration
    conf = parse_config(args.config)

    convert_to_hdf5(
        conf,
        no_of_samples=args.no_of_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
