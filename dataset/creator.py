import os
import h5py
import argparse
import configargparse
import concurrent.futures

import numpy as np

from pathlib import Path

from dataset.preprocessutils import (
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

    h, w = image.shape[:2]

    patches = []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            image_patch = image[y : y + window_size, x : x + window_size]
            
            label_patch = label[y : y + window_size, x : x + window_size]
            label_patch = (label_patch > 0).astype(np.float32)

            patches.append((image_patch, label_patch))
    return patches


def process_image(
    image_path,
    label_path,
    conf
):
    image_name = os.path.basename(image_path)

    try:
        if conf.nir_rgb_order is not None:
            exim_np, adjusted_polygons = get_image_and_polygons_reorder(
                image_path,
                label_path,
                conf.nir_rgb_order,
                conf.normalize_channelwise,
                conf.normalize_imagewise,
            )
        else:
            exim_np, adjusted_polygons = get_image_and_polygons_normalize(
                image_path,
                label_path,
                conf.normalize_channelwise,
                conf.normalize_imagewise,
            )

        exim_np = exim_np[:, :, 1:] # Remove NIR

        topolabel = segmap_to_topo(exim_np, adjusted_polygons)
        patches = extract_patches(exim_np, topolabel, conf.window_size, conf.stride)

        labeled_patches = [(patch[0], patch[1], int(np.any(patch[1])), image_name) for patch in patches]
        return image_name, labeled_patches
    
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return image_name, []


def write_to_hdf5(hdf5_file, data):
    with h5py.File(hdf5_file, "a") as hf:  # Open in append mode
        if data:  # Ensure data is not empty
            for file_stub, labeled_patches in data:
                if labeled_patches:  # Only process if labeled_patches is not empty
                    for idx, (image_patch, label_patch, contains_dead_tree, filename) in enumerate(labeled_patches):
                        key = f"{file_stub}_{idx}"
                        hf.create_group(key)
                        hf[key].create_dataset("image", data=image_patch, compression="gzip")
                        hf[key].create_dataset("label", data=label_patch, compression="gzip")
                        hf[key].attrs["contains_dead_tree"] = contains_dead_tree
                        hf[key].attrs["source_image"] = filename
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

    conf = parse_config(args.config)

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
