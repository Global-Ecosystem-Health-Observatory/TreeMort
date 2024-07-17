import os
import cv2
import json
import shutil
import random
import argparse
import rasterio

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from preprocessutils import (
    get_image_and_polygons_reorder,
    get_image_and_polygons_normalize,
    segmap_to_topo,
)


square_save_size = 320

# nir_r_g_b_order only for highres sat images (otherwise set to None)
nir_r_g_b_order = [3, 2, 1, 0]  # [3,0,1,2] # SET UP NIR-R-G-B order for example: PHR [3,0,1,2] -- alueX SS_PS [3,2,1,0]

normalize_imagewise = False  # each image_crop normalized separately to 0...255
normalize_channelwise = True  # each channel normalized separately to 0...255


""" CREATE the samples to temp folder"""


def create_dataset(project_folder, no_of_samples=None):
    output_images_folder = os.path.join(project_folder, "Temp", "Images")
    output_labels_folder = os.path.join(project_folder, "Temp", "Labels")

    for folder_path in [output_images_folder, output_labels_folder]:
        os.makedirs(folder_path, exist_ok=True)

    if no_of_samples is None:
        filut = os.listdir(image_folder)
    else:
        filut = os.listdir(image_folder)[:no_of_samples]

    total_files_in_area, total_annotations_in_area, no_of_samples = 0, 0, 0

    for filu in filut:
        try:
            if filu.endswith("jp2") or filu.endswith("tif") or filu.endswith("tiff"):
                total_files_in_area += 1
                image_filepath = os.path.join(image_folder, filu)
                geojson_filepath = os.path.join(
                    label_folder,
                    filu.replace(".jp2", ".geojson")
                    .replace(".tiff", ".geojson")
                    .replace(".tif", ".geojson"),
                )
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
                total_annotations_in_area += len(adjusted_polygons)
                topolabel = segmap_to_topo(exim_np, adjusted_polygons)
                # save images as close to 400 px images as evenly possible
                orig_h = exim_np.shape[0]
                orig_w = exim_np.shape[1]
                divider_h = np.ceil(orig_h / square_save_size)
                divider_w = np.ceil(orig_w / square_save_size)
                sample_h = int(np.floor(orig_h / divider_h))
                sample_w = int(np.floor(orig_w / divider_w))
                filestub = os.path.split(image_filepath)[1].split(".")[0]
                # split the image to smaller samples
                for x_tile in range(int(divider_w)):
                    for y_tile in range(int(divider_h)):
                        start_x = x_tile * sample_w
                        end_x = start_x + sample_w
                        start_y = y_tile * sample_h
                        end_y = start_y + sample_h
                        image_clip = exim_np[start_y:end_y, start_x:end_x, :]
                        topo_clip = topolabel[start_y:end_y, start_x:end_x]
                        # if np.max(topo_clip>0):
                        res_image_path = os.path.join(
                            output_images_folder,
                            (filestub + "_" + str(x_tile) + "_" + str(y_tile) + ".npy"),
                        )
                        res_label_path = os.path.join(
                            output_labels_folder,
                            (filestub + "_" + str(x_tile) + "_" + str(y_tile) + ".npy"),
                        )
                        np.save(res_image_path, image_clip)
                        np.save(res_label_path, topo_clip)
                        no_of_samples += 1
        except:
            print(f"failed to process {filu}")

    print(f"{total_files_in_area}  files")
    print(f"{total_annotations_in_area} polygons")
    print(f"{no_of_samples} sample-squares")

    return output_images_folder, output_labels_folder


""" Split Train / Test data """


def split_dataset(
    project_folder, output_images_folder, output_labels_folder, split_ratio=1
):
    train_images_folder = os.path.join(project_folder, "Train", "Images")
    train_labels_folder = os.path.join(project_folder, "Train", "Labels")
    test_images_folder = os.path.join(project_folder, "Test", "Images")
    test_labels_folder = os.path.join(project_folder, "Test", "Labels")

    for folder_path in [
        train_images_folder,
        train_labels_folder,
        test_images_folder,
        test_labels_folder,
    ]:
        os.makedirs(folder_path, exist_ok=True)

    all_files = os.listdir(output_images_folder)
    # move 10% to test others to train
    for filu in all_files:
        if random.randint(1, 10) == split_ratio:  # 10% test set
            shutil.move(
                os.path.join(output_images_folder, filu),
                os.path.join(test_images_folder, filu),
            )
            shutil.move(
                os.path.join(output_labels_folder, filu),
                os.path.join(test_labels_folder, filu),
            )
        else:
            shutil.move(
                os.path.join(output_images_folder, filu),
                os.path.join(train_images_folder, filu),
            )
            shutil.move(
                os.path.join(output_labels_folder, filu),
                os.path.join(train_labels_folder, filu),
            )

    print(f"Created Train and Test folders")
    print(f"Train Images: {train_images_folder}")
    print(f"Train Labels: {train_labels_folder}")
    print(f"Test Images: {test_images_folder}")
    print(f"Test Labels: {test_labels_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial images dataset creation.")
    parser.add_argument("data_folder",      type=str,               help="Path to the raw Aerial Image data")
    parser.add_argument("--no-of-samples",  type=int, default=None, help="If set N>1, creates samples from N Aerial Images, for testing purposes only")
    
    args = parser.parse_args()

    print(args)

    data_folder = args.data_folder

    image_folder = os.path.join(data_folder, "4band_25cm")
    label_folder = os.path.join(data_folder, "Geojsons")

    output_images_folder, output_labels_folder = create_dataset(data_folder, args.no_of_samples)
    split_dataset(data_folder, output_images_folder, output_labels_folder)

    print("Removing Temp folder")
    shutil.rmtree(os.path.join(data_folder, "Temp"))

#python3 ./dataset/creator.py /Users/anisr/Documents/AerialImages --no-of-samples 2