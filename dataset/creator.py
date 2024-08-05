import os
import shutil
import random
import logging
import argparse

import numpy as np

from preprocessutils import (
    get_image_and_polygons_reorder,
    get_image_and_polygons_normalize,
    segmap_to_topo,
)

# Configuration
square_save_size = 320

# nir_r_g_b_order only for highres sat images (otherwise set to None)
# SET UP NIR-R-G-B order for example: PHR [3,0,1,2] -- alueX SS_PS [3,2,1,0]
nir_r_g_b_order = [3,2,1,0]

normalize_imagewise = False  # each image_crop normalized separately to 0...255
normalize_channelwise = True  # each channel normalized separately to 0...255


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def create_dataset(project_folder, no_of_samples=None):
    output_images_folder = os.path.join(project_folder, "Temp", "Images")
    output_labels_folder = os.path.join(project_folder, "Temp", "Labels")

    for folder_path in [output_images_folder, output_labels_folder]:
        create_directory(folder_path)

    image_folder = os.path.join(project_folder, "4band_25cm")
    label_folder = os.path.join(project_folder, "Geojsons")

    file_list = os.listdir(image_folder)[:no_of_samples] if no_of_samples else os.listdir(image_folder)

    total_files_in_area, total_annotations_in_area, no_of_samples = 0, 0, 0

    for file in file_list:
        try:
            if file.endswith(("jp2", "tif", "tiff")):
                total_files_in_area += 1

                image_filepath = os.path.join(image_folder, file)
                geojson_filepath = os.path.join(label_folder, file.rsplit('.', 1)[0] + ".geojson")

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
                orig_h, orig_w = exim_np.shape[:2]

                divider_h = np.ceil(orig_h / square_save_size)
                divider_w = np.ceil(orig_w / square_save_size)

                sample_h = int(np.floor(orig_h / divider_h))
                sample_w = int(np.floor(orig_w / divider_w))

                file_stub = os.path.split(image_filepath)[1].split(".")[0]

                # split the image to smaller samples
                for x_tile in range(int(divider_w)):
                    for y_tile in range(int(divider_h)):
                        start_x, end_x = x_tile * sample_w, (x_tile + 1) * sample_w
                        start_y, end_y = y_tile * sample_h, (y_tile + 1) * sample_h

                        image_clip = exim_np[start_y:end_y, start_x:end_x, :]
                        topo_clip = topolabel[start_y:end_y, start_x:end_x]

                        res_image_path = os.path.join(output_images_folder, f"{file_stub}_{x_tile}_{y_tile}.npy")
                        res_label_path = os.path.join(output_labels_folder, f"{file_stub}_{x_tile}_{y_tile}.npy")

                        np.save(res_image_path, image_clip)
                        np.save(res_label_path, topo_clip)
                        no_of_samples += 1

        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")

    logging.info(f"{total_files_in_area} files processed")
    logging.info(f"{total_annotations_in_area} polygons found")
    logging.info(f"{no_of_samples} sample-squares created")

    return output_images_folder, output_labels_folder


def split_dataset(
    project_folder, output_images_folder, output_labels_folder, split_ratio=1
):
    train_images_folder = os.path.join(project_folder, "Train", "Images")
    train_labels_folder = os.path.join(project_folder, "Train", "Labels")
    test_images_folder  = os.path.join(project_folder, "Test", "Images")
    test_labels_folder  = os.path.join(project_folder, "Test", "Labels")

    for folder_path in [
        train_images_folder,
        train_labels_folder,
        test_images_folder,
        test_labels_folder,
    ]:
        create_directory(folder_path)

    all_files = os.listdir(output_images_folder)

    for file in all_files:
        src_image_path = os.path.join(output_images_folder, file)
        src_label_path = os.path.join(output_labels_folder, file)

        if random.randint(1, 10) == split_ratio:  # 10% test set
            dest_image_path = os.path.join(test_images_folder, file)
            dest_label_path = os.path.join(test_labels_folder, file)

        else:
            dest_image_path = os.path.join(train_images_folder, file)
            dest_label_path = os.path.join(train_labels_folder, file)

        shutil.move(src_image_path, dest_image_path)
        shutil.move(src_label_path, dest_label_path)

    logging.info("Created Train and Test folders")
    logging.info(f"Train Images: {train_images_folder}")
    logging.info(f"Train Labels: {train_labels_folder}")
    logging.info(f"Test Images: {test_images_folder}")
    logging.info(f"Test Labels: {test_labels_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial images dataset creation.")
    parser.add_argument("data_folder",     type=str,               help="Path to the raw Aerial Image data")
    parser.add_argument("--no-of-samples", type=int, default=None, help="If set N>1, creates samples from N Aerial Images, for testing purposes only")

    args = parser.parse_args()

    print(args)

    data_folder = args.data_folder

    image_folder = os.path.join(data_folder, "4band_25cm")
    label_folder = os.path.join(data_folder, "Geojsons")

    output_images_folder, output_labels_folder = create_dataset(data_folder, args.no_of_samples)
    split_dataset(data_folder, output_images_folder, output_labels_folder)

    logging.info("Removing Temp folder")
    shutil.rmtree(os.path.join(data_folder, "Temp"))

