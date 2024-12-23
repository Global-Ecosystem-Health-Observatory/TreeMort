import os

from typing import Tuple, List, Dict


def find_file_pairs(
    data_folder: str,
    predictions_folder: str = None,
    image_dir_name: str = "Images",
    geojsons_dir_name: str = "Geojsons",
    predictions_dir_name: str = "Predictions",
    file_ext: str = ".geojson",
) -> List[Tuple[str, str, str]]:
    pairs = []
    pred_folder = predictions_folder if predictions_folder else os.path.join(data_folder, predictions_dir_name)

    for root, dirs, _ in os.walk(data_folder):
        if {image_dir_name, geojsons_dir_name}.issubset(dirs):
            image_files = {
                os.path.splitext(f)[0]: os.path.join(root, image_dir_name, f)
                for f in os.listdir(os.path.join(root, image_dir_name))
                if f.endswith(".tiff") or f.endswith(".tif")  # Assuming images are in TIFF format
            }
            gt_files = {
                os.path.splitext(f)[0]: os.path.join(root, geojsons_dir_name, f)
                for f in os.listdir(os.path.join(root, geojsons_dir_name))
                if f.endswith(file_ext)
            }
            pred_files = {
                os.path.splitext(f)[0]: os.path.join(pred_folder, f)
                for f in os.listdir(pred_folder)
                if f.endswith(file_ext)
            }
            # Match files by name
            common_files = image_files.keys() & gt_files.keys() & pred_files.keys()
            for fname in common_files:
                pairs.append((image_files[fname], gt_files[fname], pred_files[fname]))
    return pairs
