import os
import torch
import argparse
import configargparse

import numpy as np

from pathlib import Path

from treemort.utils.config import setup
from treemort.utils.logger import get_logger
from treemort.modeling.builder import build_model

from inference.utils import (
    load_and_preprocess_image,
    threshold_prediction_map,
    contours_to_geojson,
    extract_contours,
    save_geojson,
    pad_image,
)

logger = get_logger(__name__)


def sliding_window_inference(model, image, window_size=256, stride=128, batch_size=8):
    model.eval()

    device = next(model.parameters()).device  # Get the device of the model

    padded_image = pad_image(image, window_size)

    _, h, w = padded_image.shape
    prediction_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patches = []
    coords = []

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = padded_image[:, y : y + window_size, x : x + window_size]
            patches.append(patch)
            coords.append((y, x))

            if len(patches) == batch_size:
                prediction_map, count_map = process_batch(patches, coords, prediction_map, count_map, model, device)
                patches = []
                coords = []

    if patches:
        prediction_map, count_map = process_batch(patches, coords, prediction_map, count_map, model, device)

    count_map[count_map == 0] = 1  # Avoid division by zero
    prediction_map /= count_map
    #prediction_map = np.maximum(prediction_map, count_map)

    return prediction_map


def process_batch(patches, coords, prediction_map, count_map, model, device):
    batch_tensor = torch.from_numpy(np.array(patches)).float().to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        predictions = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

    for i, (y, x) in enumerate(coords):
        confidence = predictions[i]
        mask = (confidence >= 0.5).astype(np.float32)
        prediction_map[y : y + confidence.shape[0], x : x + confidence.shape[1]] += confidence
        count_map[y : y + confidence.shape[0], x : x + confidence.shape[1]] += mask

    return prediction_map, count_map


def load_model(config_path, best_model, id2label):
    logger.info("Loading model configuration...")
    conf = setup(config_path)
    logger.info("Model configuration loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading or resuming model...")
    model, _, _, _ = build_model(conf, id2label, device)
    model = model.to(device)
    logger.info("Model, optimizer, criterion, and metrics are set up.")

    model.load_state_dict(torch.load(best_model, map_location=device, weights_only=True))
    logger.info(f"Loaded weights from {best_model}.")

    return model


def process_image(
    model,
    image_path,
    geojson_path,
    window_size=256,
    stride=128,
    threshold=0.5,
    nir_rgb_order=[3, 2, 1, 0],
):
    logger.info(f"Starting process for image: {image_path}")

    image, transform, crs = load_and_preprocess_image(image_path, nir_rgb_order)
    logger.info(f"Image loaded and preprocessed. Shape: {image.shape}, Transform: {transform}")

    prediction_map = sliding_window_inference(model, image, window_size, stride)
    logger.info(f"Prediction map generated with shape: {prediction_map.shape}")

    binary_mask = threshold_prediction_map(prediction_map, threshold)
    logger.info(f"Binary mask created with threshold: {threshold}. Mask shape: {binary_mask.shape}")

    contours = extract_contours(binary_mask)
    logger.info(f"{len(contours)} contours extracted from binary mask")

    geojson_data = contours_to_geojson(contours, transform, crs, os.path.splitext(os.path.basename(image_path))[0])
    logger.info("Contours converted to GeoJSON format")

    save_geojson(geojson_data, geojson_path)
    logger.info(f"GeoJSON saved to {geojson_path}")


def parse_config(config_file_path):
    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("--model-config",  type=str, required=True, help="path to model configuration")
    parser.add("--best-model",    type=str, required=True, help="path to best model weights")
    parser.add("--window-size",   type=int,   default=256, help="size of the window")
    parser.add("--stride",        type=int,   default=128, help="stride for the window")
    parser.add("--threshold",     type=float, default=0.5, help="threshold for the prediction")
    parser.add("--nir-rgb-order", type=int, nargs='+', default=[3, 2, 1, 0],   help="NIR, R, G, B order")
    
    conf, _ = parser.parse_known_args()

    return conf


def run_inference(data_path, config_file_path, output_dir):
    id2label = {0: "alive", 1: "dead"}
    conf = parse_config(config_file_path)
    model = load_model(conf.model_config, conf.best_model, id2label)

    data_path = Path(data_path)

    if data_path.is_dir():
        logger.info(f"Processing all images in folder: {data_path}")
        image_paths = list(data_path.rglob("*.tiff")) + list(data_path.rglob("*.tif")) + list(data_path.rglob("*.jp2"))
        if not image_paths:
            logger.error(f"No images found in directory or its subdirectories: {data_path}")
            return
        logger.info(f"Found {len(image_paths)} images.")

    elif data_path.is_file():
        logger.info(f"Processing single file: {data_path}")
        if data_path.suffix.lower() in [".tiff", ".tif"]:
            image_paths = [data_path]
        else:
            with open(data_path, "r") as file:
                image_paths = [Path(line.strip()) for line in file.readlines() if line.strip()]
            if not image_paths:
                logger.error(f"No valid image paths found in file: {data_path}")
                return
    else:
        logger.error(f"Invalid input path: {data_path}")
        return

    for image_path in image_paths:
        try:
            if output_dir:
                geojson_path = os.path.join(output_dir, os.path.basename(os.path.splitext(image_path)[0] + ".geojson"))
            else:
                geojson_path = str(image_path).replace("/Images/", "/Predictions/")
                geojson_path = os.path.splitext(geojson_path)[0] + ".geojson"

            directory = os.path.dirname(geojson_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created predictions directory: {directory}")

            process_image(model, image_path, geojson_path, window_size=conf.window_size, stride=conf.stride, threshold=conf.threshold, nir_rgb_order=conf.nir_rgb_order)
            logger.info(f"Processed image saved to: {geojson_path}")

        except Exception as e:
            logger.error(f"Failed to process image: {image_path}. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")

    parser.add_argument('data_path', type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config',  type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument('--outdir',  type=str, help="Directory to save GeoJSON predictions (default: same as input)")

    args = parser.parse_args()

    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    run_inference(args.data_path, args.config, args.outdir)


if __name__ == "__main__":
    main()

''' Usage:

- For single file:

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt

2) save geojsons to an output folder

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Predictions

- For entire folder

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt

2) save geojsons to output folder

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir /Users/anisr/Documents/dead_trees/Finland/Predictions

- Run viewer api service

uvicorn treemort_api:app --reload

- Run viewer application

streamlit run treemort_app.py

'''