import os
import gc
import torch
import psutil
import argparse
import configargparse

import numpy as np

from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from memory_profiler import profile

from treemort.utils.config import setup
from treemort.utils.logger import get_logger
from treemort.modeling.builder import build_model

from inference.utils import (
    pad_image,
    save_labels_as_geojson,
    load_and_preprocess_image,
    generate_watershed_labels,
    threshold_prediction_map,
    extract_contours,
    contours_to_geojson,
    save_geojson
)
from misc.graph_partition import perform_graph_partitioning

logger = get_logger(__name__)


def log_resource_usage(stage):
    mem = psutil.virtual_memory()
    logger.info(f"[{stage}] CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {mem.percent}%")
    logger.info(f"[{stage}] Available Memory: {mem.available / (1024**3):.2f} GB")
    
    swap = psutil.swap_memory()
    logger.info(f"[{stage}] Swap Usage: {swap.used / (1024**3):.2f} GB")


def sliding_window_inference(model, image, window_size=256, stride=128, batch_size=1, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device

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
                prediction_map, count_map = process_batch(patches, coords, prediction_map, count_map, model, threshold, device)
                patches = []
                coords = []

    if patches:
        prediction_map, count_map = process_batch(patches, coords, prediction_map, count_map, model, threshold, device)

    del patches, coords
    torch.cuda.empty_cache()  # If using GPU, ensure cache is cleared
    gc.collect()  # Force garbage collection

    no_contribution_mask = (count_map == 0)

    count_map[count_map == 0] = 1

    final_prediction = prediction_map / count_map
    final_prediction[no_contribution_mask] = 0
    final_prediction = np.clip(final_prediction, 0, 1)

    _, original_h, original_w = image.shape
    return final_prediction[:original_h, :original_w]


def process_batch(patches, coords, prediction_map, count_map, model, threshold, device):
    try:
        batch_tensor = torch.from_numpy(np.array(patches)).float().to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            predictions = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        for i, (y, x) in enumerate(coords):
            confidence = predictions[i]
            mask = (confidence >= threshold).astype(np.float32)
            prediction_map[y : y + confidence.shape[0], x : x + confidence.shape[1]] += confidence
            count_map[y : y + confidence.shape[0], x : x + confidence.shape[1]] += mask

        del batch_tensor, patches, coords, predictions, confidence, mask
        gc.collect()  # Force garbage collection
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise
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
    logger.info("Model loaded onto device.")

    model.load_state_dict(torch.load(best_model, map_location=device, weights_only=True))
    logger.info(f"Loaded weights from {best_model}.")

    return model


def process_image(model, image_path, geojson_path, conf, post_process):
    try:
        logger.info(f"Starting process for image: {image_path}")

        image, transform, crs = load_and_preprocess_image(image_path, conf.nir_rgb_order)
        logger.info(f"Image loaded and preprocessed. Shape: {image.shape}")

        prediction_map = sliding_window_inference(model, image, window_size=conf.window_size, stride=conf.stride, threshold=conf.threshold)
        logger.info(f"Prediction map generated with shape: {prediction_map.shape}")

        predicted_mask = threshold_prediction_map(prediction_map, conf.threshold)
        logger.info(f"Mask created with shape: {predicted_mask.shape}")

        if post_process:    
            partitioned_labels = perform_graph_partitioning(image, predicted_mask)
            logger.info(f"Partition graph segmentation completed with max label: {np.max(partitioned_labels)}")
            
            refined_labels = generate_watershed_labels(prediction_map, partitioned_labels, conf.min_distance, conf.blur_sigma, conf.dilation_radius)
            logger.info(f"Watershed segmentation completed with max label: {np.max(refined_labels)}")

            if refined_labels is not None:
                save_labels_as_geojson(
                    refined_labels,
                    transform,
                    crs,
                    geojson_path,
                    min_area_threshold=conf.min_area_threshold,
                    max_aspect_ratio=conf.max_aspect_ratio,
                    min_solidity=conf.min_solidity
                )
                logger.info(f"GeoJSON saved to {geojson_path}")

        else:
            contours = extract_contours(predicted_mask)
            logger.info(f"{len(contours)} contours extracted from binary mask")

            geojson_data = contours_to_geojson(contours, transform, crs, os.path.splitext(os.path.basename(image_path))[0])
            logger.info("Contours converted to GeoJSON format")

            save_geojson(geojson_data, geojson_path)
            logger.info(f"GeoJSON saved to {geojson_path}")
        
        del prediction_map, predicted_mask
        gc.collect()
    except Exception as e:
        logger.error(f"Error processing image: {image_path}. Error: {e}")
        raise
    

def parse_config(config_file_path):
    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("--model-config",       type=str, required=True,  help="path to model configuration")
    parser.add("--best-model",         type=str, required=True,  help="path to best model weights")
    parser.add("--window-size",        type=int,   default=256,  help="size of the window")
    parser.add("--stride",             type=int,   default=128,  help="stride for the window")
    parser.add("--threshold",          type=float, default=0.5,  help="threshold for the prediction")
    parser.add("--min-area-threshold", type=float, default=1.0,  help="threshold for the prediction")
    parser.add("--max-aspect-ratio",   type=float, default=3.0,  help="threshold for the prediction")
    parser.add("--min-solidity",       type=float, default=0.85, help="threshold for the prediction")
    parser.add("--min-distance",       type=int,   default=7,    help="threshold for the prediction")
    parser.add("--dilation-radius",    type=int,   default=0,    help="threshold for the prediction")
    parser.add("--blur-sigma",         type=float, default=1.0,  help="threshold for the prediction")
    parser.add("--nir-rgb-order",      type=int, nargs='+', default=[3, 0, 1, 2],   help="NIR, R, G, B order")
    
    conf, _ = parser.parse_known_args()

    return conf


def process_single_image(image_path, conf, output_dir, id2label, post_process):
    try:
        model = load_model(conf.model_config, conf.best_model, id2label)

        if output_dir:
            geojson_path = os.path.join(output_dir, os.path.basename(os.path.splitext(image_path)[0] + ".geojson"))
        else:
            geojson_path = str(image_path).replace("/Images/", "/Predictions/")
            geojson_path = os.path.splitext(geojson_path)[0] + ".geojson"

        directory = os.path.dirname(geojson_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        process_image(model, image_path, geojson_path, conf, post_process)

        logger.info(f"Processed image saved to: {geojson_path}")
    except Exception as e:
        logger.error(f"Failed to process image: {image_path}. Error: {e}")


def run_inference(data_path, config_file_path, output_dir, post_process):
    id2label = {0: "alive", 1: "dead"}
    conf = parse_config(config_file_path)

    data_path = Path(data_path)

    if data_path.is_dir():
        image_paths = list(data_path.rglob("*.tiff")) + list(data_path.rglob("*.tif")) + list(data_path.rglob("*.jp2"))
    elif data_path.is_file():
        image_paths = [data_path]
    else:
        logger.error(f"Invalid input path: {data_path}")
        return

    tasks = [(image_path, conf, output_dir, id2label, post_process) for image_path in image_paths]

    logger.info(f"Found {len(image_paths)} images to process.")

    num_processes = min(4, cpu_count())  # Use up to 4 processes or as many as available
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_single_image, tasks)


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")

    parser.add_argument('data_path', type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config',  type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument('--outdir',  type=str, help="Directory to save GeoJSON predictions (default: same as input)")
    parser.add_argument('--post-process', action="store_true", help="Enable or disable post-processing")

    args = parser.parse_args()

    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    
    print(args)
    
    run_inference(args.data_path, args.config, args.outdir, args.post_process)


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        logger.error(f"MemoryError: {e}")
    except Exception as e:
        logger.error(f"Critical error: {e}")


''' Usage:

- For single file:

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/copenhagen_data/Images/patches_3095_377.tif \
    --config ./configs/USA_RGBNIR_60cm_inference.txt

2) save geojsons to an output folder

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Predictions
    --post-process

- For entire folder

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/copenhagen_data \
    --config ./configs/USA_RGBNIR_60cm_inference.txt

2) save geojsons to output folder

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir /Users/anisr/Documents/dead_trees/Finland/Predictions \
    --post-process

/Users/anisr/Documents/USA \
    --config ./configs/USA_RGBNIR_60cm_inference.txt \
    --outdir /Users/anisr/Documents/USA/Predictions \
    --post-process

- Run viewer api service

uvicorn treemort_api:app --reload

- Run viewer application

streamlit run treemort_app.py

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
export TREEMORT_DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees"

sbatch --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/Finland_RGBNIR_25cm.txt",DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm" $TREEMORT_REPO_PATH/scripts/run_inference.sh

'''