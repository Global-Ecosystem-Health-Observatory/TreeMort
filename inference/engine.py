import os
import torch
import argparse
import configargparse

from pathlib import Path
from multiprocessing import Pool, cpu_count

from treemort.utils.logger import configure_logger, get_logger

from inference.utils import (
    load_model,
    load_refine_model,
    sliding_window_inference,
    refine_mask,
    initialize_logger,
    save_labels_as_geojson,
    load_and_preprocess_image,
    generate_watershed_labels,
    threshold_prediction_map,
    extract_contours,
    contours_to_geojson,
    save_geojson,
    log_and_raise,
    validate_path,
)
from inference.graph_partition import perform_graph_partitioning


def process_image(
    model: torch.nn.Module,
    refine_model: torch.nn.Module,
    image_path: str,
    geojson_path: str,
    conf: object,
    post_process: bool,
) -> None:
    logger = get_logger()

    logger.debug(f"Processing image: {os.path.basename(image_path)}")

    try:
        logger.debug('Z2')
        device = next(model.parameters()).device
        logger.debug('A1')
        image, transform, crs = load_and_preprocess_image(image_path, conf.nir_rgb_order)
        logger.debug(f"Loaded and preprocessed image: {os.path.basename(image_path)}")
        logger.debug('A2')
        prediction_maps = sliding_window_inference(model, image, window_size=conf.window_size, stride=conf.stride, threshold=conf.threshold)
        segment_map, centroid_map = prediction_maps

        if post_process:
            logger.debug('A')
            refined_mask = refine_mask(segment_map, refine_model, device)
            logger.debug('B')
            partitioned_labels = perform_graph_partitioning(image, refined_mask)

            logger.debug('C')
            refined_labels = generate_watershed_labels(
                segment_map.cpu().numpy(),
                partitioned_labels,
                centroid_map=centroid_map.cpu().numpy(),
                min_distance=conf.min_distance,
                blur_sigma=conf.blur_sigma,
                dilation_radius=conf.dilation_radius,
                centroid_threshold=conf.centroid_threshold,
            )
            logger.debug('D')
            save_labels_as_geojson(
                refined_labels,
                transform,
                crs,
                geojson_path,
                min_area_threshold=conf.min_area_threshold,
                max_aspect_ratio=conf.max_aspect_ratio,
                min_solidity=conf.min_solidity,
            )
        else:
            binary_mask = threshold_prediction_map(segment_map, conf.threshold)
            contours = extract_contours(binary_mask)
            geojson_data = contours_to_geojson(contours, transform, crs, os.path.splitext(os.path.basename(image_path))[0])
            save_geojson(geojson_data, geojson_path)
    
        logger.info(f"Successfully processed and saved GeoJSON for: {os.path.basename(image_path)}")
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error processing image {os.path.basename(image_path)}: {e}"))


def process_single_image(
    image_path: str,
    conf: object,
    output_dir: str,
    id2label: dict,
    post_process: bool = False,
) -> None:
    logger = get_logger()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Processing image: {os.path.basename(image_path)}")
    
        model = load_model(conf.model_config, conf.best_model, id2label, device)
        refine_model = load_refine_model(conf.refine_model, device)

        logger.debug('Z0')
        geojson_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson")
        os.makedirs(os.path.dirname(geojson_path), exist_ok=True)
        logger.debug('Z1')
        process_image(model, refine_model, image_path, geojson_path, conf, post_process)
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error processing image {os.path.basename(image_path)}: {e}"))


def run_inference(
    data_path: str,
    config_file_path: str,
    output_dir: str,
    post_process: bool = False,
    verbosity: str = "info",
    num_processes: int = 4,
) -> None:
    logger = configure_logger(verbosity=verbosity)

    validate_path(logger, data_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    id2label = {0: "alive", 1: "dead"}
    conf = parse_config(config_file_path)

    data_path = Path(data_path)
    image_paths = (
        list(data_path.rglob("*.tiff"))
        + list(data_path.rglob("*.tif"))
        + list(data_path.rglob("*.jp2"))
        if data_path.is_dir()
        else [data_path]
    )

    if not image_paths:
        logger.warning(f"No images found in the specified path: {data_path}")
        return

    logger.info(f"Found {len(image_paths)} images to process.")

    tasks = [
        (image_path, conf, output_dir, id2label, post_process)
        for image_path in image_paths
        if not os.path.exists(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson"))
    ]

    try:
        num_processes = min(num_processes, cpu_count())
        with Pool(processes=num_processes, initializer=initialize_logger, initargs=(verbosity,)) as pool:
            pool.starmap(process_single_image, tasks)
        logger.info(f"Batch processing completed: {len(image_paths)} images processed.")
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error during parallel processing: {e}"))


def parse_config(config_file_path: str) -> argparse.Namespace:
    logger = get_logger()
    validate_path(logger, config_file_path)
    
    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("--model-config",       type=str, required=True, help="Path to the model configuration file (e.g., architecture, hyperparameters).")
    parser.add("--best-model",         type=str, required=True, help="Path to the file containing the best model weights.")
    parser.add("--refine-model",       type=str, required=True, help="Path to the file containing the refine model weights.")
    parser.add("--window-size",        type=int,   default=256, help="Size of the sliding window for inference (default: 256 pixels).")
    parser.add("--stride",             type=int,   default=128, help="Stride length for sliding window during inference (default: 128 pixels).")
    parser.add("--threshold",          type=float, default=0.5, help="Threshold for binary classification during inference (default: 0.5).")
    parser.add("--min-area-threshold", type=float, default=1.0, help="Minimum area (in pixels) for retaining a detected region.")
    parser.add("--max-aspect-ratio",   type=float, default=3.0, help="Maximum allowable aspect ratio for detected regions.")
    parser.add("--min-solidity",       type=float, default=0.85,help="Minimum solidity for retaining a detected region (solidity = area/convex hull).")
    parser.add("--min-distance",       type=int,   default=7,   help="Minimum distance between peaks for watershed segmentation.")
    parser.add("--dilation-radius",    type=int,   default=0,   help="Radius of the structuring element for dilating binary masks.")
    parser.add("--blur-sigma",         type=float, default=1.0, help="Standard deviation for Gaussian blur applied to prediction maps.")
    parser.add("--centroid-threshold", type=float, default=0.5, help="Threshold for filtering peaks based on the centroid map.")
    parser.add("--nir-rgb-order",      type=int, nargs='+', default=[3, 0, 1, 2], help="Order of NIR, Red, Green, and Blue channels in the input imagery.")

    conf, _ = parser.parse_known_args()
    return conf


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument('data_path', type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config',  type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument('--outdir',  type=str, help="Directory to save GeoJSON predictions (default: same as input)")
    parser.add_argument('--post-process', action="store_true", help="Enable or disable post-processing")
    parser.add_argument('--verbosity', type=str, choices=['info', 'debug', 'warning'], default='info')

    args = parser.parse_args()

    logger = configure_logger(verbosity=args.verbosity)
    
    run_inference(args.data_path, args.config, args.outdir, args.post_process, verbosity=args.verbosity)


if __name__ == "__main__":
    main()


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
    --outdir /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm/2011/Predictions_r \
    --post-process --verbosity debug

- For entire folder

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/copenhagen_data \
    --config ./configs/USA_RGBNIR_60cm_inference.txt

2) save geojsons to output folder

python -m inference.engine \
    /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm \
    --config ./configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir /Users/anisr/Documents/dead_trees/Finland/Predictions_r \
    --post-process

- Run viewer api service

uvicorn treemort_api:app --reload

- Run viewer application

streamlit run treemort_app.py

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
export TREEMORT_DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees"

sbatch --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/Finland_RGBNIR_25cm_inference.txt",DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm",OUTPUT_PATH="$TREEMORT_DATA_PATH/Finland/Predictions" $TREEMORT_REPO_PATH/scripts/run_inference.sh

'''