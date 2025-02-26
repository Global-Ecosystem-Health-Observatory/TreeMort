import os
import cv2
import torch
import argparse
import configargparse

import numpy as np
import geopandas as gpd

from pathlib import Path
from multiprocessing import Pool, cpu_count
from skimage.morphology import label
from shapely.geometry import Point

from treemort.utils.logger import configure_logger, get_logger
from inference.utils import (
    load_model,
    sliding_window_inference,
    initialize_logger,
    save_labels_as_geojson,
    load_and_preprocess_image,
    threshold_prediction_map,
    extract_contours,
    contours_to_geojson,
    save_geojson,
    log_and_raise,
    validate_path,
    expand_path,
    compute_final_segmentation,
)


def process_image(
    model: torch.nn.Module,
    image_path: str,
    geojson_path: str,
    conf: object,
    post_process: bool,
) -> None:
    logger = get_logger()
    logger.debug(f"Processing image: {os.path.basename(image_path)}")

    try:
        image, transform, crs = load_and_preprocess_image(image_path, conf.nir_rgb_order)
        logger.debug(f"Loaded and preprocessed image: {os.path.basename(image_path)}")

        prediction_maps = sliding_window_inference(
            model, image, window_size=conf.window_size, stride=conf.stride, threshold=conf.threshold, output_channels=conf.output_channels
        )
        segment_map, centroid_map, hybrid_map = prediction_maps

        if post_process:
            final_segmentation = compute_final_segmentation(
                seg_map=segment_map, 
                centroid_map=centroid_map, 
                hybrid_map=hybrid_map,
                seg_threshold=conf.threshold,
                centroid_params={'min_distance': conf.min_distance, 'threshold_abs': conf.centroid_threshold},
                hybrid_threshold=conf.hybrid_threshold
            )

            save_labels_as_geojson(
                final_segmentation,
                transform,
                crs,
                geojson_path,
                min_area_threshold=conf.min_area_threshold,
                max_aspect_ratio=conf.max_aspect_ratio,
                min_solidity=conf.min_solidity,
            )

            # # Convert tensors to NumPy arrays for further processing.
            # image_np = image.cpu().numpy()
            # segment_map_np = segment_map.cpu().numpy()
            # centroid_map_np = centroid_map.cpu().numpy()
            # hybrid_map_np = hybrid_map.cpu().numpy()  # currently not used in post-processing

            # # Process the segmentation output.
            # min_size_threshold = 30  # e.g., corresponding to 1.5m crown diameter
            # filtered_segment_map = filter_segment_map(segment_map_np, threshold=conf.threshold, min_size=min_size_threshold)

            # # Process the centroid branch.
            # centroid_points = extract_centroid_points(centroid_map, threshold=conf.centroid_threshold)
            # refined_centroid_map = refine_centroid_map(filtered_segment_map, centroid_map_np, threshold=0.2)
            # multiple_peaks = detect_multiple_peaks(refined_centroid_map, min_distance=5, threshold_abs=0.1)

            # # Use watershed segmentation to generate refined segments.
            # watershed_segmented_map = segment_using_watershed(filtered_segment_map, refined_centroid_map, multiple_peaks, threshold=conf.threshold)

            # # (Optional)
            # # hybrid_np = hybrid_map.cpu().numpy().squeeze()
            # # binary_hybrid = (hybrid_np < -0.5).astype(np.uint8) * 255
            # # hybrid_contours, _ = cv2.findContours(binary_hybrid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # hybrid_geojson = hybrid_contours_to_geojson(
            # #     hybrid_contours, transform, crs,
            # #     os.path.splitext(os.path.basename(image_path))[0] + "_hybrid"
            # # )
            # # save_geojson(hybrid_geojson, geojson_path.replace(".geojson", "_hybrid.geojson"))

            # # Save final watershed segmentation as GeoJSON.
            # save_labels_as_geojson(
            #     watershed_segmented_map,
            #     transform,
            #     crs,
            #     geojson_path,
            #     min_area_threshold=conf.min_area_threshold,
            #     max_aspect_ratio=conf.max_aspect_ratio,
            #     min_solidity=conf.min_solidity,
            # )
        else:
            # Without post-processing, threshold the segmentation map and extract contours.
            binary_mask = threshold_prediction_map(segment_map, conf.threshold)
            contours = extract_contours(binary_mask)
            geojson_data = contours_to_geojson(
                contours, transform, crs, os.path.splitext(os.path.basename(image_path))[0]
            )
            save_geojson(geojson_data, geojson_path)

            # (Optional)
            # For centroid:
            # centroid_points = extract_centroid_points(centroid_map, threshold=conf.centroid_threshold)
            # from rasterio.transform import xy
            # from shapely.geometry import Point
            # geoms = []
            # for x, y in centroid_points:
            #     lon, lat = xy(transform, int(y), int(x))
            #     geoms.append(Point(lon, lat))
            # gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
            # gdf.to_file(geojson_path.replace(".geojson", "_centroid.geojson"), driver="GeoJSON")
            #
            # For hybrid (if required), similar processing could be applied after proper thresholding.

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

        geojson_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson")
        os.makedirs(os.path.dirname(geojson_path), exist_ok=True)

        process_image(model, image_path, geojson_path, conf, post_process)
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
        list(data_path.rglob("*.tiff")) + list(data_path.rglob("*.tif")) + list(data_path.rglob("*.jp2"))
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
        # Uncomment the following line to skip images already processed:
        # if not os.path.exists(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson"))
    ]

    try:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        num_processes = int(slurm_cpus) if slurm_cpus else min(num_processes, cpu_count())

        with Pool(processes=num_processes, initializer=initialize_logger, initargs=(verbosity,)) as pool:
            pool.starmap(process_single_image, tasks)
        logger.info(f"Batch processing completed: {len(image_paths)} images processed.")
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error during parallel processing: {e}"))


def parse_config(config_file_path: str) -> argparse.Namespace:
    logger = get_logger()
    validate_path(logger, config_file_path)

    parser = configargparse.ArgParser(default_config_files=[config_file_path])
    parser.add("--model-config", type=str, required=True, help="Path to the model configuration file (e.g., architecture, hyperparameters).")
    parser.add("--best-model", type=str, required=True, help="Path to the file containing the best model weights.")
    parser.add("--window-size", type=int, default=256, help="Size of the sliding window for inference (default: 256 pixels).")
    parser.add("--stride", type=int, default=128, help="Stride length for sliding window during inference (default: 128 pixels).")
    parser.add("--input-channels",  type=int, required=True, help="number of input channels")
    parser.add("--output-channels", type=int, required=True, help="number of output channels")
    parser.add("--threshold", type=float, default=0.5, help="Threshold for binary classification during inference (default: 0.5).")
    parser.add("--min-area-threshold", type=float, default=1.0, help="Minimum area (in pixels) for retaining a detected region.")
    parser.add("--max-aspect-ratio", type=float, default=3.0, help="Maximum allowable aspect ratio for detected regions.")
    parser.add("--min-solidity", type=float, default=0.85, help="Minimum solidity for retaining a detected region (solidity = area/convex hull).")
    parser.add("--min-distance", type=int, default=7, help="Minimum distance between peaks for watershed segmentation.")
    parser.add("--dilation-radius", type=int, default=0, help="Radius of the structuring element for dilating binary masks.")
    parser.add("--blur-sigma", type=float, default=1.0, help="Standard deviation for Gaussian blur applied to prediction maps.")
    parser.add("--centroid-threshold", type=float, default=0.5, help="Threshold for filtering peaks based on the centroid map.")
    parser.add("--hybrid-threshold", type=float, default=-0.5, help="Threshold for filtering contours based on the hybrid map.")
    parser.add("--nir-rgb-order", type=int, nargs='+', default=[3, 0, 1, 2], help="Order of NIR, Red, Green, and Blue channels in the input imagery.")

    conf, _ = parser.parse_known_args()
    conf.model_config = expand_path(conf.model_config)
    return conf


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument('data_path', type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config', type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument('--outdir', type=str, help="Directory to save GeoJSON predictions (default: same as input)")
    parser.add_argument('--post-process', action="store_true", help="Enable or disable post-processing")
    parser.add_argument('--verbosity', type=str, choices=['info', 'debug', 'warning'], default='info')

    args = parser.parse_args()
    logger = configure_logger(verbosity=args.verbosity)
    run_inference(args.data_path, args.config, args.outdir, args.post_process, verbosity=args.verbosity)


if __name__ == "__main__":
    main()


''' Usage:

export TREEMORT_DATA_PATH="/Users/anisr/Documents/dead_trees"
export TREEMORT_REPO_PATH="/Users/anisr/Documents/TreeSeg"

scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/output/flair_unet_sdt output

- For single file:

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/copenhagen_data/Images/patches_3095_377.tif \
    --config ./configs/USA_RGBNIR_60cm_inference.txt

2) save geojsons to an output folder

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm/2022/Images/L2344D_2022_1_ITD.tif \
    --config ${TREEMORT_REPO_PATH}/configs/Finland_RGBNIR_25cm_inference_sdt.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions_sdt_r --post-process

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ${TREEMORT_REPO_PATH}/configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions_sdt \
    --post-process --verbosity debug

- For entire folder

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    /Users/anisr/Documents/copenhagen_data \
    --config ./configs/USA_RGBNIR_60cm_inference.txt

2) save geojsons to output folder

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm \
    --config ${TREEMORT_REPO_PATH}/configs/Finland_RGBNIR_25cm_inference.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions_r \
    --post-process

- Run viewer api service

uvicorn treemort_api:app --reload

- Run viewer application

streamlit run treemort_app.py

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
export TREEMORT_DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees"

sbatch \
    --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/Finland_RGBNIR_25cm_inference.txt",\
    DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm",\
    OUTPUT_PATH="$TREEMORT_DATA_PATH/Finland/Predictions" \
    $TREEMORT_REPO_PATH/scripts/run_inference.sh

sbatch \
    --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/Finland_RGBNIR_25cm_inference.txt",\
    DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm",\
    OUTPUT_PATH="$TREEMORT_DATA_PATH/Finland/Predictions_r" \
    $TREEMORT_REPO_PATH/scripts/run_inference.sh --post-process

scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/dead_trees/Finland/Predictions ~/Documents/dead_trees/Finland
scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/dead_trees/Finland/Predictions_r ~/Documents/dead_trees/Finland

'''
