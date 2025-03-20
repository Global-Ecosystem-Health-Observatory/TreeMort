import os
import torch
import argparse

from pathlib import Path
from multiprocessing import Pool, cpu_count

from treemort.utils.logger import configure_logger, get_logger, initialize_logger
from treemort.utils.config import setup

from inference.utils import (
    load_model,
    sliding_window_inference,
    load_and_preprocess_image,
    threshold_prediction_map,
    extract_contours,
    save_geojson,
    log_and_raise,
    validate_path,
    compute_watershed, 
    extract_ellipses, 
    save_geojson,
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
            model,
            image,
            window_size=conf.window_size,
            stride=conf.stride,
            threshold=conf.segment_threshold,
            output_channels=conf.output_channels,
            activation=conf.activation,
        )
        segment_map = prediction_maps[0]
        
        image_np = image.cpu().numpy()
        segment_map_np = segment_map.cpu().numpy()
        
        if post_process:
            labels_ws = compute_watershed(segment_map_np, conf)
            features = extract_ellipses(labels_ws, transform, conf)
            save_geojson(features, geojson_path, crs, transform, name="FittedEllipses")

        else:
            binary_mask = threshold_prediction_map(segment_map_np, conf.segment_threshold)
            features = extract_contours(binary_mask, transform)
            save_geojson(features, geojson_path, crs, transform, name="Contours")

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

        model = load_model(conf, id2label, device)

        geojson_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson")
        os.makedirs(os.path.dirname(geojson_path), exist_ok=True)

        process_image(model, image_path, geojson_path, conf, post_process)
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error processing image {os.path.basename(image_path)}: {e}"))


def run_inference(
    data_path: str,
    config_file_path: str,
    model_config: str,
    data_config: str,
    output_dir: str,
    post_process: bool = False,
    verbosity: str = "info",
    num_processes: int = 4,
) -> None:
    logger = get_logger()

    validate_path(data_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    id2label = {0: "alive", 1: "dead"}
    
    conf = setup(config_file_path, model_config=model_config, data_config=data_config)

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


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument('data_path',      type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config',       type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument('--model-config', type=str, required=True, help="Path to the model configuration file (e.g., architecture, hyperparameters).")
    parser.add_argument('--data-config',  type=str, required=True, help="Path to the data configuration file")
    parser.add_argument('--outdir',       type=str, help="Directory to save GeoJSON predictions (default: same as input)")
    parser.add_argument('--post-process', action="store_true", help="Enable or disable post-processing")
    parser.add_argument('--verbosity',    type=str, choices=['info', 'debug', 'warning'], default='info')
    
    args = parser.parse_args()
    
    _ = configure_logger(verbosity=args.verbosity)
    
    run_inference(args.data_path, args.config, args.model_config, args.data_config, args.outdir, args.post_process, verbosity=args.verbosity)


if __name__ == "__main__":
    main()


''' Usage: Local

- Env variables

export TREEMORT_DATA_PATH="/Users/anisr/Documents/dead_trees"
export TREEMORT_REPO_PATH="/Users/anisr/Documents/TreeSeg"

- For single file:

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ${TREEMORT_REPO_PATH}/configs/inference/finland.txt \
    --model-config ${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt \
    --data-config ${TREEMORT_REPO_PATH}/configs/data/finland.txt

2) save geojsons to an output folder (Recommended)

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ${TREEMORT_REPO_PATH}/configs/inference/finland.txt \
    --model-config ${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt \
    --data-config ${TREEMORT_REPO_PATH}/configs/data/finland.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm/2011/Images/M3442B_2011_1.tiff \
    --config ${TREEMORT_REPO_PATH}/configs/inference/finland.txt \
    --model-config ${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt \
    --data-config ${TREEMORT_REPO_PATH}/configs/data/finland.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions \
    --post-process --verbosity debug

- For entire folder

1) save geojsons in a 'Predictions' folder alongside Images and Geojsons

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm \
    --config ${TREEMORT_REPO_PATH}/configs/inference/finland.txt \
    --model-config ${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt \
    --data-config ${TREEMORT_REPO_PATH}/configs/data/finland.txt \

2) save geojsons to output folder

python -m inference.engine \
    ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm \
    --config ${TREEMORT_REPO_PATH}/configs/inference/finland.txt \
    --model-config ${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt \
    --data-config ${TREEMORT_REPO_PATH}/configs/data/finland.txt \
    --outdir ${TREEMORT_DATA_PATH}/Finland/Predictions

2) HPC

Usage: ./submit_inference.sh <hpc_type> <model config file> <data config file> [--post-process]

Examples:

(sans post-processing) sh ./scripts/submit_inference.sh lumi flair_unet finland
(with post-processing) sh ./scripts/submit_inference.sh lumi flair_unet finland --post-process

'''

''' Data: Puhti

- Download geojsons

scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/dead_trees/Finland/Predictions ~/Documents/dead_trees/Finland

- Download model

scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/output/flair_unet output

'''

''' Usage: Inference API

- Run viewer api service

uvicorn treemort_api:app --reload

- Run viewer application

streamlit run treemort_app.py

'''
