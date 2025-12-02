import os
import time
import torch
import argparse
import configargparse

from pathlib import Path
from multiprocessing import Pool, cpu_count

from skimage.morphology import label

from treemort.utils.logger import configure_logger, get_logger, initialize_logger
from inference.utils import (
    load_model,
    sliding_window_inference,
    load_and_preprocess_image,
    threshold_prediction_map,
    extract_contours,
    extract_contours_from_labels,
    save_geojson,
    log_and_raise,
    validate_path,
    expand_path,
    compute_watershed,
    extract_ellipses,
    segment_filtering_only,
    watershed_segmentation_only,
)
from treemort.utils.config import setup


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
        total_start_time = time.time()
        start_time = time.time()
        image, transform, crs = load_and_preprocess_image(image_path, conf.nir_rgb_order)
        logger.info(f"Image loaded in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()

        prediction_maps = sliding_window_inference(
            model,
            image,
            window_size=conf.window_size,
            stride=conf.stride,
            threshold=conf.segment_threshold,
            output_channels=conf.output_channels,
        )
        logger.info(f"Sliding window inference completed in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()
        segment_map, centroid_map, hybrid_map = prediction_maps

        image_np = image.cpu().numpy()
        segment_map_np = segment_map.cpu().numpy()
        centroid_map_np = centroid_map.cpu().numpy()
        hybrid_map_np = hybrid_map.cpu().numpy()
        logger.info(f"Converted prediction maps to numpy in {time.time() - start_time:.2f} seconds.")

        if post_process:
            start_time = time.time()
            labels_ws = compute_watershed(segment_map_np, centroid_map_np, hybrid_map_np, conf)
            logger.info(f"Watershed segmentation took {time.time() - start_time:.2f} seconds.")
            start_time = time.time()
            features = list(extract_ellipses(labels_ws, transform, conf))
            logger.info(f"Ellipse extraction took {time.time() - start_time:.2f} seconds.")
            start_time = time.time()
            save_geojson(features, geojson_path, crs, transform, name="FittedEllipses")
            logger.info(f"GeoJSON saved in {time.time() - start_time:.2f} seconds.")

            # # Filtering-only variant
            # start_time = time.time()
            # filtered_mask = segment_filtering_only(segment_map_np, conf)
            # logger.info(f"Segment filtering took {time.time() - start_time:.2f} seconds.")
            # start_time = time.time()
            # features = extract_contours(filtered_mask, transform)
            # logger.info(f"Contour extraction took {time.time() - start_time:.2f} seconds.")
            # start_time = time.time()
            # save_geojson(features, geojson_path, crs, transform, name="FilteredContours")
            # logger.info(f"GeoJSON saved in {time.time() - start_time:.2f} seconds.")

            # # Watershed-only variant
            # start_time = time.time()
            # labels_ws = watershed_segmentation_only(segment_map_np, centroid_map_np, hybrid_map_np, conf)
            # logger.info(f"Watershed segmentation took {time.time() - start_time:.2f} seconds.")
            # start_time = time.time()
            # features = extract_contours_from_labels(labels_ws, transform)
            # logger.info(f"Contour extraction took {time.time() - start_time:.2f} seconds.")
            # start_time = time.time()
            # save_geojson(features, geojson_path, crs, transform, name="WatershedContours")
            # logger.info(f"GeoJSON saved in {time.time() - start_time:.2f} seconds.")

        else:
            start_time = time.time()
            binary_mask = threshold_prediction_map(segment_map_np, conf.segment_threshold)
            logger.info(f"Thresholded prediction map in {time.time() - start_time:.2f} seconds.")
            start_time = time.time()
            features = extract_contours(binary_mask, transform)
            logger.info(f"Contour extraction took {time.time() - start_time:.2f} seconds.")
            start_time = time.time()
            save_geojson(features, geojson_path, crs, transform, name="Contours")
            logger.info(f"GeoJSON saved in {time.time() - start_time:.2f} seconds.")

        logger.info(
            f"Total processing time for {os.path.basename(image_path)}: {time.time() - total_start_time:.2f} seconds."
        )
        logger.info(f"Successfully processed and saved GeoJSON for: {os.path.basename(image_path)}")
    except Exception as e:
        log_and_raise(
            logger,
            RuntimeError(f"Error processing image {os.path.basename(image_path)}: {e}"),
        )


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
        log_and_raise(
            logger,
            RuntimeError(f"Error processing image {os.path.basename(image_path)}: {e}"),
        )


def run_inference(
    data_path: str,
    config_file_path: str,
    model_config: str,
    data_config: str,
    output_dir: str,
    post_process: bool = False,
    verbosity: str = "info",
    num_processes: int = 4,
    list_file: str = None,
) -> None:
    logger = get_logger()

    validate_path(logger, data_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    id2label = {0: "alive", 1: "dead"}

    conf = setup(config_file_path, model_config=model_config, data_config=data_config)

    # Select images either from a provided list file or by directory scan
    if list_file:
        data_path = Path(data_path)
        if not os.path.isfile(list_file):
            logger.error(f"List file not found: {list_file}")
            return
        with open(list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        image_paths = []
        for p in lines:
            p_path = Path(p)
            if not p_path.is_absolute():
                p_path = data_path / p_path
            image_paths.append(p_path)
    else:
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
        if not os.path.exists(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.geojson"))
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
    parser.add(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model configuration file (e.g., architecture, hyperparameters).",
    )
    parser.add(
        "--best-model",
        type=str,
        required=True,
        help="Path to the file containing the best model weights.",
    )
    parser.add(
        "--window-size",
        type=int,
        default=256,
        help="Size of the sliding window for inference (default: 256 pixels).",
    )
    parser.add(
        "--stride",
        type=int,
        default=128,
        help="Stride length for sliding window during inference (default: 128 pixels).",
    )
    parser.add("--input-channels", type=int, required=True, help="number of input channels")
    parser.add("--output-channels", type=int, required=True, help="number of output channels")
    parser.add(
        "--min-area",
        type=float,
        default=1.0,
        help="Minimum area (in pixels) for retaining a detected region.",
    )
    parser.add(
        "--max-aspect-ratio",
        type=float,
        default=3.0,
        help="Maximum allowable aspect ratio for detected regions.",
    )
    parser.add(
        "--min-solidity",
        type=float,
        default=0.85,
        help="Minimum solidity for retaining a detected region (solidity = area/convex hull).",
    )
    parser.add(
        "--min-distance",
        type=int,
        default=7,
        help="Minimum distance between peaks for watershed segmentation.",
    )
    parser.add(
        "--dilation-radius",
        type=int,
        default=0,
        help="Radius of the structuring element for dilating binary masks.",
    )
    parser.add(
        "--erosion-radius",
        type=int,
        default=0,
        help="Radius of the structuring element for eroding binary masks.",
    )
    parser.add(
        "--blur-sigma",
        type=float,
        default=1.0,
        help="Standard deviation for Gaussian blur applied to prediction maps.",
    )
    parser.add(
        "--segment-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification during inference (default: 0.5).",
    )
    parser.add(
        "--centroid-threshold",
        type=float,
        default=0.5,
        help="Threshold for filtering peaks based on the centroid map.",
    )
    parser.add(
        "--hybrid-threshold",
        type=float,
        default=-0.5,
        help="Threshold for filtering contours based on the hybrid map.",
    )
    parser.add(
        "--tightness",
        type=float,
        default=0.1,
        help="Tightness parameter for ellipse fitting.",
    )
    parser.add(
        "--nir-rgb-order",
        type=int,
        nargs="+",
        default=[3, 0, 1, 2],
        help="Order of NIR, Red, Green, and Blue channels in the input imagery.",
    )

    conf, _ = parser.parse_known_args()
    conf.model_config = expand_path(conf.model_config)

    conf.min_area_pixels = (
        conf.min_area / 0.0625
    )  # for 25cm pix resolution; (0.25*0.25) = 0.0625 sq. m per pixel ; 1/0.0625 = 16 pixels
    return conf


def main():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument('data_path', type=str, help="Path to the input image file or directory containing images")
    parser.add_argument('--config', type=str, required=True, help="Path to the inference configuration file")
    parser.add_argument(
        '--model-config',
        type=str,
        required=True,
        help="Path to the model configuration file (e.g., architecture, hyperparameters).",
    )
    parser.add_argument('--data-config', type=str, required=True, help="Path to the data configuration file")
    parser.add_argument('--outdir', type=str, help="Directory to save GeoJSON predictions (default: same as input)")
    parser.add_argument('--post-process', action="store_true", help="Enable or disable post-processing")
    parser.add_argument('--verbosity', type=str, choices=['info', 'debug', 'warning'], default='info')
    parser.add_argument('--list-file', type=str, help="Path to text file with list of image filenames to process")

    args = parser.parse_args()

    logger = configure_logger(verbosity=args.verbosity)
    run_inference(
        args.data_path,
        args.config,
        args.model_config,
        args.data_config,
        args.outdir,
        args.post_process,
        verbosity=args.verbosity,
        list_file=args.list_file,
    )


if __name__ == "__main__":
    main()


""" Usage:

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
    ./output/M4231B_2023_RGBNIR.tif \
    --config ${TREEMORT_REPO_PATH}/configs/Finland_RGBNIR_25cm_inference_sdt.txt \
    --outdir ./output/M4231B_2023_RGBNIR.geojson --post-process
    
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

export TREEMORT_VENV_PATH="/projappl/project_462000684/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
export TREEMORT_DATA_PATH="/scratch/project_462000684/rahmanan/dead_trees"

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

"""

"""

with open('/Users/anisr/Downloads/a3s.fi.txt', 'r') as file:
    filenames = file.read().splitlines()

# Filter filenames containing '2023'
filtered_filenames = [f for f in filenames if '2023' in f]
with open('/Users/anisr/Downloads/a3s.fi.2023.txt', 'w') as f:
    f.write('\n'.join(filtered_filenames))

scp ~/Downloads/a3s.fi.2023.txt rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan

swift download \
    --skip-identical \
    DRYTREE_Orthoimagery_Finland \
    $(grep -vE '^(#|$)' a3s.fi.2023.txt) \
    -D /scratch/project_462000684/rahmanan/DRYTREE_Orthoimagery_Finland

scp -O -r rahmanan@puhti.csc.fi:/scratch/project_2008436/rahmanan/output/flair_unet_sdt output
scp -O -r output/flair_unet_sdt rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/output

export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

sh $TREEMORT_REPO_PATH/scripts/submit_inference.sh lumi flair_unet_sdt all --post-process --list-file $TREEMORT_REPO_PATH/colab/selected_images_2023.txt --chunks 5

scp rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/DRYTREE_Orthoimagery_Finland/K3423G_2023_RGBNIR.geojson.tif ~/Downloads
scp rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/Predictions_DRYTREE_Orthoimagery_Finland/K3423G_2023_RGBNIR.geojson ~/Downloads

scp -O -r aurahman@lumi.csc.fi:/scratch/project_462001070/aurahman/dead_trees/Switzerland/Predictions_t1_post_process ~/Downloads


"""
