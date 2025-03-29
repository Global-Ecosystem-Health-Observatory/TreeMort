import os
import configargparse

from treemort.utils.logger import get_logger, log_and_raise


def expand_path(path):
    return os.path.expandvars(path)


def validate_path(path: str, is_dir: bool = False) -> bool:
    logger = get_logger()

    if not os.path.exists(path):
        log_and_raise(logger, FileNotFoundError(f"Path does not exist: {path}"))
    if is_dir and not os.path.isdir(path):
        log_and_raise(logger, NotADirectoryError(f"Expected directory but got: {path}"))
    return True


def load_include_files(config_file_path: str, data_config: str = None, model_config: str = None) -> list[str]:
    logger = get_logger()

    include_files = []
    processed_files = set()

    def process_file(file_path):
        if file_path in processed_files:
            return
        processed_files.add(file_path)

        with open(file_path, "r") as f:
            for line in f:
                if line.strip().startswith("include"):
                    _, include_val = line.split("=", 1)
                    include_val = include_val.strip()

                    if data_config is not None:
                        include_val = include_val.replace("{data_config}", data_config)
                    
                    if model_config is not None:
                        include_val = include_val.replace("{model_config}", model_config)

                    if os.path.sep not in include_val:
                        include_path = os.path.join(os.path.dirname(file_path), include_val)
                    else:
                        include_path = include_val

                    if os.path.exists(include_path):
                        if include_path not in include_files:
                            process_file(include_path)
                            logger.info(f"Including config file: {include_path}")
                            include_files.append(include_path)
                    else:
                        logger.warning(f"Include file not found: {include_path}")

    process_file(config_file_path)

    if data_config is None:
        include_path = os.path.join(os.path.dirname(os.path.dirname(config_file_path)), "data", "base_config.txt")
        if include_path not in include_files:
            include_files.append(include_path)
    
    if model_config is None:
        include_path = os.path.join(os.path.dirname(os.path.dirname(config_file_path)), "model", "base_config.txt")
        if include_path not in include_files:
            include_files.append(include_path)

    return include_files


def build_parser(config_files):
    parser = configargparse.ArgParser(default_config_files=config_files)

    model_group = parser.add_argument_group('Model')
    model_group.add("--model", type=str, required=True, help="neural network model name for training")
    model_group.add("--backbone", type=str, default=None, help="model backbone")
    model_group.add("--model-weights", type=str, default="latest", help="weights file for training continuation")
    model_group.add("--best-model", type=str, default='best.weights.pth', help="Path to the file containing the best model weights.")
    model_group.add("--learning-rate", type=float, default=2e-4, help="learning rate for optimizer")
    model_group.add("--activation", type=str, default="sigmoid", help="activation function")
    model_group.add("--loss", type=str, default="hybrid", help="loss function for the network")
    model_group.add("--segment-threshold", type=float, default=0.5, help="Threshold for binary classification during inference (default: 0.5).")
    model_group.add("--centroid-threshold", type=float, default=0.5, help="Threshold for filtering peaks based on the centroid map.")
    model_group.add("--hybrid-threshold", type=float, default=-0.5, help="Threshold for filtering contours based on the hybrid map.")
    model_group.add("--class-weights", type=float, nargs="+", default=[0.5, 0.5], help="class weights for imbalanced classes")

    train_group = parser.add_argument_group('Training')
    train_group.add("--epochs", type=int, required=True, help="number of epochs for training")
    train_group.add("--train-batch-size", type=int, required=True, help="batch size for training")
    train_group.add("--val-batch-size", type=int, required=True, help="batch size for validation")
    train_group.add("--test-batch-size", type=int, required=True, help="batch size for testing")
    train_group.add("--train-crop-size", type=int, required=True, help="crop size for training")
    train_group.add("--val-crop-size", type=int, required=True, help="crop size for validation")
    train_group.add("--test-crop-size", type=int, required=True, help="crop size for testing")
    train_group.add("--val-size", type=float, default=0.2, help="split for validation set")
    train_group.add("--test-size", type=float, default=0.1, help="split for test set")
    train_group.add("--resume", action="store_true", help="resume training using stored model weights")

    data_group = parser.add_argument_group('Data')
    data_group.add("--data-folder",     type=str, required=True, help="directory with aerial image and label data")
    data_group.add("--hdf5-file",       type=str, required=True, help="hdf5 file containing data")
    data_group.add("--window-size",     type=int, default=256,   help="Size of the sliding window for inference (default: 256 pixels).")
    data_group.add("--stride",          type=int, default=128,   help="Stride length for sliding window during inference (default: 128 pixels).")
    data_group.add("--input-channels",  type=int, required=True, help="number of input channels")
    data_group.add("--output-channels", type=int, required=True, help="number of output channels")
    data_group.add("--nir-rgb-order",   type=int, nargs='+', default=[3, 0, 1, 2], help="Order of NIR, Red, Green, and Blue channels in the input imagery.")
    data_group.add("--normalize-imagewise",   action="store_true", help="normalize imagewise")
    data_group.add("--normalize-channelwise", action="store_true", help="normalize channelwise")

    inference_group = parser.add_argument_group('Inference')
    inference_group.add("--min-area", type=float, default=1.0, help="Minimum area (in pixels) for retaining a detected region.")
    inference_group.add("--max-aspect-ratio", type=float, default=3.0, help="Maximum allowable aspect ratio for detected regions.")
    inference_group.add("--min-solidity", type=float, default=0.85, help="Minimum solidity for retaining a detected region (solidity = area/convex hull).")
    inference_group.add("--min-distance", type=int, default=5, help="Minimum distance between peaks for watershed segmentation.")
    inference_group.add("--dilation-radius", type=int, default=0, help="Radius of the structuring element for dilating binary masks.")
    inference_group.add("--erosion-radius", type=int, default=0, help="Radius of the structuring element for eroding binary masks.")
    inference_group.add("--blur-sigma", type=float, default=1.0, help="Standard deviation for Gaussian blur applied to prediction maps.")
    inference_group.add("--tightness", type=float, default=0.1, help="Tightness parameter for ellipse fitting.")
    
    output_group = parser.add_argument_group('Output')
    output_group.add("--output-dir", type=str, default="./output", help="directory to save output files")

    return parser


def setup(config_file_path, model_config=None, data_config=None):
    logger = get_logger()

    validate_path(config_file_path)

    logger.info(f"Using config file: {config_file_path}")

    include_files = load_include_files(config_file_path, model_config=model_config, data_config=data_config)
    config_files = include_files + [config_file_path]

    parser = build_parser(config_files)
    conf, _ = parser.parse_known_args()

    conf.data_folder = expand_path(conf.data_folder)
    conf.output_dir = expand_path(conf.output_dir)

    conf.min_area_pixels = conf.min_area / 0.0625

    logger.info("Configuration successfully loaded.")
    return conf