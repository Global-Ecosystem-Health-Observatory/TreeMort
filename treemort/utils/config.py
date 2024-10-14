import os
import configargparse

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def setup(config_file_path=None):
    if config_file_path:
        if not os.path.exists(config_file_path):
            logger.error(f"Config file not found at: {config_file_path}")
            raise FileNotFoundError(f"Config file not found at: {config_file_path}")
        logger.info(f"Using config file: {config_file_path}")
    
    parser = configargparse.ArgParser(default_config_files=[config_file_path] if config_file_path else [])
    
    data_group = parser.add_argument_group('Data')
    data_group.add( "-df-fi", "--data-folder-finnish", type=str, required=True, help="Directory with Finnish aerial image and label data")
    data_group.add( "-df-us", "--data-folder-us",      type=str, required=True, help="Directory with US aerial image and label data")
    data_group.add( "-hf-fi", "--hdf5-file-finnish",   type=str, required=True, help="HDF5 file containing Finnish dataset")
    data_group.add( "-hf-us", "--hdf5-file-us",        type=str, required=True, help="HDF5 file containing US dataset")
    
    model_group = parser.add_argument_group('Model')
    model_group.add("-m", "--model",            type=str,   required=True,     help="Neural network model name for training")
    model_group.add( "-b", "--backbone",        type=str,   default=None,      help="model backbone")
    model_group.add("-mw", "--model-weights",   type=str,   default="latest",  help="Weights file for training continuation (best, latest)")
    model_group.add("-lr", "--learning-rate",   type=float, default=1e-4,      help="Learning rate for optimizer")
    model_group.add("-av", "--activation",      type=str,   default="sigmoid", help="Activation function")
    model_group.add("-ls", "--loss",            type=str,   default="hybrid",  help="Loss function for the segmentation task (mse, hybrid, weighted_dice_loss)")
    model_group.add("-th", "--threshold",       type=float, default=0.5,       help="threshold for classifier")
    model_group.add("-ts", "--train-crop-size", type=int,   required=True,     help="Crop size for training")
    model_group.add("-vs", "--val-crop-size",   type=int,   required=True,     help="Crop size for validation")
    model_group.add("-os", "--test-crop-size",  type=int,   required=True,     help="Crop size for testing")
    model_group.add("-ic", "--input-channels",  type=int,   required=True,     help="Number of input channels")
    model_group.add("-oc", "--output-channels", type=int,   required=True,     help="Number of output channels")
    model_group.add("-cw", "--class-weights",   type=float, nargs="+", default=[0.5, 0.5], help="Class weights for imbalanced classes")
    
    train_group = parser.add_argument_group('Training')
    train_group.add("-e", "--epochs",            type=int,   required=True, help="Number of epochs for training")
    train_group.add("-ib", "--train-batch-size", type=int,   required=True, help="Batch size for training")
    train_group.add("-vb", "--val-batch-size",   type=int,   required=True, help="Batch size for validation")
    train_group.add("-ob", "--test-batch-size",  type=int,   required=True, help="Batch size for testing")
    train_group.add("-vz", "--val-size",         type=float, default=0.2,   help="Validation set split ratio")
    train_group.add("-tz", "--test-size",        type=float, default=0.1,   help="Test set split ratio")
    train_group.add("-rs", "--resume",           action="store_true",       help="Resume training using stored model weights")
    
    domain_group = parser.add_argument_group('Domain-Adversarial Training')
    domain_group.add("--lambda-adv",  type=float, default=0.5,             help="Weight for the adversarial domain classification loss")
    domain_group.add("--domain-loss", type=str,   default="cross_entropy", help="Loss function for the domain classification task (e.g., cross_entropy)")

    output_group = parser.add_argument_group('Output')
    output_group.add("-o", "--output-dir", type=str, default="output", help="Directory to save output files")
    
    conf, _ = parser.parse_known_args()
    
    logger.info("Configuration successfully loaded.")
    
    return conf