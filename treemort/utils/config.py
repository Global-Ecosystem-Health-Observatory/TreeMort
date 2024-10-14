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
    
    model_group = parser.add_argument_group('Model')
    model_group.add( "-m", "--model",         type=str,   required=True,        help="neural network model name for training")
    model_group.add( "-b", "--backbone",      type=str,   default=None,         help="model backbone")
    model_group.add("-mw", "--model-weights", type=str,   default="latest",     help="weights file for training continuation")
    model_group.add("-lr", "--learning-rate", type=float, default=2e-4,         help="learning rate for optimizer")
    model_group.add("-av", "--activation",    type=str,   default="sigmoid",    help="activation function")
    model_group.add("-ls", "--loss",          type=str,   default="hybrid",     help="loss function for the network")
    model_group.add("-th", "--threshold",     type=float, default=0.5,          help="threshold for classifier")
    model_group.add("-cw", "--class-weights", type=float, nargs="+", default=[0.5, 0.5], help="class weights for imbalanced classes")

    model_group.add("--teacher-model-name",       type=str,   default=None, help="weight file of pre-trained teacher model")
    model_group.add("--distillation-alpha",       type=float, default=0.5,  help="alpha value for blending distillation and standard loss")
    model_group.add("--distillation-temperature", type=float, default=2.0,  help="temperature for softening logits during distillation")

    train_group = parser.add_argument_group('Training')
    train_group.add( "-e", "--epochs",           type=int,   required=True, help="number of epochs for training")
    train_group.add("-ib", "--train-batch-size", type=int,   required=True, help="batch size for training")
    train_group.add("-vb", "--val-batch-size",   type=int,   required=True, help="batch size for validation")
    train_group.add("-ob", "--test-batch-size",  type=int,   required=True, help="batch size for testing")
    train_group.add("-is", "--train-crop-size",  type=int,   required=True, help="crop size for training")
    train_group.add("-vs", "--val-crop-size",    type=int,   required=True, help="crop size for validation")
    train_group.add("-os", "--test-crop-size",   type=int,   required=True, help="crop size for testing")
    train_group.add("-vz", "--val-size",         type=float, default=0.2,   help="split for validation set")
    train_group.add("-tz", "--test-size",        type=float, default=0.1,   help="split for test set")
    train_group.add("-rs", "--resume",           action="store_true",       help="resume training using stored model weights")

    data_group = parser.add_argument_group('Data')
    data_group.add( "-d", "--data-folder",     type=str, required=True, help="directory with aerial image and label data")
    data_group.add("-hf", "--hdf5-file",       type=str, required=True, help="hdf5 file containing data")
    data_group.add("-ic", "--input-channels",  type=int, required=True, help="number of input channels")
    data_group.add("-oc", "--output-channels", type=int, required=True, help="number of output channels")
    
    output_group = parser.add_argument_group('Output')
    output_group.add("-o", "--output-dir", type=str, default="output", help="directory to save output files")
    
    conf, _ = parser.parse_known_args()
    
    logger.info("Configuration successfully loaded.")
    
    return conf
