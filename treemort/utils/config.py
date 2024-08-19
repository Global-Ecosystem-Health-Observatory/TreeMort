import os
import configargparse


def setup(config_file_path):
    assert os.path.exists(config_file_path), f"Config file not found at: {config_file_path}"

    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("-d",    "--data-folder",        type=str,   required=True,      help="directory with aerial image and label data")
    parser.add("-hf",   "--hdf5-file",          type=str,   required=True,      help="hdf5 file of aerial image and label data")
    parser.add("-m",    "--model",              type=str,   required=True,      help="neural network model name for training")
    parser.add("-b",    "--backbone",           type=str,   default=None,       help="neural network model for feature extraction and/or backbone")
    parser.add("-e",    "--epochs",             type=int,   required=True,      help="number of epochs for training")
    parser.add("-ib",   "--train-batch-size",   type=int,   required=True,      help="batch size for training")
    parser.add("-vb",   "--val-batch-size",     type=int,   required=True,      help="batch size for validation")
    parser.add("-ob",   "--test-batch-size",    type=int,   required=True,      help="batch size for testing")
    parser.add("-is",   "--train-crop-size",    type=int,   required=True,      help="crop size for training")
    parser.add("-vs",   "--val-crop-size",      type=int,   required=True,      help="crop size for validation")
    parser.add("-os",   "--test-crop-size",     type=int,   required=True,      help="crop size for testing")
    parser.add("-vz",   "--val-size",           type=float, default=0.2,        help="split for validation")
    parser.add("-tz",   "--test-size",          type=float, default=0.1,        help="split for testing")
    parser.add("-ic",   "--input-channels",     type=int,   required=True,      help="input channels for training")
    parser.add("-oc",   "--output-channels",    type=int,   required=True,      help="output channels for training")
    parser.add("-o",    "--output-dir",         type=str,   default="output",   help="output dir for TensorBoard and models")
    parser.add("-mw",   "--model-weights",      type=str,   default="latest",   help="weights file of trained model for training continuation")
    parser.add("-lr",   "--learning-rate",      type=float, default=2e-4,       help="learning rate of Adam optimizer for training")
    parser.add("-th",   "--threshold",          type=float, default=0.5,        help="threshold for foreground/background classifier")
    parser.add("-av",   "--activation",         type=str,   default="sigmoid",  help="activation function of output layer of network")
    parser.add("-ls",   "--loss",               type=str,   default="hybrid",   help="loss function of network")
    parser.add("-rs",   "--resume",             action="store_true",            help="resume training using stored model weights")
    
    conf, _ = parser.parse_known_args()

    return conf

