import configargparse


def setup(config_file_path):
    parser = configargparse.ArgParser(
        default_config_files=[config_file_path])

    parser.add("-d",    "--data-folder",        type=str,   required=True,      help="directory of input samples for training and testing")
    parser.add("-m",    "--model",              type=str,   required=True,      help="neural network model name for training")
    parser.add("-e",    "--epochs",             type=int,   required=True,      help="number of epochs for training")
    parser.add("-ib",   "--train-batch-size",   type=int,   required=True,      help="batch size for training")
    parser.add("-ob",   "--test-batch-size",    type=int,   required=True,      help="batch size for testing")
    parser.add("-is",   "--train-crop-size",    type=int,   required=True,      help="crop size for training")
    parser.add("-os",   "--test-crop-size",     type=int,   required=True,      help="crop size for testing")
    parser.add("-vs",   "--val-size",           type=float, default=0.2,        help="split for validation")
    parser.add("-ic",   "--input-channels",     type=int,   required=True,      help="input channels for training")
    parser.add("-oc",   "--output-channels",    type=int,   required=True,      help="output channels for training")
    parser.add("-o",    "--output-dir",         type=str,   default="output",   help="output dir for TensorBoard and models")
    parser.add("-mw",   "--model-weights",      type=str,   default="latest",   help="weights file of trained model for training continuation")
    parser.add("-lr",   "--learning-rate",      type=float, default=2e-4,       help="learning rate of Adam optimizer for training")
    parser.add("-th",   "--threshold",          type=float, default=-0.5,       help="threshold for foreground/background classifier")
    parser.add("-av",   "--activation",         type=str,   default="tanh",     help="activation function of output layer of network")
    parser.add("-ls",   "--loss",               type=str,   default="mse",      help="loss function of network")
    parser.add("-rs",   "--resume",             action="store_true",            help="resume training using stored model weights")
    parser.add("-bn",   "--binarize",           action="store_true",            help="binarize data labels to foreground and background mask")
    
    conf, _ = parser.parse_known_args()

    return conf

