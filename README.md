# TreeMort: Dead Tree Detection and Segmentation with Hybrid Self-Attention U-Nets

## Overview
TreeMort is an open-source project for instance-level segmentation of standing dead trees in high-resolution aerial imagery, developed by the Global Ecosystem Health Observatory. This README provides instructions for running the TreeMort-3T-UNet model locally on your machine. The model integrates a Self-Attention U-Net with multi-task learning (segmentation masks, centroid heatmaps, hybrid SDT-boundary maps) and a hybrid loss function (BCE, Dice, Focal, MSE), enhanced by a watershed-guided post-processing pipeline. It achieves a 41.5% improvement in Mean Tree IoU (0.371 vs. 0.262 for U-Net) and a 57% reduction in centroid error (3.70 px vs. 8.60 px), as detailed in our accepted manuscript (Rahman et al., 2025, Int J Appl Earth Obs Geoinf).

## Features
- Multi-task learning for segmentation, centroid localization, and boundary refinement.
- Hybrid loss function optimizing pixel-level and instance-level accuracy.
- Watershed post-processing for enhanced instance delineation.
- Support for RGB-NIR aerial imagery (0.25 m/pixel resolution).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git
   cd TreeMort
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv --system-site-packages venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies (listed in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
   - Required packages: PyTorch (>=2.0), segmentation_models_pytorch (>=0.3), numpy, pandas, geopandas, scikit-learn, tqdm.
   - Ensure CUDA is installed if using a GPU (recommended for performance).

## Usage
### Dataset Preparation
- Curate your own aerial imagery dataset containing RGB-NIR images and annotated dead tree segmentations.
- The dataset should include paired TIFF images and corresponding GeoJSON annotation files for each region.
- Organize your dataset in a directory structure similar to the example below:
  ```
  Region/
    RGBNIR/
      Images/
        image_001.tif
        image_002.tif
        ...
      Geojsons/
        image_001.geojson
        image_002.geojson
        ...
  ```
- Modify the dataset configuration file located at `configs/data/<region>.txt` to reflect the paths to your images and annotations.
- Run the dataset creator module to convert your data into the required format:
  ```bash
  python3 -m dataset.creator configs/data/<region>.txt
  ```

#### Config Files Explained

The dataset creator uses configuration files in `configs/data/` to control how datasets are processed. There are two main types:

- **`base_config.txt`**: Contains default/common settings shared across all datasets. Example:

  ```
  # Common settings for all data
  window-size = 256
  stride = 128
  nir-rgb-order = [3, 0, 1, 2]
  normalize-channelwise = True
  ```
  - `window-size`: The size (in pixels) of the sliding window used to crop images and annotations into tiles.
  - `stride`: The step size (in pixels) between sliding windows (controls overlap between tiles).
  - `nir-rgb-order`: The order of channels in the input TIFF images (e.g., [NIR, R, G, B] = [3, 0, 1, 2]).
  - `normalize-channelwise`: Whether to normalize each channel independently.

- **`<region>.txt`** (e.g., `finland.txt`): Inherits from `base_config.txt` and specifies dataset-specific settings. Example:

  ```
  # Configurations for dataset creation for Finland
  include = base_config.txt

  data-folder = ${TREEMORT_DATA_PATH}/Finland/RGBNIR/25cm
  hdf5-file = Finland_RGBNIR_25cm.h5
  ```
  - `include = base_config.txt` tells the loader to use the base settings and override with any region-specific parameters below.
  - `data-folder`: Path to the folder containing images and geojsons for this region.
  - `hdf5-file`: Name of the output HDF5 file to be generated.

#### Model Config Files Explained

Model training uses configuration files located in `configs/model/` to control training parameters, model architecture, and data handling.

- **`base_config.txt`**: This file contains common settings shared across all model training runs. A typical example looks like:

  ```
  # Common settings for all models

  include = {data_config}
  data-config = {data_config}
  output-dir = ./output
  model = test
  train-crop-size = 256
  val-crop-size = 256
  test-crop-size = 256
  input-channels = 4
  output-channels = 1
  train-batch-size = 8
  val-batch-size = 8
  test-batch-size = 8
  epochs = 100
  class-weights = [0.1, 0.9]
  model-weights = best
  learning-rate = 1e-4
  segment-threshold = 0.5
  centroid-threshold = 0.4
  loss = hybrid
  activation = sigmoid
  val-size = 0.2
  test-size = 0.1
  ```

  Key parameters explained:
  - `include` and `data-config`: Specify the dataset configuration file to use for training data paths and preprocessing.
  - `train-crop-size`, `val-crop-size`, `test-crop-size`: Define the input crop sizes for training, validation, and testing phases.
  - `input-channels` and `output-channels`: Number of input image channels (e.g., 4 for RGB-NIR) and output mask channels.
  - `train-batch-size`, `val-batch-size`, `test-batch-size`: Batch sizes for training, validation, and testing.
  - `epochs`: Number of training epochs.
  - `class-weights`: Weights for classes in the loss function to handle class imbalance.
  - `model-weights`: Which model checkpoint to load for evaluation or resuming (e.g., 'best').
  - `learning-rate`: Initial learning rate for the optimizer.
  - `segment-threshold` and `centroid-threshold`: Thresholds for segmentation mask and centroid detection during inference.
  - `loss` and `activation`: Loss function type and output activation function.
  - `val-size` and `test-size`: Proportions of data reserved for validation and testing.

- **`flair_unet.txt`**: An example model configuration for the TreeMort-1T-UNet (self-attention U-Net with a FLAIR backbone):

  ```
  # settings for TreeMort-1T-UNet (self attention unet with flair backbone)

  include = base_config.txt

  model = flair_unet

  resume = True
  ```

  This file inherits all settings from `base_config.txt`, specifies the model architecture as `flair_unet`, and sets `resume = True` to continue training from the latest checkpoint if available.

Users can create their own model configuration files to experiment with different architectures or hyperparameters by inheriting from `base_config.txt` and overriding specific parameters as needed.

### Training
- Train the model using both model and data configuration files:
  ```bash
  python3 -m treemort.main configs/model/flair_unet_sdt.txt --data-config configs/data/finland.txt
  ```

### Evaluation
- Evaluate the trained model:
  ```bash
  python3 -m treemort.main configs/model/flair_unet_sdt.txt --data-config configs/data/finland.txt --eval-only
  ```

## Results
- **Performance**: Achieves Mean Pixel IoU 0.259, Mean Tree IoU 0.371, Instance F1-Score 0.59, and Centroid Error 3.70 px on test set (see manuscript for details).
- **Manuscript Reference**: Rahman, A. U., Heinaro, E., Ahishali, M., & Junttila, S. (2025). Dual-Task Learning for Dead Tree Detection and Segmentation with Hybrid Self-Attention U-Nets in Aerial Imagery. *Int J Appl Earth Obs Geoinf*. Accepted September 2025.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
- Contributions are welcome! Please fork the repository, create a feature branch, and submit pull requests.
- Report issues or suggestions via GitHub Issues.
- Cite our work if used:  
  ```bibtex
  @article{RAHMAN2025104851,
    title = {Dual-task learning for dead tree detection and segmentation with hybrid self-attention U-Nets in aerial imagery},
    journal = {International Journal of Applied Earth Observation and Geoinformation},
    volume = {144},
    pages = {104851},
    year = {2025},
    issn = {1569-8432},
    doi = {https://doi.org/10.1016/j.jag.2025.104851},
    url = {https://www.sciencedirect.com/science/article/pii/S1569843225004984},
    author = {Anis Ur Rahman and Einari Heinaro and Mete Ahishali and Samuli Junttila}
  }
  ```

## Acknowledgments
This work was supported by the University of Eastern Finland. We thank the National Land Survey of Finland for providing the aerial imagery dataset.

## Contact
For questions, contact Anis Ur Rahman (aniskhan25@gmail.com).