# TreeMort: Dead Tree Detection and Segmentation with Hybrid Self-Attention U-Nets

## Overview
TreeMort is an open-source project for instance-level segmentation of standing dead trees in high-resolution aerial imagery, developed by the Global Ecosystem Health Observatory. This README provides instructions for running the TreeMort-3T-UNet model on HPC systems (LUMI or Puhti) via CSC – IT Center for Science, Finland. The model integrates a Self-Attention U-Net with multi-task learning (segmentation masks, centroid heatmaps, hybrid SDT-boundary maps) and a hybrid loss function (BCE, Dice, Focal, MSE), enhanced by a watershed-guided post-processing pipeline. It achieves a 41.5% improvement in Mean Tree IoU (0.371 vs. 0.262 for U-Net) and a 57% reduction in centroid error (3.70 px vs. 8.60 px), as detailed in our accepted manuscript (Rahman et al., 2025, Int J Appl Earth Obs Geoinf).

## Features
- Multi-task learning for segmentation, centroid localization, and boundary refinement.
- Hybrid loss function optimizing pixel-level and instance-level accuracy.
- Watershed post-processing for enhanced instance delineation.
- Support for RGB-NIR aerial imagery (0.25 m/pixel resolution).
- Optimized for HPC environments with parallel processing capabilities.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git
   cd TreeMort
   ```
2. Set up a virtual environment on the HPC node:
   ```bash
   module load python
   virtualenv venv
   source venv/bin/activate
   ```
3. Install dependencies (listed in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
   - Required packages: PyTorch (>=2.0), segmentation_models_pytorch (>=0.3), numpy, pandas, geopandas, scikit-learn, tqdm.
   - Ensure CUDA support is available on the HPC GPU nodes (e.g., NVIDIA A100).

## Usage
### Dataset Preparation
- Download aerial imagery and label data from Allas (CSC object storage):
  ```bash
  module load allas
  allas-conf
  swift download DRYTREE_Annotations -p dead_trees/Finland -D /scratch/project_2008436/rahmanan
  ```
- Convert to HDF5 format:
  ```bash
  bash scripts/submit_creator.sh lumi finland
  ```
  - Use `puhti` instead of `lumi` for Puhti HPC.
- Ensure dataset paths are configured in `configs/data/finland.txt` (e.g., `data-folder = /scratch/project_2008436/rahmanan/Finland/RGBNIR/25cm`).

### Training
- Configure hyperparameters in `configs/model/flair_unet_sdt.txt` (e.g., loss weights: mask=1.0, centroid=0.7, sdt=0.5, boundary=1.0).
- Submit training job:
  ```bash
  bash scripts/submit_treemort.sh lumi unet finland
  ```
  - Replace `lumi` with `puhti` for Puhti HPC.
- Training settings: 100 epochs with early stopping (patience 10) based on validation Mean Tree IoU, AdamW optimizer (lr=1e-4, weight decay=1e-4), batch size 8, with augmentations (random flip, rotation, brightness, contrast, multiplicative noise, gamma correction), executed on NVIDIA A100 GPU.

### Evaluation
- Submit evaluation job:
  ```bash
  bash scripts/submit_treemort.sh lumi unet finland --eval-only
  ```
  - Replace `lumi` with `puhti` for Puhti HPC.
- Results are saved in `output/eval` as CSV files (e.g., `eval_Predictions.csv`).

## Results
- **Performance**: Achieves Mean Pixel IoU 0.259, Mean Tree IoU 0.371, Instance F1-Score 0.59, and Centroid Error 3.70 px on test set (see manuscript for details).
- **Ablation Studies**: Demonstrates 35% Tree IoU gain from pretraining, 6% from self-attention, and 41.5% from post-processing (see manuscript for details).
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
This work was supported by the University of Eastern Finland and CSC – IT Center for Science, Finland. We thank the National Land Survey of Finland for providing the aerial imagery dataset.

## Contact
For questions, contact Anis Ur Rahman (aniskhan25@gmail.com).