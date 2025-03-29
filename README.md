# TreeMort Segmentation

## Getting Started (Puhti)

**Step A.** Clone TreeMort repository and install it.

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git

export TREEMORT_VENV_PATH="/path/to/venv"
export TREEMORT_REPO_PATH="/path/to/package"

bash $TREEMORT_REPO_PATH/scripts/install_treemort.sh
```

**Step B.** Create dataset in HDF5 format.

1. Download aerial image and label data to scratch.

```bash
module load allas
allas-conf

swift download <container_name> -p <remote_directory> -D <local_directory>
```

e.g. download Finnish aerial imagery from Allas

```bash
swift download DRYTREE_Annotations -p dead_trees/Finland -D /scratch/project_2008436/rahmanan
```

2. Run script to create the HDF5 dataset.

```bash
bash /path/to/submit_creator.sh <hpc_type> <data config file>
```

e.g. create HDF5 for Finnish aerial imagery downloaded from Allas to /scratch

```bash
bash $TREEMORT_REPO_PATH/scripts/submit_creator.sh lumi finland
```

**Step C.** Train/Evaluate TreeMort

```bash
bash /path/to/submit_treemort.sh <hpc_type> <model config file> <data config file> [--eval-only]
```

e.g. train and test U-Net for Finnish aerial imagery

```bash
bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland
```

e.g. test U-Net for Finnish aerial imagery

```bash
bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland --eval-only
```

## Results

| METHOD | MEAN IOU PIXELS | MEAN IOU TREES | PRECISION | RECALL | F1-SCORE | CENTROID ERROR (PX) |
| :----: | :-------------: | :------------: | :-------: | :----: | :------: | :-----------------: |
| KokoNet                   | 0.0817 | 0.2158 | 0.6557 | 0.3930 | 0.3864 | 8.89 |
| DETR                      | 0.1546 | 0.2479 | 0.7098 | 0.4403 | 0.4545 | 7.50 |
| Self-Attention UNet       | 0.1731 | 0.2752 | 0.6407 | 0.4801 | 0.4753 | 6.09 |
| DINOv2                    | 0.1781 | 0.2722 | 0.6663 | 0.5349 | 0.5130 | 4.74 |
| BEiT                      | 0.1783 | 0.2817 | 0.7207 | 0.4784 | 0.5021 | 7.37 |
| PSPNet                    | 0.2093 | 0.3234 | 0.6372 | 0.6463 | 0.5703 | 3.12 |
| FPN                       | 0.2147 | 0.3251 | 0.7198 | 0.5868 | 0.5770 | 4.15 |
| DeepLabv3+                | 0.2288 | 0.3263 | 0.6657 | 0.6015 | 0.5691 | 4.77 |
| HCF-Net                   | 0.2365 | 0.3336 | 0.6903 | 0.5993 | 0.5501 | 4.36 |
| MaskFormer                | 0.2397 | 0.3503 | 0.6202 | 0.6882 | 0.5774 | 3.04 |
| U-Net (EfficientNet-B7)   | 0.2428 | 0.3420 | 0.6319 | 0.6684 | 0.5619 | 2.71 |
| TreeMort-3T-UNet          | 0.2588 | 0.3710 | 0.6418 | 0.6687 | 0.5895 | 3.70 |

**Step D.** Run inference engine.

```bash
bash /path/to/submit_inference.sh <hpc_type> <model config file> <data config file> [--post-process]
```

e.g. inference without post-processing using U-Net for Finnish aerial imagery

```bash
bash $TREEMORT_REPO_PATH/scripts/submit_inference.sh lumi unet finland
```

e.g. inference with post-processing using U-Net for Finnish aerial imagery

```bash
bash $TREEMORT_REPO_PATH/scripts/submit_inference.sh lumi unet finland --post-process
```

## Getting Started (Mac OS, for development)

- Clone repository

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git
cd TreeMort
```

- Create a new environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -e .
```

- Create a dataset

```bash
python -m dataset.creator <data configfile>
```

- Train/evaluate a model

```bash
python -m treemort.main <config file> --data-config <data config file> [--eval-only]
```

- Perform inference

```bash
python -m inference.engine \
    /path/to/input/data \
    --config <inference config file> \
    --model-config <model config file> \
    --data-config <data config file> \
    --outdir /path/to/output/directory
```

- Running from source, refer to ./colab/treemort_colab.ipynb

## (Optional) Upload own data

**Method I.** Copy local directory to remote server using scp

```bash
scp -O -r /path/to/local/directory <username>@puhti.csc.fi:/path/to/remote/directory
```

**Method II.** Upload files to Allas container using swift
 
1. Clone allas utils and change shell to bash, as zsh is not supported.

```bash
git clone https://github.com/CSCfi/allas-cli-utils.git

chsh -s /bin/bash
exit
```

2. Restart terminal in bash. Create a new virtual environment.

```bash
echo $SHELL

cd allas-cli-utils

python3 -m venv atools-venv
source atools-venv/bin/activate

pip3 install --upgrade pip
pip3 install openstackclient python-swiftclient

source allas_conf -u <username>
```

3. Upload aerial image and label data to Allas.

```bash
swift upload <container-name> --skip-identical /path/to/files/*.*
```

4. Restore zsh

```bash
chsh -s /bin/zsh
echo $SHELL
```