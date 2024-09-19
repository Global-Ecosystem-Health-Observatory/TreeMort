# TreeMort Segmentation

## Getting Started (Puhti)

**Step A.** Clone TreeMort repository and install it.

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git
cd TreeMort

sh ./scripts/install_treemort.sh
```

**Step B.** Upload aerial image and label data from local machine to Allas cloud storage. Move to Step D, if the dataset in HDF5 is already created and available on scratch.

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

source allas_conf -u rahmanan
```

3. Upload aerial image and label data to Allas.

```bash
cd ~/Documents/AerialImages

swift upload DRYTREE-project-AerialImageModel_ITD --skip-identical ./4band_25cm/*.*
swift upload DRYTREE-project-AerialImageModel_ITD --skip-identical ./Geojsons/*.*
```

4. Restore zsh

```bash
chsh -s /bin/zsh
echo $SHELL
```

**Step C.** Create dataset in HDF5 format. 

1. Download aerial image and label data to scratch.

```bash
module load allas
allas-conf

swift download DRYTREE_Annotations -p dead_trees/Finland -D /scratch/project_2008436/rahmanan/dead_trees/Finland
```

2. Run script to create the HDF5 dataset.

```bash
cd ~/TreeMort

sbatch ./scripts/run_creator.sh
```

**Step D.** Demo TreeMort

1. Train a model.

```bash
sh ./scripts/run_treemort.sh ./configs/unet_bs8_cs256.txt --eval-only false
```

2.  Evaluate the model.

```bash
sh ./scripts/run_treemort.sh ./configs/unet_bs8_cs256.txt --eval-only true
```

## Results

|          Model                      |  Mean IOU Pixels  |  Mean IOU Trees  |
| :---------------------------------: | :---------------: | :--------------: |
| Unet with Self Attention            | 0.281             | 0.749            |
| FLAIR (Encoder) + SA-Unet (Decoder) | 0.291             | 0.785            |

| Model      | Mean IOU Pixels | Mean IOU Trees | Mean IOU | Mean Balanced IOU | Mean Dice Score | Mean Adjusted Dice Score | Mean MCC | 
| :--------: | :-------------: | :------------: ---------: | :---------------: | :-------------: | :----------------------: | :------: |
| SA-Unet    | 0.281 | 0.749 | 0.840 | 0.997 | 0.334 | 0.736 | 0.342 |
| SA-Unet MS | 0.272 | 0.738 | 0.837 | 0.996 | 0.323 | 0.727 | 0.331 | 
| FLAIR-Unet | 0.291 | 0.749 | 0.845 | 0.996 | 0.346 | 0.747 | 0.354 |
| HCFNet     | 0.283 | 0.742 | 0.842 | 0.996 | 0.336 | 0.740 | 0.345 |
| DeepLabV3+ | 0.287 | 0.773 | 0.856 | 0.997 | 0.342 | 0.769 | 0.350 |

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

python -m pip install treemort@git+https://github.com/aniskhan25/climate-pix.git
```

- Train a model

```bash
python -m treemort.main ./configs/kokonet_bs8_cs256.txt
```

- Evaluate the model
```bash
python -m treemort.main ./configs/kokonet_bs8_cs256.txt --eval-only
```

- Running from source, refer to ./colab/treemort_colab.ipynb
