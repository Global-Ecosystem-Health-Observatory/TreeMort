# TreeMort Segmentation

## Getting Started (Puhti)

**Step A.** Clone TreeMort repository and install it.

```bash
git clone https://github.com/Global-Ecosystem-Health-Observatory/TreeMort.git

export TREEMORT_VENV_PATH="/path/to/venv"
export TREEMORT_REPO_PATH="/path/to/package"

sh $TREEMORT_REPO_PATH/scripts/install_treemort.sh
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
sbatch --export=ALL,CONFIG_PATH="/custom/path/to/config",CHUNK_SIZE=<integer> run_creator.sh
```

**Step C.** Train/Evaluate TreeMort

```bash
sh /path/to/scripts/run_treemort.sh /path/to/config.txt --eval-only <true/false>
```

## Results

| Model | Mean IOU Pixels | Mean IOU Trees | Mean IOU | Mean Balanced IOU | Mean Dice Score | Mean Adjusted Dice Score | Mean MCC |
| :---: | :-------------: | :------------: | -------: | :---------------: | :-------------: | :----------------------: | :------: |
| FLAIR-Unet | 0.263 | 0.774 | 0.861 | 0.996 | 0.313 | 0.773 | 0.318 |
| DeepLabV3+ | 0.261 | 0.790 | 0.867 | 0.996 | 0.311 | 0.786 | 0.316 |

**Step D.** Run inference engine.

```bash
sbatch --export=ALL,CONFIG_PATH="/custom/path/to/config",DATA_PATH="/custom/path/to/data" run_inference.sh
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
python -m dataset.creator ./configs/<config-file>.txt
```

- Train a model

```bash
python -m treemort.main ./configs/<config-file>.txt
```

- Evaluate the model

```bash
python -m treemort.main ./configs/<config-file>.txt --eval-only
```

- Perform inference

```bash
python -m inference.engine \
    /path/to/input/data \
    --config ./configs/<config-file>.txt \
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