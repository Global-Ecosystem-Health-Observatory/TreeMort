#!/bin/bash
#SBATCH --job-name=treemort-creator
#SBATCH --account=project_2008436
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH --time=02:00:00 --partition=small
#SBATCH --mem-per-cpu=4000

module load python-data
pip install --user --upgrade hdf5 configargparse

srun python3 ./dataset/creator.py ./configs/AerialImageModel_ITD.txt

# module load python-data
# pip install --user --upgrade hdf5 configargparse

