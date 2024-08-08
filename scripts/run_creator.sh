#!/bin/bash
#SBATCH --job-name=tree-mort-creator
#SBATCH --account=project_462000508
#SBATCH --output=stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH --time=02:00:00 --partition=small
#SBATCH --mem-per-cpu=4000

module load python-data

srun python3 python dataset/creator.py ./configs/AerialImageModel_ITD.txt