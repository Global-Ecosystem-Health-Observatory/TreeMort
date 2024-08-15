#!/bin/bash
#SBATCH --job-name=treemort-creator
#SBATCH --account=project_2008436
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH --time=05:00:00 --partition=small
#SBATCH --mem-per-cpu=6000

module load python-data
source venv/bin/activate

srun python3 ./dataset/creator.py ./configs/AerialImageModel_ITD.txt

