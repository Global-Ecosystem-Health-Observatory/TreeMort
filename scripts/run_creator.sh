#!/bin/bash
#SBATCH --job-name=treemort-creator
#SBATCH --account=project_2008436
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH --time=02:00:00 --partition=small
#SBATCH --mem-per-cpu=4000

module load tensorflow
source venv/bin/activate

srun python3 ./dataset/creator.py ./configs/AerialImageModel_ITD.txt
