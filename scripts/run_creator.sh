#!/bin/bash
#SBATCH --job-name=treemort-creator
#SBATCH --account=project_2004205
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH --time=05:00:00 --partition=small
#SBATCH --mem-per-cpu=6000

module load python-data
source /projappl/project_2004205/rahmanan/venv/bin/activate

srun python3 -m dataset.creator ./configs/dead_trees_finland.txt
