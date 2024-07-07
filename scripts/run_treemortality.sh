#!/bin/bash
#SBATCH --job-name=tree-mortality
#SBATCH --account=project_2008436
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=36:00:00 --partition=gpu
#SBATCH --gres=gpu:v100:1

module load tensorflow
export PYTHONPATH=$PWD:$PYTHONPATH

srun python3 train_net.py ./configs/kokonet_bs8_cs256.txt --name kokonet_focal 
