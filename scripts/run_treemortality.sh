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
export SM_FRAMEWORK="tf.keras"

# Check if at least the config file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch <sbatch script> <config file> [--eval_only true|false]"
    exit 1
fi

CONFIG_FILE=$1
EVAL_ONLY=false

# Check for additional arguments
while [[ "$#" -gt 1 ]]; do
    case $2 in
        --eval_only) EVAL_ONLY="$3"; shift ;;
    esac
    shift
done

# Adjust partition and time if eval_only is true
if [ "$EVAL_ONLY" = true ]; then
    #SBATCH --time=00:14:59
    #SBATCH --partition=gputest
    srun python3 train_net.py "$CONFIG_FILE" --eval_only
else
    srun python3 train_net.py "$CONFIG_FILE"
fi