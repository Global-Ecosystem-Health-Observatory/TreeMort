#!/bin/bash
#SBATCH --job-name=tree-mort
#SBATCH --account=project_2004205
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# Usage:
# export VENV_PATH="/custom/path/to/venv"
# sbatch --export=CONFIG_PATH="/custom/path/to/config" run_fisher.sh

MODULE_NAME="pytorch/2.3"

TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"

echo "Loading module: $MODULE_NAME"
module load "$MODULE_NAME"

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

srun python3 -c "import torch; print(\"TORCH\")"
srun python3 -c "import treemort; print(\"TREEMORT\")"

