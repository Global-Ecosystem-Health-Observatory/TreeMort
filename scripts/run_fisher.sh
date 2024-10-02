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
module load $MODULE_NAME

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
    echo "[ERROR] CONFIG_PATH variable is not set. Please provide a config path using --export."
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found at $CONFIG_PATH"
    exit 1
fi

echo "[INFO] Starting dataset creation with the following settings:"
echo "       Config file: $CONFIG_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "       Job time limit: $SLURM_TIMELIMIT"

srun python -m treemort.utils.fisher "$CONFIG_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Fisher matrix creation failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Fisher matrix creation completed successfully"
fi

exit $EXIT_STATUS
