#!/bin/bash
#SBATCH --job-name=treemort-refine
#SBATCH --account=project_2004205
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:v100:1

# Usage:
# export TREEMORT_VENV_PATH="/custom/path/to/venv"
# export TREEMORT_REPO_PATH="/custom/path/to/treemort/repo"
# sbatch --export=ALL,DATA_PATH="/custom/path/to/data" run_refine.sh

MODULE_NAME="pytorch/2.3"
module load $MODULE_NAME

TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH:-/users/rahmanan/TreeMort}"

REFINE_PATH="${TREEMORT_REPO_PATH}/misc/refine.py"

if [ ! -f "$REFINE_PATH" ]; then
    echo "[ERROR] Model refinement source file not found at $REFINE_PATH"
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo "[ERROR] DATA_PATH variable is not set. Please provide a data path using --export."
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory not found at $DATA_PATH"
    exit 1
fi

echo "[INFO] Starting refinement with the following settings:"
echo "       Data path: $DATA_PATH"
echo "       Inference engine: $REFINE_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"

srun python3 "$REFINE_PATH" "$DATA_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
