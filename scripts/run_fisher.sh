#!/bin/bash
#SBATCH --job-name=treemort-creator                # Job name
#SBATCH --account=project_2004205                  # Project account
#SBATCH --output=output/stdout/%A_%a.out           # Output log
#SBATCH --error=output/stderr/%A_%a.err            # Error log
#SBATCH --ntasks=1                                 # Number of tasks (1 process)
#SBATCH --cpus-per-task=6                          # Number of CPU cores per task
#SBATCH --time=00:05:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=small                          # Partition to submit to
#SBATCH --mem-per-cpu=6000                         # Memory per CPU in MB (6GB per CPU)

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

srun python -c "import treemort; print('TREEMORT')"