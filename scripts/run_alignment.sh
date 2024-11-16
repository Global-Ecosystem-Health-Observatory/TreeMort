#!/bin/bash
#SBATCH --job-name=image-aligning              # Job name
#SBATCH --account=project_2004205              # Project account
#SBATCH --output=output/stdout/%A_%a.out       # Output log path
#SBATCH --error=output/stderr/%A_%a.err        # Error log path
#SBATCH --ntasks=1                             # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                      # Number of CPU cores per task
#SBATCH --time=05:00:00                        # Time limit (hh:mm:ss)
#SBATCH --partition=small                      # Partition to submit to
#SBATCH --mem-per-cpu=10000                     # Memory per CPU in MB

# Usage:
# export TREEMORT_VENV_PATH="/custom/path/to/venv"
# export TREEMORT_REPO_PATH="/custom/path/to/treemort/repo"
# sbatch --export=ALL,SCRIPT_PATH="/custom/path/to/function.py",DATA_PATH="/custom/path/to/data" run_alignment.sh

MODULE_NAME="python-data"
module load $MODULE_NAME

VENV_PATH="${VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"

if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    exit 1
fi

if [ -z "$SCRIPT_PATH" ]; then
    echo "[ERROR] SCRIPT_PATH variable is not set. Please provide a script path using --export."
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
    echo "[ERROR] DATA_PATH variable is not set. Please provide a data path using --export."
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory not found at $DATA_PATH"
    exit 1
fi

echo "[INFO] Starting image alignment with the following settings:"
echo "       Script path: $SCRIPT_PATH"
echo "       Data path: $DATA_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"

srun python3 "$SCRIPT_PATH" "$DATA_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS

'''

Usage:

sbatch --export=ALL,SCRIPT_PATH="$TREEMORT_REPO_PATH/misc/register.py",DATA_PATH="./Finland_CHM" $TREEMORT_REPO_PATH/scripts/run_alignment.sh

'''