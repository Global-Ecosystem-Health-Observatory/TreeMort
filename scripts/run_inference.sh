#!/bin/bash
#SBATCH --job-name=treemort-inference
#SBATCH --account=project_2004205
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --partition=small
#SBATCH --mem-per-cpu=6000

module load python-data

VENV_PATH="/projappl/project_2004205/rahmanan/venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    exit 1
fi

DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees/Finland/RGBNIR/25cm"
CONFIG_PATH="/users/rahmanan/TreeMort/configs/Finland_RGBNIR_25cm_inference.txt"

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory not found at $DATA_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found at $CONFIG_PATH"
    exit 1
fi

echo "[INFO] Starting inference with the following settings:"
echo "       Data path: $DATA_PATH"
echo "       Config file: $CONFIG_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "       Job time limit: $SLURM_TIMELIMIT"

srun python3 -m inference.engine "$DATA_PATH" --config "$CONFIG_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS