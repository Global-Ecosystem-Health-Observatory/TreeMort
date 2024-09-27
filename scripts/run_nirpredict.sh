#!/bin/bash
#SBATCH --job-name=nir-predict                     # Job name
#SBATCH --account=project_2004205                  # Project account
#SBATCH --output=output/stdout/%A_%a.out           # Output log
#SBATCH --error=output/stderr/%A_%a.err            # Error log
#SBATCH --ntasks=1                                 # Number of tasks (1 process)
#SBATCH --cpus-per-task=4                          # Number of CPU cores per task
#SBATCH --time=05:00:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                            # Partition to submit to
#SBATCH --gres=gpu:v100:1                          # Number and type of GPU
#SBATCH --mem-per-cpu=32G                          # Memory per GPU in GB

module load pytorch/2.3

VENV_PATH="/projappl/project_2004205/rahmanan/venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    exit 1
fi

CONFIG_PATH="/users/rahmanan/TreeMort/configs/nirpredict.txt"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found at $CONFIG_PATH"
    exit 1
fi

echo "[INFO] Starting dataset creation with the following settings:"
echo "       Config file: $CONFIG_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "       Job time limit: $SLURM_TIMELIMIT"

srun python3 -m nirpredict.main "$CONFIG_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] NIR Predictor failed with exit status $EXIT_STATUS"
else
    echo "[INFO] NIR Predictor completed successfully"
fi

exit $EXIT_STATUS