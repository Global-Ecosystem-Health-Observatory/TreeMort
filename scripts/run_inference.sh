#!/bin/bash
#SBATCH --job-name=treemort-inference              # Job name
#SBATCH --account=project_2004205                  # Project account
#SBATCH --output=output/stdout/%A_%a.out           # Output log
#SBATCH --error=output/stderr/%A_%a.err            # Error log
#SBATCH --ntasks=1                                 # Number of tasks (1 process)
#SBATCH --cpus-per-task=6                          # Number of CPU cores per task
#SBATCH --time=05:00:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=small                          # Partition to submit to
#SBATCH --mem-per-cpu=24000                        # Memory per CPU in MB (6GB per CPU)

# Usage:
# export TREEMORT_VENV_PATH="/custom/path/to/venv"
# export TREEMORT_REPO_PATH="/custom/path/to/treemort/repo"
# sbatch --export=ALL,CONFIG_PATH="/custom/path/to/config",DATA_PATH="/custom/path/to/data",OUTPUT_PATH="/custom/path/to/output" run_inference.sh

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

ENGINE_PATH="${TREEMORT_REPO_PATH}/inference/engine.py"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "[ERROR] Inference engine source file not found at $ENGINE_PATH"
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

if [ -z "$DATA_PATH" ]; then
    echo "[ERROR] DATA_PATH variable is not set. Please provide a data path using --export."
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "[ERROR] OUTPUT_PATH variable is not set. Please provide a data path using --export."
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory not found at $DATA_PATH"
    exit 1
fi

if [ ! -d "$OUTPUT_PATH" ]; then
    echo "[WARNING] Output directory not found at $OUTPUT_PATH. Creating it now..."
    mkdir -p "$OUTPUT_PATH" || { echo "[ERROR] Failed to create output directory at $OUTPUT_PATH"; exit 1; }
fi

POST_PROCESS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --post-process) POST_PROCESS="--post-process"; shift ;;
        *) echo "[ERROR] Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -n "$POST_PROCESS" ]; then
    echo "[INFO] Post-processing is enabled"
fi

echo "[INFO] Starting inference with the following settings:"
echo "       Data path: $DATA_PATH"
echo "       Output path: $OUTPUT_PATH"
echo "       Config file: $CONFIG_PATH"
echo "       Inference engine: $ENGINE_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"

srun python3 "$ENGINE_PATH" "$DATA_PATH" --config "$CONFIG_PATH" --outdir "$OUTPUT_PATH" $POST_PROCESS

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
