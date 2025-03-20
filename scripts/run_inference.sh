#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462000684"
    PARTITION_NAME="small-g"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD="module use /appl/local/csc/modulefiles/"
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=1"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD=""
    GPU_DIRECTIVE="#SBATCH --gres=gpu:v100:1"
fi

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

# SLURM Job Configuration
cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-inference
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --partition=$PARTITION_NAME
#SBATCH --mem-per-cpu=24000
$GPU_DIRECTIVE

$MODULE_USE_CMD
echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

ENGINE_PATH="${TREEMORT_REPO_PATH}/inference/engine.py"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "[ERROR] Inference engine source file not found at $ENGINE_PATH"
    exit 1
fi

if [ -z "$CONFIG_PATH" ] || [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file is missing or invalid."
    exit 1
fi

if [ -z "$MODEL_CONFIG_PATH" ] || [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "[ERROR] Model config file is missing or invalid."
    exit 1
fi

if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid."
    exit 1
fi

if [ -z "$DATA_PATH" ] || [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory is missing or invalid."
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "[ERROR] Output directory is not set."
    exit 1
elif [ ! -d "$OUTPUT_PATH" ]; then
    echo "[WARNING] Output directory not found at $OUTPUT_PATH. Creating it now..."
    mkdir -p "$OUTPUT_PATH" || { echo "[ERROR] Failed to create output directory."; exit 1; }
fi

POST_PROCESS=""

while [[ "$#" -gt 0 ]]; do
    if [ -z "$1" ]; then
        break
    fi
    case $1 in
        --post-process) POST_PROCESS="--post-process" ;;
        *) echo "[ERROR] Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$POST_PROCESS" ]; then
    echo "[INFO] Post-processing is enabled"
fi

echo "[INFO] Starting inference..."
srun python3 "$ENGINE_PATH" "$DATA_PATH" --config "$CONFIG_PATH" --model-config "$MODEL_CONFIG_PATH" --data-config "$DATA_CONFIG_PATH" --outdir "$OUTPUT_PATH" $POST_PROCESS

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job
bash $SBATCH_SCRIPT "$@"