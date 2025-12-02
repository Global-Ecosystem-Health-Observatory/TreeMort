#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small-g"
    MODULE_NAME="pytorch/2.7"
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

export HF_HOME="$TREEMORT_DATA_PATH/huggingface_cache"

$MODULE_USE_CMD
echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
    VENV_PY="\$TREEMORT_VENV_PATH/bin/python3"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
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

# Force venv and repo on path
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$TREEMORT_REPO_PATH"

if [ -z "\$VENV_PY" ] || [ ! -x "\$VENV_PY" ]; then
    echo "[ERROR] VENV_PY is not set or not executable: '\$VENV_PY'"
    exit 1
fi

POST_PROCESS=""
LIST_FILE=""

while [[ "\$#" -gt 0 ]]; do
    case "\$1" in
        --post-process)
            POST_PROCESS="--post-process"
            shift
            ;;
        --list-file)
            if [[ -n "\$2" ]]; then
                LIST_FILE="\$2"
                shift 2
            else
                echo "[ERROR] --list-file requires a filename argument."
                exit 1
            fi
            ;;
        *)
            echo "[ERROR] Unknown parameter passed: \$1"
            exit 1
            ;;
    esac
done

if [ -n "$POST_PROCESS" ]; then
    echo "[INFO] Post-processing is enabled"
fi
if [[ -n "$LIST_FILE" ]]; then
    echo "[INFO] Processing only files listed in: $LIST_FILE"
fi

echo "[INFO] Pre-downloading Beit and Maskformer models..."
rm -rf "$TREEMORT_DATA_PATH/huggingface_cache/microsoft/beit-base-finetuned-ade-640-640"
"$VENV_PY" -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/beit-base-finetuned-ade-640-640', cache_dir='$TREEMORT_DATA_PATH/huggingface_cache')"

rm -rf "$TREEMORT_DATA_PATH/huggingface_cache/facebook/maskformer-swin-base-ade"
"$VENV_PY" -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/maskformer-swin-base-ade', cache_dir='$TREEMORT_DATA_PATH/huggingface_cache')"

rm -rf "$TREEMORT_DATA_PATH/huggingface_cache/facebook/detr-resnet-50-panoptic"
"$VENV_PY" -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/detr-resnet-50-panoptic', cache_dir='$TREEMORT_DATA_PATH/huggingface_cache')"

echo "[INFO] Starting inference..."
srun "\$VENV_PY" "$TREEMORT_REPO_PATH/inference/engine.py" \
    "$DATA_PATH" \
    --config "$CONFIG_PATH" \
    --model-config "$MODEL_CONFIG_PATH" \
    --data-config "$DATA_CONFIG_PATH" \
    --outdir "$OUTPUT_PATH" \
    $POST_PROCESS \
    ${LIST_FILE:+--list-file "$LIST_FILE"}

EXIT_STATUS=$?
if [ "$EXIT_STATUS" -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job
sbatch -- $SBATCH_SCRIPT "$@"
