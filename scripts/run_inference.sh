#!/bin/bash

# Parse input flags
POST_PROCESS=""
LIST_FILE=""
CHUNKS=""
CHUNK_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --post-process)
            POST_PROCESS="--post-process"
            shift
            ;;
        --list-file)
            if [[ -n "$2" ]]; then
                LIST_FILE="$2"
                shift 2
            else
                echo "[ERROR] --list-file requires a filename argument."
                exit 1
            fi
            ;;
        --chunks)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                CHUNKS="$2"
                shift 2
            else
                echo "[ERROR] --chunks requires a numeric argument."
                exit 1
            fi
            ;;
        *)
            echo "[ERROR] Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# If chunking is requested, split the list file into CHUNKS parts
if [[ -n "$LIST_FILE" && -n "$CHUNKS" ]]; then
    echo "[INFO] Splitting $LIST_FILE into $CHUNKS chunks"
    TOTAL_LINES=$(wc -l < "$LIST_FILE" | tr -d ' ')
    CHUNK_SIZE=$(( (TOTAL_LINES + CHUNKS - 1) / CHUNKS ))
    mkdir -p "$TREEMORT_DATA_PATH/tmp"
    CHUNK_DIR=$(mktemp -d "$TREEMORT_DATA_PATH/tmp/chunk_dir.XXXXXX")
    split -l "$CHUNK_SIZE" "$LIST_FILE" "$CHUNK_DIR/chunk_"
    SUFFIX_LENGTH=${#CHUNKS}
    export SUFFIX_LENGTH
fi

if [[ -n "$POST_PROCESS" ]]; then
    echo "[INFO] Post-processing is enabled"
fi
if [[ -n "$LIST_FILE" ]]; then
    echo "[INFO] Processing only files listed in: $LIST_FILE"
fi

# Configure job array for individual image processing if list-file is provided
if [[ -n "$LIST_FILE" && -n "$CHUNKS" ]]; then
    ARRAY_DIRECTIVE="#SBATCH --array=1-$CHUNKS"
elif [[ -n "$LIST_FILE" ]]; then
    if [[ ! -f "$LIST_FILE" ]]; then
        echo "[ERROR] List file not found: $LIST_FILE"
        exit 1
    fi
    NUM_FILES=$(wc -l < "$LIST_FILE" | tr -d ' ')
    if (( NUM_FILES > 1 )); then
        ARRAY_DIRECTIVE="#SBATCH --array=1-$NUM_FILES"
        echo "[INFO] Using Slurm array for $NUM_FILES tasks"
    else
        echo "[INFO] Only $NUM_FILES image in list; running single-task job"
        ARRAY_DIRECTIVE=""
    fi
else
    ARRAY_DIRECTIVE=""
fi

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

SBATCH_SCRIPT=$(mktemp)

cat <<EOT > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --job-name=treemort-inference
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
$ARRAY_DIRECTIVE
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=$PARTITION_NAME
#SBATCH --mem=16G
$GPU_DIRECTIVE

export LIST_FILE="$LIST_FILE"
export POST_PROCESS="$POST_PROCESS"
export CHUNKS="$CHUNKS"
export CHUNK_DIR="$CHUNK_DIR"
export SUFFIX_LENGTH="$SUFFIX_LENGTH"

export TRANSFORMERS_CACHE="$TREEMORT_DATA_PATH/huggingface_cache"
export HF_HOME="$TREEMORT_DATA_PATH/huggingface_cache"

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

# If running as an array task and a list file is provided
if [[ -n "\$SLURM_ARRAY_TASK_ID" && -n "\$LIST_FILE" ]]; then
    if [[ -n "\$CHUNKS" ]]; then
        CHUNK_FILE="\$CHUNK_DIR/chunk_\$(printf "%0${SUFFIX_LENGTH}d" "${SLURM_ARRAY_TASK_ID}")"
        echo "[INFO] Array task #\${SLURM_ARRAY_TASK_ID} → processing chunk file \$CHUNK_FILE"
        srun python3 "\$TREEMORT_REPO_PATH/inference/engine.py" \
            "\$DATA_PATH" \
            --list-file "\$CHUNK_FILE" \
            --config "\$CONFIG_PATH" \
            --outdir "\$OUTPUT_PATH" \
            \$POST_PROCESS
        exit \$?
    else
        IMAGE_REL_PATH=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$LIST_FILE")
        IMAGE_PATH="\$DATA_PATH/\$IMAGE_REL_PATH"
        echo "[INFO] Array task #\${SLURM_ARRAY_TASK_ID} → \$IMAGE_PATH"
        srun python3 "\$TREEMORT_REPO_PATH/inference/engine.py" \
            "\$IMAGE_PATH" \
            --config "\$CONFIG_PATH" \
            --outdir "\$OUTPUT_PATH" \
            \$POST_PROCESS
        exit \$?
    fi
fi

# Otherwise (no array OR missing LIST_FILE), run on full $DATA_PATH
srun python3 "$TREEMORT_REPO_PATH/inference/engine.py" \
    "$DATA_PATH" \
    --config "$CONFIG_PATH" \
    --outdir "$OUTPUT_PATH" \
    $POST_PROCESS \
    ${LIST_FILE:+--list-file "$LIST_FILE"}

EXIT_STATUS=\$?
if [ \$EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status \$EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit \$EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

sbatch $SBATCH_SCRIPT