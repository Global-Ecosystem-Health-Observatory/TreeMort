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

# Ensure TREEMORT_VENV_PATH is set (fallback to defaults if missing)
if [ -z "${TREEMORT_VENV_PATH:-}" ]; then
    if [ "$HPC_TYPE" == "lumi" ]; then
        TREEMORT_VENV_PATH="/projappl/project_462001070/aurahman/venv"
    elif [ "$HPC_TYPE" == "puhti" ]; then
        TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
    fi
fi

# Parse optional flags passed to this wrapper
POST_PROCESS=""
LIST_FILE=""
while [[ $# -gt 0 ]]; do
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
        *)
            break
            ;;
    esac
done

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
    echo "[INFO] VENV_PY resolved to: \$VENV_PY"
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

# Ensure MIOpen cache path is writable (apply to both user DB and cache dir)
if [ -z "$MIOPEN_USER_DB_PATH" ]; then
    MIOPEN_USER_DB_PATH="/tmp/miopen_cache_${SLURM_JOB_ID:-$$}"
fi
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_USER_DB_PATH
export MIOPEN_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export HIP_CACHE_DIR="$MIOPEN_USER_DB_PATH"
echo "[INFO] MIOpen cache path: $MIOPEN_USER_DB_PATH"

if [ -z "\$VENV_PY" ] || [ ! -x "\$VENV_PY" ]; then
    echo "[ERROR] VENV_PY is not set or not executable: '\$VENV_PY'"
    exit 1
fi

[ -n "$POST_PROCESS" ] && echo "[INFO] Post-processing is enabled"
[ -n "$LIST_FILE" ] && echo "[INFO] Processing only files listed in: $LIST_FILE"

echo "[INFO] Starting inference..."
CMD=(srun "\$VENV_PY" "$TREEMORT_REPO_PATH/inference/engine.py"
    "$DATA_PATH"
    --config "$CONFIG_PATH"
    --model-config "$MODEL_CONFIG_PATH"
    --data-config "$DATA_CONFIG_PATH"
    --outdir "$OUTPUT_PATH")
[ -n "$POST_PROCESS" ] && CMD+=("$POST_PROCESS")
[ -n "$LIST_FILE" ] && CMD+=(--list-file "$LIST_FILE")
printf "[INFO] Command: %q " "\${CMD[@]}"; echo
"\${CMD[@]}"

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
sbatch --export=ALL -- $SBATCH_SCRIPT "$@"
