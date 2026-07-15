#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small-g"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
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

# Parse optional flags passed to this wrapper (before SBATCH submission)
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

# Build inference args at generation time (baked as literals in the SBATCH script).
# No inner quotes — paths are literal and HPC paths never contain spaces.
INFER_ARGS="$DATA_PATH --config $CONFIG_PATH --model-config $MODEL_CONFIG_PATH --data-config $DATA_CONFIG_PATH --outdir $OUTPUT_PATH"
[ -n "$POST_PROCESS" ] && INFER_ARGS="$INFER_ARGS --post-process"
[ -n "$LIST_FILE" ] && INFER_ARGS="$INFER_ARGS --list-file $LIST_FILE"

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

if [ "$HPC_TYPE" == "lumi" ]; then

    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-inference
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=05:00:00
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Remove old CSC AI paths — lumi-aif-singularity-bindings refuses to run if they are present
PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

export HF_HOME="$TREEMORT_DATA_PATH/huggingface_cache"

if [ -z "$CONFIG_PATH" ] || [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file is missing or invalid: $CONFIG_PATH"
    exit 1
fi

if [ -z "$MODEL_CONFIG_PATH" ] || [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "[ERROR] Model config file is missing or invalid: $MODEL_CONFIG_PATH"
    exit 1
fi

if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid: $DATA_CONFIG_PATH"
    exit 1
fi

if [ -z "$DATA_PATH" ] || [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data directory is missing or invalid: $DATA_PATH"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "[ERROR] Output directory is not set."
    exit 1
elif [ ! -d "$OUTPUT_PATH" ]; then
    echo "[WARNING] Output directory not found at $OUTPUT_PATH. Creating it now..."
    mkdir -p "$OUTPUT_PATH" || { echo "[ERROR] Failed to create output directory."; exit 1; }
fi

echo "[INFO] Starting inference..."
srun singularity run "\${SIF}" bash -c 'source $TREEMORT_VENV_PATH/bin/activate && PYTHONNOUSERSITE=1 PYTHONPATH=$TREEMORT_REPO_PATH python $TREEMORT_REPO_PATH/inference/engine.py $INFER_ARGS'
EOT

else
    # Puhti / default: module + venv approach
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

module load pytorch/2.5

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

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$TREEMORT_REPO_PATH"

echo "[INFO] Starting inference..."
CMD=("srun" "\$VENV_PY" "$TREEMORT_REPO_PATH/inference/engine.py")
CMD+=($INFER_ARGS)
"\${CMD[@]}"
EOT

fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job
sbatch --export=ALL $SBATCH_SCRIPT "$@"
