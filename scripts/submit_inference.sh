#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: HPC_TYPE argument is required (e.g., 'puhti' or 'lumi')."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: MODEL_TYPE argument is required (e.g., 'flair_unet')."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: DATA_TYPE argument is required (e.g., 'finland' or 'poland')."
    exit 1
fi

export HPC_TYPE="$1"
export MODEL_TYPE="$2"
export DATA_TYPE="$3"

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"

# Set global environment variables based on HPC type.
if [ "$HPC_TYPE" == "puhti" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_2004205/anisrahm/venv"
    export TREEMORT_DATA_PATH="/scratch/project_2008436/anisrahm/dead_trees"
elif [ "$HPC_TYPE" == "lumi" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
    export TREEMORT_DATA_PATH="/scratch/project_462001070/anisrahm/dead_trees"
else
    echo "Error: Unsupported HPC_TYPE '$HPC_TYPE'."
    exit 1
fi

# Strip variant suffixes (e.g. _tta) to get the base data type for paths and data config.
BASE_DATA_TYPE="${DATA_TYPE%%_*}"

# Set necessary environment variables for inference.
# CONFIG_PATH uses the full DATA_TYPE so variant configs (e.g. finland_tta.txt) are respected.
# DATA_CONFIG_PATH and data/output paths use BASE_DATA_TYPE (the underlying dataset).
export CONFIG_PATH="$TREEMORT_REPO_PATH/configs/inference/${DATA_TYPE}.txt"
export DATA_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/data/${BASE_DATA_TYPE}.txt"
export MODEL_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/model/${MODEL_TYPE}.txt"

export PREDICTIONS_FOLDER="Predictions_${MODEL_TYPE}"
if [[ "$@" == *"--post-process"* ]]; then
    PREDICTIONS_FOLDER="${PREDICTIONS_FOLDER}_post_process"
fi

if [ "$BASE_DATA_TYPE" == "finland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm"
    export OUTPUT_PATH="$TREEMORT_DATA_PATH/Finland/$PREDICTIONS_FOLDER"
elif [ "$BASE_DATA_TYPE" == "poland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Poland/RGBNIR/25cm"
    export OUTPUT_PATH="$TREEMORT_DATA_PATH/Poland/$PREDICTIONS_FOLDER"
else
    echo "Error: Unsupported DATA_TYPE '$DATA_TYPE' (base: '$BASE_DATA_TYPE')."
    exit 1
fi

# Parse flags from positional args 4+.
# --chunks N  → split work across N array tasks (one GPU each).
# All other flags are forwarded to run_inference.sh.
CHUNKS=1
PASS_ARGS=()
set -- "${@:4}"
while [ $# -gt 0 ]; do
    case "$1" in
        --chunks)
            CHUNKS="$2"; shift 2 ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

# For array jobs: discover images and split into CHUNKS list files.
if [ "$CHUNKS" -gt 1 ]; then
    PROJECT_NAME="project_462001070"
    LIST_DIR="/scratch/$PROJECT_NAME/anisrahm/output/lists/infer_$(date +%s)"
    mkdir -p "$LIST_DIR"

    find "$DATA_PATH" -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.jp2" \) \
        | sort > "$LIST_DIR/all.txt"
    TOTAL=$(wc -l < "$LIST_DIR/all.txt")
    echo "[INFO] Found $TOTAL images — splitting into $CHUNKS chunks (list dir: $LIST_DIR)"

    mapfile -t IMAGES < "$LIST_DIR/all.txt"
    PER_CHUNK=$(( (TOTAL + CHUNKS - 1) / CHUNKS ))
    for ((i=0; i<TOTAL; i++)); do
        chunk=$(( i / PER_CHUNK ))
        [ "$chunk" -ge "$CHUNKS" ] && chunk=$(( CHUNKS - 1 ))
        echo "${IMAGES[$i]}" >> "$LIST_DIR/$chunk.txt"
    done

    for ((c=0; c<CHUNKS; c++)); do
        COUNT=$(wc -l < "$LIST_DIR/$c.txt" 2>/dev/null || echo 0)
        echo "[INFO]   chunk $c: $COUNT images"
    done

    export LIST_DIR
    export CHUNKS
fi

bash "$TREEMORT_REPO_PATH/scripts/run_inference.sh" "${PASS_ARGS[@]}"