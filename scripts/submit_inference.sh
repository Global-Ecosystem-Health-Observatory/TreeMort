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

export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

# Set global environment variables based on HPC type.
if [ "$HPC_TYPE" == "puhti" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
    export TREEMORT_DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees"
elif [ "$HPC_TYPE" == "lumi" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_462000684/rahmanan/venv"
    export TREEMORT_DATA_PATH="/scratch/project_462000684/rahmanan/dead_trees"
else
    echo "Error: Unsupported HPC_TYPE '$HPC_TYPE'."
    exit 1
fi

# Set necessary environment variables for inference.
export CONFIG_PATH="$TREEMORT_REPO_PATH/configs/inference/${DATA_TYPE}.txt"
export DATA_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/data/${DATA_TYPE}.txt"
export MODEL_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/model/${MODEL_TYPE}.txt"

export PREDICTIONS_FOLDER="Predictions_${MODEL_TYPE}"
if [[ "$@" == *"--post-process"* ]]; then
    PREDICTIONS_FOLDER="${PREDICTIONS_FOLDER}_post_process"
fi

if [ "$DATA_TYPE" == "finland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm"
    export OUTPUT_PATH="$TREEMORT_DATA_PATH/Finland/$PREDICTIONS_FOLDER"
elif [ "$DATA_TYPE" == "poland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Poland/RGBNIR/25cm"
    export OUTPUT_PATH="$TREEMORT_DATA_PATH/Poland/$PREDICTIONS_FOLDER"
else
    echo "Error: Unsupported DATA_TYPE '$DATA_TYPE'."
    exit 1
fi

# Filter and forward only the --post-process flag to the inference script.
POST_PROCESS_FLAG=""
if [[ "$@" == *"--post-process"* ]]; then
    POST_PROCESS_FLAG="--post-process"
fi

# Call the inference script with the filtered flag.
bash $TREEMORT_REPO_PATH/scripts/run_inference.sh "$POST_PROCESS_FLAG"