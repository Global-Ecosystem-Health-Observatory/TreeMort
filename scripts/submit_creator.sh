#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: HPC_TYPE argument is required (e.g., 'puhti' or 'lumi')."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: DATA_TYPE argument is required (e.g., 'finland' or 'poland')."
    exit 1
fi

export HPC_TYPE="$1"
export DATA_TYPE="$2"

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
export DATA_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/data/${DATA_TYPE}.txt"

# Call the inference script with the filtered flag.
bash $TREEMORT_REPO_PATH/scripts/run_creator.sh