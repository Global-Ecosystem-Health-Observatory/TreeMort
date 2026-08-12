#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./submit_treemort_dann.sh <hpc_type> <data_type> [--eval-only]"
    echo "  hpc_type:  lumi | puhti"
    echo "  data_type: poland | germany | estonia"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: DATA_TYPE argument is required (e.g., 'poland')."
    exit 1
fi

export HPC_TYPE="$1"
export DATA_TYPE="$2"

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"

if [ "$HPC_TYPE" == "lumi" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
    export TREEMORT_DATA_PATH="/scratch/project_462001070/anisrahm/dead_trees"
elif [ "$HPC_TYPE" == "puhti" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_2004205/anisrahm/venv"
    export TREEMORT_DATA_PATH="/scratch/project_2008436/anisrahm/dead_trees"
else
    echo "Error: Unsupported HPC_TYPE '$HPC_TYPE'."
    exit 1
fi

MODEL_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/model/flair_unet_dann.txt"
TARGET_DATA_CONFIG="$TREEMORT_REPO_PATH/configs/data/${DATA_TYPE}.txt"
SOURCE_DATA_CONFIG="$TREEMORT_REPO_PATH/configs/data/finland.txt"

if [ "$DATA_TYPE" == "poland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Poland/RGBNIR/25cm"
elif [ "$DATA_TYPE" == "estonia" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Estonia/RGBNIR/25cm"
elif [ "$DATA_TYPE" == "germany" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Germany/RGBNIR/10cm"
else
    echo "Error: Unsupported DATA_TYPE '$DATA_TYPE'."
    exit 1
fi

EVAL_ONLY_FLAG="false"
if [[ "$@" == *"--eval-only"* ]]; then
    EVAL_ONLY_FLAG="true"
fi

if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small-g"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
fi

if [ "$EVAL_ONLY_FLAG" = "true" ]; then
    TIME_LIMIT="01:00:00"
else
    TIME_LIMIT="06:00:00"
fi

SBATCH_SCRIPT=$(mktemp)

if [ "$HPC_TYPE" == "lumi" ]; then
    LAUNCHER_ARGS="\"$MODEL_CONFIG_PATH\" --data-config \"$TARGET_DATA_CONFIG\""
    [ "$EVAL_ONLY_FLAG" != "true" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --source-data-config \"$SOURCE_DATA_CONFIG\""
    [ "$EVAL_ONLY_FLAG" = "true" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --eval-only"

    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-dann
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_OUTPUT_DIR/stdout/%A_%a.out
#SBATCH --error=$SCRATCH_OUTPUT_DIR/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=$TIME_LIMIT
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

if [ ! -d "$TREEMORT_REPO_PATH" ]; then
    echo "[ERROR] Repository path not found: $TREEMORT_REPO_PATH"
    exit 1
fi

if [ ! -d "$TREEMORT_VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found: $TREEMORT_VENV_PATH"
    exit 1
fi

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
echo "[INFO] MIOpen dirs: \$MIOPEN_DIR"

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/anisrahm/TreeMort/output}"
mkdir -p "\$TREEMORT_OUTPUT_DIR"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Nodes=\$SLURM_JOB_NUM_NODES GPUs/node=\$GPUS_PER_NODE MASTER=\$MASTER_ADDR:\$MASTER_PORT"

srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_dann_launcher.sh" $LAUNCHER_ARGS
EOT

fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

if [ "$HPC_TYPE" == "lumi" ]; then
    mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"
fi

sbatch --export=ALL $SBATCH_SCRIPT

rm $SBATCH_SCRIPT
