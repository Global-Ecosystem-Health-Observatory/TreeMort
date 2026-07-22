#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: HPC_TYPE argument is required (e.g., 'puhti' or 'lumi')."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: MODEL_TYPE argument is required (e.g., 'flair_unet_kd_basic')."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: DATA_TYPE argument is required (e.g., 'finland' or 'poland')."
    exit 1
fi

export HPC_TYPE="$1"
export MODEL_TYPE="$2"
export DATA_TYPE="$3"

export TREEMORT_REPO_PATH="/users/aurahman/TreeMort"

# Set global environment variables based on HPC type.
if [ "$HPC_TYPE" == "puhti" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_2004205/aurahman/venv"
    export TREEMORT_DATA_PATH="/scratch/project_2008436/aurahman/dead_trees"
elif [ "$HPC_TYPE" == "lumi" ]; then
    export TREEMORT_VENV_PATH="/projappl/project_462001070/aurahman/venv"
    export TREEMORT_DATA_PATH="/scratch/project_462001070/aurahman/dead_trees"
else
    echo "Error: Unsupported HPC_TYPE '$HPC_TYPE'."
    exit 1
fi

# Set necessary environment variables for inference.
export DATA_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/data/${DATA_TYPE}.txt"
export MODEL_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/model/${MODEL_TYPE}.txt"

if [ "$DATA_TYPE" == "finland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Finland/RGBNIR/25cm"
elif [ "$DATA_TYPE" == "poland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Poland/RGBNIR/25cm"
elif [ "$DATA_TYPE" == "estonia" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Estonia/RGBNIR/25cm"
elif [ "$DATA_TYPE" == "germany" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Germany/RGBNIR/10cm"
elif [ "$DATA_TYPE" == "switzerland" ]; then
    export DATA_PATH="$TREEMORT_DATA_PATH/Switzerland/RGB/10cm"
else
    echo "Error: Unsupported DATA_TYPE '$DATA_TYPE'."
    exit 1
fi

# TREEMORT_TEACHER_PATH must be set by the caller before invoking this script.
if [ -z "$TREEMORT_TEACHER_PATH" ]; then
    echo "Error: TREEMORT_TEACHER_PATH is not set. Export it before calling this script."
    exit 1
fi

# Filter and forward only the --eval-only flag.
EVAL_ONLY_FLAG="false"
if [[ "$@" == *"--eval-only"* ]]; then
    EVAL_ONLY_FLAG="true"
fi

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small-g"
    TEST_PARTITION_NAME="dev-g"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/aurahman/output"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
    TEST_PARTITION_NAME="gputest"
fi

if [ "$EVAL_ONLY_FLAG" = "true" ]; then
    TIME_LIMIT="01:00:00"
else
    TIME_LIMIT="06:00:00"
fi

SBATCH_SCRIPT=$(mktemp)

if [ "$HPC_TYPE" == "lumi" ]; then
    LAUNCHER_ARGS="\"$MODEL_CONFIG_PATH\" --data-config \"$DATA_CONFIG_PATH\""
    [ "$EVAL_ONLY_FLAG" = "true" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --eval-only"

    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-kd
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

export MIOPEN_USER_DB_PATH="$TREEMORT_REPO_PATH/.cache/miopen"
mkdir -p "\$MIOPEN_USER_DB_PATH"
echo "[INFO] MIOpen cache: \$MIOPEN_USER_DB_PATH"

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/aurahman/TreeMort/output}"
mkdir -p "\$TREEMORT_OUTPUT_DIR"
export TREEMORT_TEACHER_PATH="${TREEMORT_TEACHER_PATH}"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Nodes=\$SLURM_JOB_NUM_NODES GPUs/node=\$GPUS_PER_NODE MASTER=\$MASTER_ADDR:\$MASTER_PORT"

srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_kd_launcher.sh" $LAUNCHER_ARGS
EOT

else
    # Puhti / default
    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-kd
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=$TIME_LIMIT
#SBATCH --partition=$PARTITION_NAME
#SBATCH --gres=gpu:v100:1

module load pytorch/2.5

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

if [ -n "$TREEMORT_REPO_PATH" ] && [ -d "$TREEMORT_REPO_PATH" ]; then
    echo "[INFO] Changing directory to $TREEMORT_REPO_PATH"
    cd "$TREEMORT_REPO_PATH" || exit 1
else
    echo "[ERROR] Repository path not found or TREEMORT_REPO_PATH not set."
    exit 1
fi

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-./output}"
mkdir -p "\$TREEMORT_OUTPUT_DIR"

export TREEMORT_TEACHER_PATH="${TREEMORT_TEACHER_PATH}"

EOT

    CMD="srun python3 -m treemort.main_kd \"$MODEL_CONFIG_PATH\" --data-config \"$DATA_CONFIG_PATH\""
    [ "$EVAL_ONLY_FLAG" = "true" ] && CMD="$CMD --eval-only"
    echo "$CMD" >> $SBATCH_SCRIPT
fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

if [ "$HPC_TYPE" == "lumi" ]; then
    mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"
fi

sbatch --export=ALL $SBATCH_SCRIPT

rm $SBATCH_SCRIPT
