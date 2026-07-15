#!/bin/bash

# Check if at least the HPC type and config file are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: ./run_treemort.sh <hpc_type> <config file> [--eval-only true|false] [--data-config <file>] [--test-run true|false]"
    exit 1
fi

HPC_TYPE=$1
CONFIG_FILE=$2
EVAL_ONLY=false
DATA_CONFIG=""
TEST_RUN=false

shift 2

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --eval-only) EVAL_ONLY="$2"; shift ;;
        --data-config) DATA_CONFIG="$2"; shift ;;
        --test-run) TEST_RUN="$2"; shift ;;
    esac
    shift
done

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small-g"
    TEST_PARTITION_NAME="dev-g"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
    TEST_PARTITION_NAME="gputest"
fi

# Adjust partition and time limit
if [ "$TEST_RUN" = true ]; then
    TIME_LIMIT="00:15:00"
    PARTITION_NAME=$TEST_PARTITION_NAME
elif [ "$EVAL_ONLY" = true ]; then
    TIME_LIMIT="01:00:00"
else
    TIME_LIMIT="36:00:00"
fi

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

if [ "$HPC_TYPE" == "lumi" ]; then

    # Build launcher args at generation time (baked as literals in the SBATCH script)
    LAUNCHER_ARGS="\"$CONFIG_FILE\""
    [ -n "$DATA_CONFIG" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --data-config \"$DATA_CONFIG\""
    [ "$EVAL_ONLY" = true ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --eval-only"

    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=tree-mort
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=$TIME_LIMIT
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
export GPUS_PER_NODE=$GPUS_PER_NODE

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

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-./output}"
mkdir -p "\$TREEMORT_OUTPUT_DIR"

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Nodes=\$SLURM_JOB_NUM_NODES GPUs/node=\$GPUS_PER_NODE MASTER=\$MASTER_ADDR:\$MASTER_PORT"

srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_train_launcher.sh" $LAUNCHER_ARGS
EOT

else
    # Puhti / default: module + venv approach
    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=tree-mort
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

EOT

    CMD="srun python3 -m treemort.main \"$CONFIG_FILE\""
    [ -n "$DATA_CONFIG" ] && CMD="$CMD --data-config \"$DATA_CONFIG\""
    [ "$EVAL_ONLY" = true ] && CMD="$CMD --eval-only"
    echo "$CMD" >> $SBATCH_SCRIPT
fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit job to SLURM
sbatch --export=ALL $SBATCH_SCRIPT

rm $SBATCH_SCRIPT
