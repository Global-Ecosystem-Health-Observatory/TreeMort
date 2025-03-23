#!/bin/bash

# Check if at least the HPC type and config file are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: ./run_treemort.sh <hpc_type> <config file> [--eval-only true|false] [--test-run true|false]"
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
    PROJECT_NAME="project_462000684"
    PARTITION_NAME="small-g"
    TEST_PARTITION_NAME="dev-g"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD="module use /appl/local/csc/modulefiles/"
    GPU_DIRECTIVE="#SBATCH --gpus-per-node=1"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="gpu"
    TEST_PARTITION_NAME="gputest"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD=""
    GPU_DIRECTIVE="#SBATCH --gres=gpu:v100:1"
fi

# Adjust partition for test or eval runs
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

cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=tree-mort
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=$TIME_LIMIT
#SBATCH --partition=$PARTITION_NAME
$GPU_DIRECTIVE

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
EOT

# Build the command string with optional flags
CMD="srun python3 -m treemort.main \"$CONFIG_FILE\""
if [ -n "$DATA_CONFIG" ]; then
    CMD="$CMD --data-config \"$DATA_CONFIG\""
fi
if [ "$EVAL_ONLY" = true ]; then
    CMD="$CMD --eval-only"
fi
echo "$CMD" >> $SBATCH_SCRIPT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit job to SLURM
sbatch $SBATCH_SCRIPT

rm $SBATCH_SCRIPT