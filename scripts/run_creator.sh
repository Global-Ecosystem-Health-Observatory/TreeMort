#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462000684"
    PARTITION_NAME="small"
    MODULE_NAME="pytorch/2.5"
    VENV_PATH="/projappl/project_462000684/rahmanan/venv"
    TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
    MODULE_USE_CMD="module use /appl/local/csc/modulefiles/"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="small"
    MODULE_NAME="pytorch/2.5"
    VENV_PATH="/projappl/project_2004205/rahmanan/venv"
    TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
    MODULE_USE_CMD=""
fi

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

# SLURM Job Configuration
cat <<'EOT' > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-creator
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=output/stdout/%A_%a.out
#SBATCH --error=output/stderr/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --partition=$PARTITION_NAME
#SBATCH --mem-per-cpu=6000

$MODULE_USE_CMD
echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    exit 1
fi

CREATOR_PATH="${TREEMORT_REPO_PATH}/dataset/creator.py"

if [ ! -f "$CREATOR_PATH" ]; then
    echo "[ERROR] Creator source file not found at $CREATOR_PATH"
    exit 1
fi

if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid."
    exit 1
fi

echo "[INFO] Starting creator..."
srun python3 "$CREATOR_PATH" "$DATA_CONFIG_PATH" --num-workers "$SLURM_CPUS_PER_TASK"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job
bash $SBATCH_SCRIPT "$@"