#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462000684"
    PARTITION_NAME="small"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD="module use /appl/local/csc/modulefiles/"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="small"
    MODULE_NAME="pytorch/2.5"
    MODULE_USE_CMD=""
fi

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

# SLURM Job Configuration
cat <<EOT > $SBATCH_SCRIPT
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

# Set SLURM_CPUS_PER_TASK
# export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-6}

# If on Lumi, set the module path
$MODULE_USE_CMD
echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

# Reset PATH to minimal system directories
export PATH="/usr/bin:/bin"

# Activate virtual environment
if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
    # Prepend virtual environment's bin directory to PATH
    export PATH="$TREEMORT_VENV_PATH/bin:$PATH"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

# Verify PATH (for debugging)
echo "Current PATH: \$PATH"

# Check if DATA_CONFIG_PATH is set and exists
if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid."
    exit 1
fi

echo "[INFO] Starting creator..."
if [ -z "$TREEMORT_REPO_PATH" ]; then
    echo "[ERROR] TREEMORT_REPO_PATH is not set."
    exit 1
fi

if [ -z "$SLURM_TRES_PER_TASK" ]; then
    echo "[WARNING] SLURM_TRES_PER_TASK is not set. Defaulting to 1."
    SLURM_TRES_PER_TASK=1
fi

# Run the Python script using the virtual environment's python3
srun python3 "$TREEMORT_REPO_PATH/dataset/creator.py" "$DATA_CONFIG_PATH" --num-workers "$SLURM_TRES_PER_TASK"

EXIT_STATUS=$?
if [ "${EXIT_STATUS:-0}" -ne 0 ]; then
    echo "[ERROR] Job failed with exit status $EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit $EXIT_STATUS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job with minimal environment
sbatch $SBATCH_SCRIPT