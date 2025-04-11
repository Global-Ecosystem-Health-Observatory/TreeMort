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

# If on Lumi, set the module path
$MODULE_USE_CMD
echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

EOT

if [ "$HPC_TYPE" == "lumi" ]; then
    cat <<EOT >> $SBATCH_SCRIPT
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

# Load the module environment for Lumi
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

# Check if SINGULARITY_CONTAINER is set by the module
if [ -z "\$SINGULARITY_CONTAINER" ]; then
    echo "[ERROR] SINGULARITY_CONTAINER environment variable is not set."
    exit 1
fi

# Determine the Python version inside the container
CONTAINER_PYTHON_VERSION=\$(singularity exec \$SINGULARITY_CONTAINER python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

# Set the site-packages path based on the container's Python version
SITE_PACKAGES="$TREEMORT_VENV_PATH/lib/\$CONTAINER_PYTHON_VERSION/site-packages"

# Verify that the site-packages directory exists
if [ ! -d "\$SITE_PACKAGES" ]; then
    echo "[ERROR] Virtual environment site-packages directory not found: \$SITE_PACKAGES"
    exit 1
fi

# Execute the Python script inside the container with the correct PYTHONPATH
srun singularity exec -e PYTHONPATH="\$SITE_PACKAGES" \$SINGULARITY_CONTAINER python3 "$TREEMORT_REPO_PATH/dataset/creator.py" "$DATA_CONFIG_PATH" --num-workers 6

# Check the exit status of the job
EXIT_STATUS=\$?
if [ "\${EXIT_STATUS:-0}" -ne 0 ]; then
    echo "[ERROR] Job failed with exit status \$EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit \$EXIT_STATUS
EOT
else
    # Original method for Puhti
    cat <<EOT >> $SBATCH_SCRIPT
# Reset PATH to minimal system directories
export PATH="/usr/bin:/bin"

# Activate virtual environment
if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
    # Prepend virtual environment's bin directory to PATH
    export PATH="$TREEMORT_VENV_PATH/bin:\$PATH"
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
srun python3 "$TREEMORT_REPO_PATH/dataset/creator.py" "$DATA_CONFIG_PATH" --num-workers 6

EXIT_STATUS=\$?
if [ "\${EXIT_STATUS:-0}" -ne 0 ]; then
    echo "[ERROR] Job failed with exit status \$EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit \$EXIT_STATUS
EOT
fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job with minimal environment
sbatch $SBATCH_SCRIPT