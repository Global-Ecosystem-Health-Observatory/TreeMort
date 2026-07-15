#!/bin/bash

# Set default HPC type to "puhti"
HPC_TYPE=${HPC_TYPE:-"puhti"}

# Set HPC-specific variables
if [ "$HPC_TYPE" == "lumi" ]; then
    PROJECT_NAME="project_462001070"
    PARTITION_NAME="small"
else
    PROJECT_NAME="project_2004205"
    PARTITION_NAME="small"
fi

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

if [ "$HPC_TYPE" == "lumi" ]; then

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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid."
    exit 1
fi

echo "[INFO] Starting creator..."
srun singularity run "\${SIF}" bash -c "source $TREEMORT_VENV_PATH/bin/activate && PYTHONPATH=$TREEMORT_REPO_PATH python3 -m dataset.creator \"$DATA_CONFIG_PATH\" --num-workers \$SLURM_CPUS_PER_TASK"

EXIT_STATUS=\$?
if [ \$EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status \$EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit \$EXIT_STATUS
EOT

else
    # Puhti / default: module + venv approach
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

module load pytorch/2.5

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

if [ -z "$DATA_CONFIG_PATH" ] || [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "[ERROR] Data config file is missing or invalid."
    exit 1
fi

export PYTHONPATH="$TREEMORT_REPO_PATH:\${PYTHONPATH:-}"

echo "[INFO] Starting creator..."
srun python3 -m dataset.creator "$DATA_CONFIG_PATH" --num-workers \$SLURM_CPUS_PER_TASK

EXIT_STATUS=\$?
if [ \$EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] Job failed with exit status \$EXIT_STATUS"
else
    echo "[INFO] Job completed successfully"
fi

exit \$EXIT_STATUS
EOT

fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit SLURM Job
sbatch --export=ALL $SBATCH_SCRIPT "$@"
