#!/bin/bash

# Check if at least the config file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: ./run_treemort.sh <config file> [--eval-only true|false] [--test-run true|false]"
    exit 1
fi

CONFIG_FILE=$1
EVAL_ONLY=false

# Check for additional arguments
while [[ "$#" -gt 1 ]]; do
    case $2 in
        --eval-only) EVAL_ONLY="$3"; shift ;;
	--test-run) TEST_RUN="$3"; shift ;;
    esac
    shift
done

# Create a temporary SBATCH script in the local directory
SBATCH_SCRIPT=$(mktemp)

cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=tree-mort
#SBATCH --account=project_2004205
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
EOT

if [ "$TEST_RUN" = true ]; then
    echo "#SBATCH --time=00:08:00" >> $SBATCH_SCRIPT
    echo "#SBATCH --partition=gputest" >> $SBATCH_SCRIPT
elif [ "$EVAL_ONLY" = true ]; then
    echo "#SBATCH --time=01:00:00" >> $SBATCH_SCRIPT
    echo "#SBATCH --partition=gpu" >> $SBATCH_SCRIPT
else
    echo "#SBATCH --time=16:00:00" >> $SBATCH_SCRIPT
    echo "#SBATCH --partition=gpu" >> $SBATCH_SCRIPT
fi

# Add the rest of the script
cat <<EOT >> $SBATCH_SCRIPT

MODULE_NAME="pytorch/2.3"

echo "Loading module: \$MODULE_NAME"
module load \$MODULE_NAME

TREEMORT_VENV_PATH="\${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"

if [ -d "\$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at \$TREEMORT_VENV_PATH"
    source "\$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at \$TREEMORT_VENV_PATH"
    exit 1
fi

EOT

if [ "$EVAL_ONLY" = true ]; then
    echo "srun python3 -m treemort.main \"$CONFIG_FILE\" --eval-only" >> $SBATCH_SCRIPT
else
    echo "srun python3 -m treemort.main \"$CONFIG_FILE\"" >> $SBATCH_SCRIPT
fi

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

sbatch $SBATCH_SCRIPT

rm $SBATCH_SCRIPT

