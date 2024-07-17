#!/bin/bash

# Check if at least the config file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: ./run_treemort.sh <config file> [--eval-only true|false]"
    exit 1
fi

CONFIG_FILE=$1
EVAL_ONLY=false

# Check for additional arguments
while [[ "$#" -gt 1 ]]; do
    case $2 in
        --eval-only) EVAL_ONLY="$3"; shift ;;
    esac
    shift
done

# Create a temporary SBATCH script in the local directory
SBATCH_SCRIPT=$(mktemp)

# Common SBATCH script part
cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=tree-mort
#SBATCH --account=project_2008436
#SBATCH --output=output/stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:v100:1
EOT

# Conditional SBATCH settings
if [ "$EVAL_ONLY" = true ]; then
    echo "#SBATCH --time=00:14:59" >> $SBATCH_SCRIPT
    echo "#SBATCH --partition=gputest" >> $SBATCH_SCRIPT
else
    echo "#SBATCH --time=36:00:00" >> $SBATCH_SCRIPT
    echo "#SBATCH --partition=gpu" >> $SBATCH_SCRIPT
fi

# Add the rest of the script
cat <<EOT >> $SBATCH_SCRIPT

module load tensorflow
export PYTHONPATH=\$PWD:\$PYTHONPATH
export SM_FRAMEWORK="tf.keras"

EOT

if [ "$EVAL_ONLY" = true ]; then
    echo "srun python3 -m treemort.main \"$CONFIG_FILE\" --eval-only" >> $SBATCH_SCRIPT
else
    echo "srun python3 -m treemort.main \"$CONFIG_FILE\"" >> $SBATCH_SCRIPT
fi

# Print the contents of the SBATCH script
echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

# Submit the job
sbatch $SBATCH_SCRIPT

# Remove the temporary SBATCH script
rm $SBATCH_SCRIPT

