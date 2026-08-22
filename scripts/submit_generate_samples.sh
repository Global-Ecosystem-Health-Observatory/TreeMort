#!/bin/bash
# Generate Figure 2 sample prediction comparison images on LUMI.
#
# Runs misc/generate_samples.py with all 3 model checkpoints on the
# Polish test set and writes report/images/s1.png through s4.png.
#
# Usage: bash scripts/submit_generate_samples.sh lumi
#
# After the job finishes, sync the images locally with:
#   rsync -av lumi.csc.fi:/users/anisrahm/TreeMort/report/images/s{1,2,3,4}.png \
#         report/images/

if [ -z "$1" ] || [ "$1" != "lumi" ]; then
    echo "Usage: $0 lumi"
    exit 1
fi

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
export TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
export TREEMORT_OUTPUT_DIR="/users/anisrahm/TreeMort/output"
export TREEMORT_DATA_PATH="/scratch/project_462001070/anisrahm/dead_trees"

PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_LOG_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"

BASELINE_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/high_recall/best.weights.pth"
FT_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.pth"
FEATURE_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.feature.pth"

SBATCH_SCRIPT=$(mktemp)

cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-samples
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_LOG_DIR/stdout/%j.out
#SBATCH --error=$SCRATCH_LOG_DIR/stderr/%j.err
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

for CKPT in "$BASELINE_CKPT" "$FT_CKPT" "$FEATURE_CKPT"; do
    if [ ! -f "\$CKPT" ]; then
        echo "[ERROR] Checkpoint not found: \$CKPT"
        exit 1
    fi
done

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"

export TREEMORT_OUTPUT_DIR="$TREEMORT_OUTPUT_DIR"

echo "[INFO] Generating sample figures..."

singularity exec "\${SIF}" bash -c "
  source $TREEMORT_VENV_PATH/bin/activate && \\
  cd $TREEMORT_REPO_PATH && \\
  python3 misc/generate_samples.py \\
    --config        configs/model/flair_unet_highrecall.txt \\
    --data-config   configs/data/poland.txt \\
    --baseline-ckpt $BASELINE_CKPT \\
    --ft-ckpt       $FT_CKPT \\
    --feature-ckpt  $FEATURE_CKPT \\
    --output-dir    $TREEMORT_REPO_PATH/report/images \\
    --n-samples     4 \\
    --n-candidates  100 \\
    --min-trees     8 \\
    --seed          42
"

echo "[INFO] Done. Output: $TREEMORT_REPO_PATH/report/images/s{1..4}.png"
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

mkdir -p "$SCRATCH_LOG_DIR/stdout" "$SCRATCH_LOG_DIR/stderr"

sbatch --export=ALL $SBATCH_SCRIPT

rm $SBATCH_SCRIPT
