#!/bin/bash
# Submit feature extraction jobs for representational analysis (Figures 3-6).
#
# Extracts encoder layer features for 3 models x 2 datasets = 6 jobs.
# Checkpoints used are the main Table 2 runs (no --run-id).
#
# Usage: bash submit_feature_extraction.sh lumi

if [ -z "$1" ] || [ "$1" != "lumi" ]; then
    echo "Usage: $0 lumi"
    exit 1
fi

TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
SCRATCH_FEAT_DIR="/scratch/$PROJECT_NAME/anisrahm/output"

TEACHER_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/high_recall/best.weights.pth"
FT_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.pth"
KD_FEAT_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.feature.pth"

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

submit_extract() {
    local LABEL="$1"        # e.g. poland_baseline
    local MODEL_CONFIG="$2"
    local DATA_CONFIG="$3"
    local CHECKPOINT="$4"

    local SBATCH_SCRIPT
    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=feat-${LABEL}
#SBATCH --account=${PROJECT_NAME}
#SBATCH --output=${SCRATCH_OUTPUT_DIR}/stdout/%A.out
#SBATCH --error=${SCRATCH_OUTPUT_DIR}/stderr/%A.err
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --partition=${PARTITION_NAME}

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
export TMPDIR="\$MIOPEN_DIR/tmp"
mkdir -p "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_DIR/config" "\$TMPDIR"

echo "[INFO] Extracting features: label=${LABEL}"
srun singularity run "\${SIF}" bash -c "
    cd ${TREEMORT_REPO_PATH}
    ${TREEMORT_VENV_PATH}/bin/python3 scripts/extract_features.py \
        --config      ${MODEL_CONFIG} \
        --data-config ${DATA_CONFIG} \
        --checkpoint  ${CHECKPOINT} \
        --output-dir  ${SCRATCH_FEAT_DIR}/features_${LABEL} \
        --max-patches 1000
"
SBATCH

    local JOB_ID
    JOB_ID=$(sbatch --export=ALL "$SBATCH_SCRIPT" | awk '{print $NF}')
    rm "$SBATCH_SCRIPT"
    echo "  features_${LABEL}: job ${JOB_ID}"
    echo "$JOB_ID"
}

echo "=== Submitting feature extraction jobs ==="

# --- Poland ---
submit_extract "poland_baseline" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_highrecall.txt" \
    "$TREEMORT_REPO_PATH/configs/data/poland.txt" \
    "$TEACHER_CKPT"

submit_extract "poland_transfer" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_transfer.txt" \
    "$TREEMORT_REPO_PATH/configs/data/poland.txt" \
    "$FT_CKPT"

submit_extract "poland_feature" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_kd_feature.txt" \
    "$TREEMORT_REPO_PATH/configs/data/poland.txt" \
    "$KD_FEAT_CKPT"

# --- Finland ---
submit_extract "finland_baseline" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_highrecall.txt" \
    "$TREEMORT_REPO_PATH/configs/data/finland.txt" \
    "$TEACHER_CKPT"

submit_extract "finland_transfer" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_transfer.txt" \
    "$TREEMORT_REPO_PATH/configs/data/finland.txt" \
    "$FT_CKPT"

submit_extract "finland_feature" \
    "$TREEMORT_REPO_PATH/configs/model/flair_unet_kd_feature.txt" \
    "$TREEMORT_REPO_PATH/configs/data/finland.txt" \
    "$KD_FEAT_CKPT"

echo ""
echo "Once all jobs complete, generate figures with:"
echo "  python3 misc/representational_analysis.py \\"
echo "    --feature-root ${SCRATCH_FEAT_DIR} \\"
echo "    --report-images report/images"
