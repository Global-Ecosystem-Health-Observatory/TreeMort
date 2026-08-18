#!/bin/bash
# Submit feature extraction jobs for representational analysis (Figures 3-6).
#
# Extracts encoder layer features for 3 models x 2 datasets = 6 jobs.
# Features saved to $TREEMORT_REPO_PATH/output/features_<label>/.
#
# Usage: bash submit_feature_extraction.sh lumi

if [ "$1" != "lumi" ] || [ "$#" -lt 1 ]; then
    echo "Usage: $0 lumi"
    exit 1
fi

TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
GPUS_PER_NODE=8

TEACHER_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/high_recall/best.weights.pth"
FT_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.pth"
KD_FEAT_CKPT="$TREEMORT_REPO_PATH/output/flair_unet/Poland_RGBNIR_25cm/best.weights.feature.pth"

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

submit_extract() {
    local LABEL="$1"        # e.g. poland_baseline
    local MODEL_CONFIG="$2"
    local DATA_CONFIG="$3"
    local CHECKPOINT="$4"
    local OUT_DIR="$TREEMORT_REPO_PATH/output/features_${LABEL}"

    local SBATCH_SCRIPT
    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=feat-${LABEL}
#SBATCH --account=${PROJECT_NAME}
#SBATCH --output=${SCRATCH_OUTPUT_DIR}/stdout/%A_%a.out
#SBATCH --error=${SCRATCH_OUTPUT_DIR}/stderr/%A_%a.err
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --partition=${PARTITION_NAME}

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
export TMPDIR="\$MIOPEN_DIR/tmp"
mkdir -p "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_DIR/config" "\$TMPDIR"

export TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH}"
export TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH}"
export TREEMORT_OUTPUT_DIR="${TREEMORT_REPO_PATH}/output"
export TREEMORT_TEACHER_PATH="${TEACHER_CKPT}"
export TREEMORT_DATA_PATH="/scratch/${PROJECT_NAME}/anisrahm/dead_trees"
export GPUS_PER_NODE=${GPUS_PER_NODE}

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] feat-${LABEL}: checkpoint=${CHECKPOINT}"
srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "${TREEMORT_REPO_PATH}/scripts/lumi_feature_launcher.sh" \\
        --config      "${MODEL_CONFIG}" \\
        --data-config "${DATA_CONFIG}" \\
        --checkpoint  "${CHECKPOINT}" \\
        --output-dir  "${OUT_DIR}" \\
        --max-patches 1000
SBATCH

    local JOB_ID
    JOB_ID=$(sbatch --export=ALL "$SBATCH_SCRIPT" | awk '{print $NF}')
    rm "$SBATCH_SCRIPT"
    echo "  features_${LABEL}: job ${JOB_ID}"
}

echo "=== Submitting feature extraction jobs ==="

# Poland
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

# Finland
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
echo "Once complete, generate figures:"
echo "  python3 misc/representational_analysis.py \\"
echo "    --feature-root ${TREEMORT_REPO_PATH}/output \\"
echo "    --report-images report/images"
