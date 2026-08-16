#!/bin/bash
# Submit eval-only jobs for one or more run-ids.
#
# Usage:
#   bash submit_treemort_eval.sh lumi <model_config> <data_config> <run_id> [run_id ...]
#
# Example (re-run Fine-tuned data efficiency evals):
#   bash submit_treemort_eval.sh lumi \
#       configs/model/flair_unet_transfer.txt \
#       configs/data/poland.txt \
#       dataeff_ft_100 dataeff_ft_75 dataeff_ft_50 dataeff_ft_25

if [ "$1" != "lumi" ] || [ "$#" -lt 4 ]; then
    echo "Usage: $0 lumi <model_config> <data_config> <run_id> [run_id ...]"
    exit 1
fi

shift  # remove 'lumi'
MODEL_CONFIG="$1"; shift
DATA_CONFIG="$1";  shift
RUN_IDS=("$@")

TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
GPUS_PER_NODE=8
TEACHER_CHECKPOINT="$TREEMORT_REPO_PATH/output/flair_unet/high_recall/best.weights.pth"

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

for RUN_ID in "${RUN_IDS[@]}"; do
    SBATCH_SCRIPT=$(mktemp)
    cat > "$SBATCH_SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eval-${RUN_ID}
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
export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-${TREEMORT_REPO_PATH}/output}"
export TREEMORT_TEACHER_PATH="${TEACHER_CHECKPOINT}"
export GPUS_PER_NODE=${GPUS_PER_NODE}

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] eval run_id=${RUN_ID}"
srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "${TREEMORT_REPO_PATH}/scripts/lumi_train_launcher.sh" \\
        "${TREEMORT_REPO_PATH}/${MODEL_CONFIG}" \\
        --data-config "${TREEMORT_REPO_PATH}/${DATA_CONFIG}" \\
        --run-id "${RUN_ID}" \\
        --eval-only
SBATCH

    JOB_ID=$(sbatch --export=NONE "$SBATCH_SCRIPT" | awk '{print $NF}')
    rm "$SBATCH_SCRIPT"
    echo "  eval ${RUN_ID}: ${JOB_ID}"
done
