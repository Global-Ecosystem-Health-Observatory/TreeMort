#!/bin/bash
# Data efficiency experiments: Fine-tuned vs Feature-level KD on Poland
# at 100%, 75%, 50%, 25% of training data with the updated teacher.
#
# Each fraction uses a unique --run-id to isolate checkpoints.
# An eval job is submitted with --dependency=afterok on each train job.
#
# Usage: bash submit_treemort_dataeff.sh lumi

if [ -z "$1" ] || [ "$1" != "lumi" ]; then
    echo "Usage: $0 lumi"
    exit 1
fi

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
export TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
export TREEMORT_DATA_PATH="/scratch/project_462001070/anisrahm/dead_trees"

PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

TEACHER_CHECKPOINT="/users/anisrahm/TreeMort/output/flair_unet/high_recall/best.weights.pth"
DATA_CONFIG="$TREEMORT_REPO_PATH/configs/data/poland.txt"

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

submit_job() {
    local LABEL="$1"        # e.g. ft_100, kd_75
    local MODEL_CONFIG="$2"
    local FRACTION="$3"
    local EVAL_ONLY="$4"    # true | false
    local TRAIN_JOB_ID="$5" # dependency job id (empty = no dependency)
    local MODULE="$6"       # treemort.main or treemort.main_kd
    local RUN_ID="dataeff_${LABEL}"

    local TIME_LIMIT="06:00:00"
    [ "$EVAL_ONLY" = "true" ] && TIME_LIMIT="01:00:00"

    local LAUNCHER_ARGS="\"$MODEL_CONFIG\" --data-config \"$DATA_CONFIG\" --run-id $RUN_ID"
    [ "$EVAL_ONLY" = "true" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --eval-only"
    [ "$EVAL_ONLY" = "false" ] && LAUNCHER_ARGS="$LAUNCHER_ARGS --train-fraction $FRACTION"

    local SBATCH_SCRIPT=$(mktemp)
    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=dataeff-${LABEL}$([ "$EVAL_ONLY" = "true" ] && echo "-eval")
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_OUTPUT_DIR/stdout/%A_%a.out
#SBATCH --error=$SCRATCH_OUTPUT_DIR/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=$TIME_LIMIT
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
# Redirect TMPDIR so HIP gfx*.ufdb.txt files go to per-job dir, not /tmp
export TMPDIR="\$MIOPEN_DIR/tmp"
mkdir -p "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_DIR/config" "\$TMPDIR"

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/anisrahm/TreeMort/output}"
export TREEMORT_TEACHER_PATH="$TEACHER_CHECKPOINT"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] dataeff ${LABEL} module=$MODULE eval=$EVAL_ONLY fraction=$FRACTION run_id=$RUN_ID"
srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash -c "
        export RANK=\\\${SLURM_PROCID}
        export LOCAL_RANK=\\\${SLURM_LOCALID}
        export WORLD_SIZE=\\\${SLURM_NTASKS}
        export ROCR_VISIBLE_DEVICES=\\\${SLURM_LOCALID}
        cd $TREEMORT_REPO_PATH
        ${TREEMORT_VENV_PATH}/bin/python3 -m $MODULE $LAUNCHER_ARGS
    "
EOT

    local DEP_FLAG=""
    [ -n "$TRAIN_JOB_ID" ] && DEP_FLAG="--dependency=afterok:${TRAIN_JOB_ID}"

    local JOB_ID
    JOB_ID=$(sbatch --export=ALL $DEP_FLAG $SBATCH_SCRIPT | awk '{print $NF}')
    rm $SBATCH_SCRIPT
    echo "$JOB_ID"
}

for FRACTION in 1.0 0.75 0.5 0.25; do
    FRAC_TAG=$(echo "$FRACTION" | sed 's/\\./_/g' | sed 's/^0_//' | sed 's/_0$/0/')
    # Make clean tag: 1.0->100, 0.75->75, 0.5->50, 0.25->25
    case "$FRACTION" in
        1.0)  FRAC_TAG="100" ;;
        0.75) FRAC_TAG="75"  ;;
        0.5)  FRAC_TAG="50"  ;;
        0.25) FRAC_TAG="25"  ;;
    esac

    FT_CONFIG="$TREEMORT_REPO_PATH/configs/model/flair_unet_transfer.txt"
    KD_CONFIG="$TREEMORT_REPO_PATH/configs/model/flair_unet_kd_feature.txt"

    echo "=== Fine-tuned $FRAC_TAG% ==="
    TRAIN_ID=$(submit_job "ft_${FRAC_TAG}" "$FT_CONFIG" "$FRACTION" "false" "" "treemort.main")
    echo "  train job: $TRAIN_ID"
    EVAL_ID=$(submit_job "ft_${FRAC_TAG}" "$FT_CONFIG" "$FRACTION" "true" "$TRAIN_ID" "treemort.main")
    echo "  eval  job: $EVAL_ID (depends on $TRAIN_ID)"

    echo "=== Feature KD $FRAC_TAG% ==="
    TRAIN_ID=$(submit_job "kd_${FRAC_TAG}" "$KD_CONFIG" "$FRACTION" "false" "" "treemort.main_kd")
    echo "  train job: $TRAIN_ID"
    EVAL_ID=$(submit_job "kd_${FRAC_TAG}" "$KD_CONFIG" "$FRACTION" "true" "$TRAIN_ID" "treemort.main_kd")
    echo "  eval  job: $EVAL_ID (depends on $TRAIN_ID)"

done
