#!/bin/bash
# Zero-shot evaluation: Finnish teacher applied directly to a target country
# with no adaptation (no fine-tuning, no KD, no DANN).
#
# Usage: bash submit_treemort_zeroshot.sh lumi <country>
#   country: poland | germany | estonia
#
# Example (submit all three):
#   for c in poland germany estonia; do
#       bash submit_treemort_zeroshot.sh lumi $c
#   done

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <hpc_type> <country>"
    echo "  hpc_type: lumi"
    echo "  country:  poland | germany | estonia"
    exit 1
fi

HPC_TYPE="$1"
DATA_TYPE="$2"

if [ "$HPC_TYPE" != "lumi" ]; then
    echo "Error: Only 'lumi' is supported."
    exit 1
fi

case "$DATA_TYPE" in
    poland|germany|estonia) ;;
    *) echo "Error: Unsupported country '$DATA_TYPE'."; exit 1 ;;
esac

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
export TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
export TREEMORT_DATA_PATH="/scratch/project_462001070/anisrahm/dead_trees"

PROJECT_NAME="project_462001070"
PARTITION_NAME="small-g"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

MODEL_CONFIG_PATH="$TREEMORT_REPO_PATH/configs/model/flair_unet_highrecall.txt"
TARGET_DATA_CONFIG="$TREEMORT_REPO_PATH/configs/data/${DATA_TYPE}.txt"

# Finnish teacher checkpoint — trained with flair_unet_highrecall on finland.txt
# run_dir = TREEMORT_OUTPUT_DIR / flair_unet / high_recall  (run-id = high_recall was set
# at training time, before it was removed from the config in commit 134d9580)
FINNISH_CHECKPOINT="/users/anisrahm/TreeMort/output/flair_unet/high_recall/best.weights.pth"

LAUNCHER_ARGS="\"$MODEL_CONFIG_PATH\" --data-config \"$TARGET_DATA_CONFIG\" --resume-from \"$FINNISH_CHECKPOINT\" --eval-only"

SBATCH_SCRIPT=$(mktemp)

cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=treemort-zeroshot-${DATA_TYPE}
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_OUTPUT_DIR/stdout/%A_%a.out
#SBATCH --error=$SCRATCH_OUTPUT_DIR/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

if [ ! -d "$TREEMORT_REPO_PATH" ]; then
    echo "[ERROR] Repository path not found: $TREEMORT_REPO_PATH"
    exit 1
fi

if [ ! -d "$TREEMORT_VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found: $TREEMORT_VENV_PATH"
    exit 1
fi

if [ ! -f "$FINNISH_CHECKPOINT" ]; then
    echo "[ERROR] Finnish teacher checkpoint not found: $FINNISH_CHECKPOINT"
    exit 1
fi

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
echo "[INFO] MIOpen dirs: \$MIOPEN_DIR"

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/anisrahm/TreeMort/output}"
mkdir -p "\$TREEMORT_OUTPUT_DIR"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Zero-shot eval: Finnish teacher on ${DATA_TYPE}"
echo "[INFO] Checkpoint: $FINNISH_CHECKPOINT"
echo "[INFO] Nodes=\$SLURM_JOB_NUM_NODES GPUs/node=\$GPUS_PER_NODE MASTER=\$MASTER_ADDR:\$MASTER_PORT"

srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_train_launcher.sh" $LAUNCHER_ARGS
EOT

echo "Generated SBATCH script:"
cat $SBATCH_SCRIPT

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

sbatch --export=ALL $SBATCH_SCRIPT

rm $SBATCH_SCRIPT
