#!/bin/bash
# Data efficiency experiments: Fine-tuned vs Feature-level KD on Poland
# at 100%, 75%, 50%, 25% of training data with the updated teacher.
#
# Usage: bash submit_treemort_dataeff.sh lumi
#
# Submits 8 jobs: 4 fractions × 2 methods (fine-tuned, feature KD)

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

for FRACTION in 1.0 0.75 0.5 0.25; do

    FRAC_TAG=$(echo "$FRACTION" | sed 's/\./_/g')  # e.g. 0_75

    # --- Fine-tuned ---
    MODEL_CONFIG="$TREEMORT_REPO_PATH/configs/model/flair_unet_transfer.txt"
    LAUNCHER_ARGS="\"$MODEL_CONFIG\" --data-config \"$DATA_CONFIG\" --train-fraction $FRACTION"

    SBATCH_SCRIPT=$(mktemp)
    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=dataeff-ft-${FRAC_TAG}
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_OUTPUT_DIR/stdout/%A_%a.out
#SBATCH --error=$SCRATCH_OUTPUT_DIR/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=06:00:00
#SBATCH --partition=$PARTITION_NAME

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v '/appl/local/csc/soft/ai' | tr '\n' ':'); PATH="\${PATH%:}"; export PATH

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_DIR/cache"
export MIOPEN_USER_DB="\$MIOPEN_DIR/config"
# Redirect TMPDIR so HIP runtime writes gfx*.ufdb.txt to a per-job dir,
# not /tmp — avoids permission collisions when multiple jobs share a node.
export TMPDIR="\$MIOPEN_DIR/tmp"
mkdir -p "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_DIR/config" "\$TMPDIR"

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/anisrahm/TreeMort/output}"
export TREEMORT_TEACHER_PATH="$TEACHER_CHECKPOINT"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Data efficiency: Fine-tuned fraction=$FRACTION"
srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_train_launcher.sh" $LAUNCHER_ARGS
EOT

    echo "=== Fine-tuned fraction=$FRACTION ==="
    mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"
    sbatch --export=ALL $SBATCH_SCRIPT
    rm $SBATCH_SCRIPT

    # --- Feature-level KD ---
    MODEL_CONFIG="$TREEMORT_REPO_PATH/configs/model/flair_unet_kd_feature.txt"
    LAUNCHER_ARGS="\"$MODEL_CONFIG\" --data-config \"$DATA_CONFIG\" --train-fraction $FRACTION"

    SBATCH_SCRIPT=$(mktemp)
    cat <<EOT > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=dataeff-kd-${FRAC_TAG}
#SBATCH --account=$PROJECT_NAME
#SBATCH --output=$SCRATCH_OUTPUT_DIR/stdout/%A_%a.out
#SBATCH --error=$SCRATCH_OUTPUT_DIR/stderr/%A_%a.err
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=06:00:00
#SBATCH --partition=$PARTITION_NAME

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

export TREEMORT_OUTPUT_DIR="\${TREEMORT_OUTPUT_DIR:-/users/anisrahm/TreeMort/output}"
export TREEMORT_TEACHER_PATH="$TEACHER_CHECKPOINT"
export GPUS_PER_NODE=$GPUS_PER_NODE

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="1\${SLURM_JOB_ID:0-4}"
export MASTER_ADDR MASTER_PORT

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "[INFO] Data efficiency: Feature KD fraction=$FRACTION"
srun --cpu-bind="v,mask_cpu=\${CPU_BIND_MASKS}" \\
    singularity run "\${SIF}" \\
    bash "$TREEMORT_REPO_PATH/scripts/lumi_train_launcher.sh" $LAUNCHER_ARGS
EOT

    echo "=== Feature KD fraction=$FRACTION ==="
    sbatch --export=ALL $SBATCH_SCRIPT
    rm $SBATCH_SCRIPT

done
