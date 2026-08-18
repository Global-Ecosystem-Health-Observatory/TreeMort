#!/bin/bash
# Submit the representational analysis job on LUMI (CPU-only, inside container).
# Generates Figures 3-6 and prints linear probing values to stdout.
#
# Prerequisites: feature files must already exist in
#   $TREEMORT_REPO_PATH/output/features_{poland,finland}_{baseline,transfer,feature}/
# Run submit_feature_extraction.sh first if needed.
#
# Usage: bash submit_representational_analysis.sh lumi

if [ "$1" != "lumi" ] || [ "$#" -lt 1 ]; then
    echo "Usage: $0 lumi"
    exit 1
fi

TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"
TREEMORT_VENV_PATH="/projappl/project_462001070/anisrahm/venv"
PROJECT_NAME="project_462001070"
SCRATCH_OUTPUT_DIR="/scratch/$PROJECT_NAME/anisrahm/logs"

mkdir -p "$SCRATCH_OUTPUT_DIR/stdout" "$SCRATCH_OUTPUT_DIR/stderr"

SBATCH_SCRIPT=$(mktemp)
cat > "$SBATCH_SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=rep-analysis
#SBATCH --account=${PROJECT_NAME}
#SBATCH --output=${SCRATCH_OUTPUT_DIR}/stdout/%A.out
#SBATCH --error=${SCRATCH_OUTPUT_DIR}/stderr/%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=small

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

export TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH}"
export TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH}"

echo "[INFO] Running representational analysis"
singularity exec "\${SIF}" \
    "${TREEMORT_VENV_PATH}/bin/python3" "${TREEMORT_REPO_PATH}/misc/representational_analysis.py" \
    --feature-root "${TREEMORT_REPO_PATH}/output" \
    --report-images "${TREEMORT_REPO_PATH}/report/images"
echo "[INFO] Done"
SBATCH

JOB_ID=$(sbatch --export=ALL "$SBATCH_SCRIPT" | awk '{print $NF}')
rm "$SBATCH_SCRIPT"
echo "Submitted representational analysis: job ${JOB_ID}"
echo "  stdout: ${SCRATCH_OUTPUT_DIR}/stdout/${JOB_ID}.out"
