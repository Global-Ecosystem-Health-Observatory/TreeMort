#!/bin/bash
# Runs inside the Singularity container on LUMI.
# Invoked by submit_feature_extraction.sh via:
#   srun --cpu-bind=v,mask_cpu=... singularity run $SIF bash $THIS_SCRIPT [args]
#
# Mirrors lumi_train_launcher.sh but calls extract_features.py (no DDP needed).

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

cd "${TREEMORT_REPO_PATH}"

"${TREEMORT_VENV_PATH}/bin/python3" scripts/extract_features.py "$@"
