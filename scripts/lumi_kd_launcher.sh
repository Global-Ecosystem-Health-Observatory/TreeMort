#!/bin/bash
# Runs inside the Singularity container on LUMI for KD training.
# Invoked by submit_treemort_kd.sh via:
#   srun --cpu-bind=v,mask_cpu=... singularity run $SIF bash $THIS_SCRIPT <config> [flags]

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

cd "${TREEMORT_REPO_PATH}"

"${TREEMORT_VENV_PATH}/bin/python3" -m treemort.main_kd "$@"
