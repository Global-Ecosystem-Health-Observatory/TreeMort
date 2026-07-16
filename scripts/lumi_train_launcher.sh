#!/bin/bash
# Runs inside the Singularity container on LUMI.
# Invoked by run_treemort.sh via:
#   srun --cpu-bind=v,mask_cpu=... singularity run $SIF bash $THIS_SCRIPT <config> [flags]
#
# SLURM per-task vars (SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS) are injected
# by srun into each task's environment and inherited by the container automatically.
# All other vars (TREEMORT_VENV_PATH, TREEMORT_REPO_PATH, MASTER_ADDR, MASTER_PORT)
# come from the exported SBATCH job environment.

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

cd "${TREEMORT_REPO_PATH}"

# Use the full venv path to avoid resolving to the CSC AI python3 wrapper via PATH
"${TREEMORT_VENV_PATH}/bin/python3" -m treemort.main "$@"
