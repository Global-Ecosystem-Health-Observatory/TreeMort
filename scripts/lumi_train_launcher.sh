#!/bin/bash
# Runs inside the Singularity container on LUMI.
# Invoked by run_treemort.sh via:
#   srun --cpu-bind=v,mask_cpu=... singularity run $SIF bash $THIS_SCRIPT <config> [flags]
#
# SLURM per-task vars (SLURM_PROCID, SLURM_LOCALID) are injected by srun into each
# task's environment and are inherited by the container automatically.
# All other vars (TREEMORT_VENV_PATH, TREEMORT_REPO_PATH, GPUS_PER_NODE,
# MASTER_ADDR, MASTER_PORT) come from the exported SBATCH job environment.

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

cd "${TREEMORT_REPO_PATH}"

# Use the full venv path to avoid resolving to the CSC AI python3 wrapper via PATH
"${TREEMORT_VENV_PATH}/bin/python3" -m torch.distributed.run \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m treemort.main "$@"
