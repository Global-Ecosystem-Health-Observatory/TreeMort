#!/bin/bash
# Example W&B-enabled workflows for TreeMort.
# These are illustrative commands; adjust paths/artifact names for your project.

REPO_ROOT=${TREEMORT_REPO_PATH:-"$(pwd)"}
MODEL_CFG="$REPO_ROOT/configs/model/flair_unet_sdt.txt"
DATA_CFG="$REPO_ROOT/configs/data/finland.txt"

set -euo pipefail

function train_from_scratch() {
    python3 -m treemort.main "$MODEL_CFG" \
        --data-config "$DATA_CFG" \
        --wandb --wandb-entity your_org --wandb-project treemort \
        --dataset-artifact dataset-FIN-25cm-4ch:v1
}

function resume_same_country() {
    python3 -m treemort.main "$MODEL_CFG" \
        --data-config "$DATA_CFG" \
        --wandb --wandb-run-id <existing_run_id>
}

function transfer_from_finland_to_switzerland() {
    python3 -m treemort.main "$REPO_ROOT/configs/model/flair_unet_sdt.txt" \
        --data-config "$REPO_ROOT/configs/data/switzerland.txt" \
        --wandb --init-model-artifact model-unet-FIN-4ch-25cm:production
}

function promote_to_production() {
    python3 -m tools.promote_artifact --artifact model-unet-CH-3ch-10cm:staging --alias production
}

echo "Defined helper functions: train_from_scratch, resume_same_country, transfer_from_finland_to_switzerland, promote_to_production"
