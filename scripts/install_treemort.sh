#!/bin/bash

# Exit if any command fails
set -euo pipefail

# Usage:
#   export TREEMORT_VENV_PATH="/path/to/venv"
#   export TREEMORT_REPO_PATH="/path/to/package"
#   export HPC_TYPE="lumi" # or puhti/local
#   sh $TREEMORT_REPO_PATH/scripts/install_treemort.sh

HPC_TYPE="${HPC_TYPE:-local}"
TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_462001070/aurahman/venv}"
TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH:-/users/aurahman/TreeMort}"

if [ "$HPC_TYPE" == "lumi" ]; then

    SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"

    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings

    echo "Creating virtual environment (inheriting container site-packages) at: $TREEMORT_VENV_PATH"
    singularity exec "$SIF" python3 -m venv --system-site-packages "$TREEMORT_VENV_PATH" \
        || { echo "Error: Failed to create virtual environment."; exit 1; }

    pushd "$TREEMORT_REPO_PATH" >/dev/null

    # Filter torch packages — already provided by the container's site-packages
    TMP_REQ=$(mktemp)
    grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > "$TMP_REQ"

    echo "Installing dependencies."
    singularity exec "$SIF" bash -c "
source '$TREEMORT_VENV_PATH/bin/activate' &&
python -m pip install --upgrade pip setuptools wheel build &&
python -m pip install --no-cache-dir -r '$TMP_REQ' &&
python -m pip install --no-cache-dir -e '$TREEMORT_REPO_PATH'
" || { echo "Error: Failed to install dependencies."; exit 1; }

    rm -f "$TMP_REQ"
    popd >/dev/null

    echo "Verifying TreeMort installation."
    singularity exec "$SIF" bash -c "
source '$TREEMORT_VENV_PATH/bin/activate' &&
python - <<'PY'
import importlib, sys
for name in ('treemort', 'tree_mort', 'TreeMort'):
    try:
        m = importlib.import_module(name)
        print(f\"Imported '{name}' from {getattr(m, '__file__', None)}\")
        break
    except Exception as e:
        last = e
else:
    raise SystemExit(f'Could not import treemort/tree_mort/TreeMort: {last}')
PY
" || { echo "Error: Failed to import TreeMort."; exit 1; }

else
    # Puhti / local: module + pip approach
    declare -a MODULE_STACK
    TORCH_INSTALL_CMD=""

    case "$HPC_TYPE" in
      puhti)
        MODULE_STACK=("pytorch/2.5")
        ;;
      *)
        MODULE_STACK=("pytorch/2.5")
        ;;
    esac

    for mod in "${MODULE_STACK[@]}"; do
      echo "Loading module: $mod"
      module load "$mod"
    done

    echo "Creating virtual environment at: $TREEMORT_VENV_PATH"
    python3 -m venv "$TREEMORT_VENV_PATH" || { echo "Error: Failed to create virtual environment."; exit 1; }

    echo "Activating virtual environment."
    source "$TREEMORT_VENV_PATH/bin/activate" || { echo "Error: Failed to activate virtual environment."; exit 1; }

    echo "Upgrading pip."
    python -m pip install --upgrade pip setuptools wheel build || { echo "Error: Failed to upgrade pip."; exit 1; }

    pushd "$TREEMORT_REPO_PATH" >/dev/null

    echo "Installing dependencies."
    python -m pip install --only-binary=:all: --no-cache-dir -r requirements.txt || {
      echo "Error: Failed to install dependencies."; exit 1;
    }

    echo "Installing package from: $TREEMORT_REPO_PATH"
    python -m pip install --only-binary=:all: --no-cache-dir -e . || {
      echo "Error: Failed to install the TreeMort package."; exit 1;
    }

    popd >/dev/null

    echo "Verifying TreeMort installation."
    python - <<'PY' || { echo "Error: Failed to import TreeMort."; exit 1; }
import importlib, sys
for name in ("treemort", "tree_mort", "TreeMort"):
    try:
        m = importlib.import_module(name)
        print(f"Imported '{name}' from {getattr(m, '__file__', None)}")
        break
    except Exception as e:
        last = e
else:
    raise SystemExit(f"Could not import treemort/tree_mort/TreeMort: {last}")
PY

fi

echo "Script completed successfully."
