#!/bin/bash

# Exit if any command fails
set -euo pipefail

# Usage:
#   export TREEMORT_VENV_PATH="/path/to/venv"
#   export TREEMORT_REPO_PATH="/path/to/package"
#   export HPC_TYPE="lumi" # or puhti/local
#   sh $TREEMORT_REPO_PATH/scripts/install_treemort.sh

HPC_TYPE="${HPC_TYPE:-local}"
TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"
TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH:-/users/rahmanan/TreeMort}"

declare -a MODULE_STACK
TORCH_INSTALL_CMD=""

MODULE_USE_CMD=""

case "$HPC_TYPE" in
  lumi)
    MODULE_USE_CMD="module use /appl/local/csc/modulefiles/"
    MODULE_STACK=("LUMI/23.09" "partition/G" "rocm" "pytorch/2.7")
    # Pin to the ROCm wheels that were verified on LUMI; do not install CUDA wheels.
    TORCH_INSTALL_CMD="python -m pip install torch==2.2.2+rocm5.7 torchvision==0.17.2+rocm5.7 torchaudio==2.2.2+rocm5.7 --index-url https://download.pytorch.org/whl/rocm5.7"
    ;;
  puhti)
    MODULE_STACK=("pytorch/2.5")
    TORCH_INSTALL_CMD=""
    ;;
  *)
    MODULE_STACK=("pytorch/2.5")
    ;;
esac

if [ -n "$MODULE_USE_CMD" ]; then
  echo "Running: $MODULE_USE_CMD"
  eval "$MODULE_USE_CMD"
fi

for mod in "${MODULE_STACK[@]}"; do
  echo "Loading module: $mod"
  module load "$mod"
done

echo "Creating virtual environment at: $TREEMORT_VENV_PATH"
python3 -m venv $TREEMORT_VENV_PATH || { echo "Error: Failed to create virtual environment."; exit 1; }

echo "Activating virtual environment."
source $TREEMORT_VENV_PATH/bin/activate || { echo "Error: Failed to activate virtual environment."; exit 1; }

echo "Upgrading pip."
python -m pip install --upgrade pip setuptools wheel build || { echo "Error: Failed to upgrade pip."; exit 1; }

if [ -n "$TORCH_INSTALL_CMD" ]; then
  echo "Installing ROCm-enabled PyTorch stack."
  eval "$TORCH_INSTALL_CMD" || { echo "Error: Failed to install ROCm PyTorch."; exit 1; }
fi

pushd "$TREEMORT_REPO_PATH" >/dev/null

REQ_FILE="requirements.txt"
TMP_REQ=""
if [ -n "$TORCH_INSTALL_CMD" ]; then
  TMP_REQ=$(mktemp)
  grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > "$TMP_REQ"
  REQ_FILE="$TMP_REQ"
fi

echo "Installing dependencies."
python -m pip install --only-binary=:all: --no-cache-dir -r "$REQ_FILE" || {
  echo "Error: Failed to install dependencies."; exit 1;
}

if [ -n "$TMP_REQ" ]; then
  rm -f "$TMP_REQ"
fi

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

echo "Script completed successfully."
