#!/bin/bash

# Exit if any command fails
set -e

# Usage:
# export TREEMORT_VENV_PATH="/path/to/venv"
# export TREEMORT_REPO_PATH="/path/to/package"
#
# sh $TREEMORT_REPO_PATH/scripts/install_treemort.sh

MODULE_NAME="pytorch/2.3"

TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"
TREEMORT_REPO_PATH="${TREEMORT_REPO_PATH:-/users/rahmanan/TreeMort}"

echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

echo "Creating virtual environment at: $TREEMORT_VENV_PATH"
python3 -m venv --system-site-packages $TREEMORT_VENV_PATH || { echo "Error: Failed to create virtual environment."; exit 1; }

echo "Activating virtual environment."
source $TREEMORT_VENV_PATH/bin/activate || { echo "Error: Failed to activate virtual environment."; exit 1; }

echo "Upgrading pip."
python -m pip install --upgrade pip || { echo "Error: Failed to upgrade pip."; exit 1; }

echo "Installing package from: $TREEMORT_REPO_PATH"
python -m pip install -e $TREEMORT_REPO_PATH || { echo "Error: Failed to install the package."; exit 1; }

echo "Verifying TreeMort installation."
python -c "import treemort; print('TreeMort imported successfully.')" || { echo "Error: Failed to import TreeMort."; exit 1; }

echo "Script completed successfully."
