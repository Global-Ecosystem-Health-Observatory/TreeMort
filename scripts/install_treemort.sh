#!/bin/bash

MODULE_NAME="pytorch/2.3"
VENV_PATH="/projappl/project_2004205/rahmanan/venv"
PACKAGE_DIR="/users/rahmanan/TreeMort"

module load $MODULE_NAME

python3 -m venv --system-site-packages $VENV_PATH

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

source $VENV_PATH/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install -e $PACKAGE_DIR

if [ $? -ne 0 ]; then
    echo "Error: Failed to install the package."
    exit 1
fi

python -c "import treemort; print('TreeMort imported successfully.')"

if [ $? -ne 0 ]; then
    echo "Error: Failed to import TreeMort."
    exit 1
fi

echo "Script completed successfully."