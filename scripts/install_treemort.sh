#!/bin/bash

module load tensorflow

python -m venv --system-site-packages venv
source venv/bin/activate

python -m pip install --user --upgrade --no-deps --force-reinstall .
python -c "import treemort; print(\"TreeMort imported successfully.\")"