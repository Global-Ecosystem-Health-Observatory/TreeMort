#!/bin/bash

module load pytorch/2.3

python -m venv --system-site-packages venv
source venv/bin/activate

python -m pip install --upgrade .
python -c "import treemort; print(\"TreeMort imported successfully.\")"
