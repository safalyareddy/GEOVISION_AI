#!/usr/bin/env bash
set -e

# Create a virtual environment in ./venv and install dependencies
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed into ./venv"

# Print quick instructions
echo "To activate the venv: source venv/bin/activate"
echo "Run tests: PYTHONPATH=. pytest"