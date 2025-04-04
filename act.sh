#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Set PYTHONPATH directly (not appending)
export PYTHONPATH="$PROJECT_ROOT"

# Install/upgrade pip
python -m pip install --upgrade pip

# Install the package in development mode
pip install -e .

echo "Environment setup complete!"
echo "Virtual environment activated."
echo "PYTHONPATH set to: $PYTHONPATH" 