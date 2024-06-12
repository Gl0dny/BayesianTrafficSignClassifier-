#!/bin/bash

# Create the virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
fi

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment 'venv' activated."

# Check installed packages before installing new ones
echo "Installed packages before installing new ones:"
pip list

# Install the requirements
pip install -r setup/requirements.txt
echo "Requirements installed."

# Check installed packages after installing new ones
echo "Installed packages after installing new ones:"
pip list

# Run the main script
python control/main.py
echo "Script executed."

# Keep the virtual environment activated
$SHELL
