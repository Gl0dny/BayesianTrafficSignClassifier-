#!/bin/bash

# Create virtual environment if it doesn't exist
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

# Install requirements
pip install -r setup/requirements.txt
echo "Requirements installed."

# Check installed packages after installing new ones
echo "Installed packages after installing new ones:"
pip list

# Inform the user how to run the main script
echo "To run the main script, use the following command:"
echo "python control/main.py"

echo "The script has default values set. To get help with the main script, use the following command:"
echo "python control/main.py --help"

# Inform the user how to deactivate the virtual environment
echo "To exit the virtual environment, use the following command:"
echo "deactivate"

# Keep the virtual environment activated
$SHELL
