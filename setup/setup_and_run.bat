@echo off

REM Check if the virtual environment doesn't exist
if not exist "venv" (
    python -m venv venv
    echo Virtual environment 'venv' created.
)

REM Activate the virtual environment
call venv\Scripts\activate.bat
echo Virtual environment 'venv' activated.

REM Check installed packages before installing new ones
echo Installed packages before installing new ones:
pip list

REM Install the requirements
pip install -r setup\requirements.txt
echo Requirements installed.

REM Check installed packages after installing new ones
echo Installed packages after installing new ones:
pip list

REM Run the main script
python control\main.py
echo Script executed.

REM Keep the virtual environment activated
cmd /k
