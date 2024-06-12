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

REM Inform the user how to run the main script
echo To run the main script, use the following command:
echo python .\control\main.py

echo The script has default values set. To get help with the main script, use the following command:
echo python .\control\main.py --help

REM Inform the user how to deactivate the virtual environment
echo To exit the virtual environment, use the following command:
echo deactivate

REM Keep the virtual environment activated
cmd /k
