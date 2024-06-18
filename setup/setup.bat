@echo off

REM Sprawdź, czy środowisko wirtualne nie istnieje
if not exist "venv" (
    python -m venv venv
    echo Virtual environment 'venv' created.
)

REM Aktywuj środowisko wirtualne
call venv\Scripts\activate.bat
echo Virtual environment 'venv' activated.

REM Sprawdź zainstalowane pakiety przed instalacją nowych
echo Installed packages before installing new ones:
pip list

REM Zainstaluj wymagania
pip install -r setup\requirements.txt
echo Requirements installed.

REM Sprawdź zainstalowane pakiety po instalacji nowych
echo Installed packages after installing new ones:
pip list

REM Poinformuj użytkownika, jak uruchomić główny skrypt
echo To run the main script, use the following command:
echo python .\control\main.py

echo The script has default values set. To get help with the main script, use the following command:
echo python .\control\main.py --help

REM Poinformuj użytkownika, jak dezaktywować środowisko wirtualne
echo To exit the virtual environment, use the following command:
echo deactivate

REM Utrzymuj aktywowane środowisko wirtualne
cmd /k