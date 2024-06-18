@echo off

REM Sprawdź, czy środowisko wirtualne nie istnieje
if not exist "venv" (
    python -m venv venv
    echo Virtual environment 'venv' created.
)
