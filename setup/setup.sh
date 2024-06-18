#!/bin/bash

# Tworzenie środowiska wirtualnego, jeśli nie istnieje
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
fi

# Aktywacja środowiska wirtualnego
source venv/bin/activate
echo "Virtual environment 'venv' activated."

# Sprawdzenie zainstalowanych pakietów przed instalacją nowych
echo "Installed packages before installing new ones:"
pip list

# Instalacja wymagań
pip install -r setup/requirements.txt
echo "Requirements installed."

# Sprawdzenie zainstalowanych pakietów po instalacji nowych
echo "Installed packages after installing new ones:"
pip list

# Informacja dla użytkownika, jak uruchomić główny skrypt
echo "To run the main script, use the following command:"
echo "python control/main.py"

echo "The script has default values set. To get help with the main script, use the following command:"
echo "python control/main.py --help"

# Informacja dla użytkownika, jak dezaktywować środowisko wirtualne
echo "To exit the virtual environment, use the following command:"
echo "deactivate"

# Utrzymanie aktywowanego środowiska wirtualnego
$SHELL
