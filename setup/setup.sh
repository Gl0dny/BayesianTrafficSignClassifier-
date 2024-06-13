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

# Dokumentacja:

# Tworzenie środowiska wirtualnego, jeśli nie istnieje
#     Opis: Sprawdza, czy katalog "venv" istnieje, a jeśli nie, tworzy nowe środowisko wirtualne przy użyciu python3 -m venv.

# Aktywacja środowiska wirtualnego
#     Opis: Aktywuje środowisko wirtualne poprzez uruchomienie skryptu aktywacyjnego (source venv/bin/activate).

# Sprawdzenie zainstalowanych pakietów przed instalacją nowych
#     Opis: Wyświetla listę zainstalowanych pakietów przed instalacją nowych pakietów (pip list).

# Instalacja wymagań
#     Opis: Instaluje wymagania z pliku requirements.txt znajdującego się w katalogu setup (pip install -r setup/requirements.txt).

# Sprawdzenie zainstalowanych pakietów po instalacji nowych
#     Opis: Wyświetla listę zainstalowanych pakietów po instalacji nowych pakietów (pip list).

# Informacja dla użytkownika, jak uruchomić główny skrypt
#     Opis: Wyświetla instrukcje dotyczące uruchamiania głównego skryptu (python control/main.py).

# Informacja dla użytkownika, jak dezaktywować środowisko wirtualne
#     Opis: Wyświetla instrukcje dotyczące dezaktywacji środowiska wirtualnego (deactivate).

# Utrzymanie aktywowanego środowiska wirtualnego
#     Opis: Utrzymuje otwarte okno powłoki z aktywowanym środowiskiem wirtualnym ($SHELL).
