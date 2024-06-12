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

REM Dokumentacja:

REM Sprawdź, czy środowisko wirtualne nie istnieje
REM     Opis: Sprawdza, czy katalog "venv" istnieje, a jeśli nie, tworzy nowe środowisko wirtualne.

REM Aktywuj środowisko wirtualne
REM     Opis: Aktywuje środowisko wirtualne poprzez uruchomienie skryptu aktywacyjnego.

REM Sprawdź zainstalowane pakiety przed instalacją nowych
REM     Opis: Wyświetla listę zainstalowanych pakietów przed instalacją nowych pakietów.

REM Zainstaluj wymagania
REM     Opis: Instaluje wymagania z pliku requirements.txt znajdującego się w katalogu setup.

REM Sprawdź zainstalowane pakiety po instalacji nowych
REM     Opis: Wyświetla listę zainstalowanych pakietów po instalacji nowych pakietów.

REM Poinformuj użytkownika, jak uruchomić główny skrypt
REM     Opis: Wyświetla instrukcje dotyczące uruchamiania głównego skryptu.

REM Poinformuj użytkownika, jak dezaktywować środowisko wirtualne
REM     Opis: Wyświetla instrukcje dotyczące dezaktywacji środowiska wirtualnego.

REM Utrzymuj aktywowane środowisko wirtualne
REM     Opis: Utrzymuje otwarte okno wiersza poleceń z aktywowanym środowiskiem wirtualnym.
