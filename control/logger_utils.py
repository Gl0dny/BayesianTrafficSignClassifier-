import os
import sys
from datetime import datetime
import subprocess

class Tee:
    def __init__(self, name, mode):
        """
        Klasa Tee przechwytuje wyjście i przekierowuje je zarówno do pliku, jak i do terminala.

        Parameters:
        - name (str): Nazwa pliku, do którego będzie zapisywane wyjście.
        - mode (str): Tryb otwarcia pliku (np. 'a' dla dopisania do pliku).
        """
        log_dir = os.path.dirname(name)
        os.makedirs(log_dir, exist_ok=True)  # Tworzy folder, jeśli nie istnieje
        self.name = name
        self.mode = mode
        self.stdout = sys.stdout

    def write(self, data):
        """
        Zapisuje dane zarówno do pliku, jak i do terminala.

        Parameters:
        - data (str): Dane do zapisania.
        """
        with open(self.name, self.mode, encoding='utf-8') as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        """
        Spłukuje buforowane dane do pliku i terminala.
        """
        self.stdout.flush()

class Logger:
    def __init__(self, log_file):
        """
        Klasa Logger zarządza zapisywaniem komunikatów logów do pliku oraz ich wyświetlaniem w terminalu.

        Parameters:
        - log_file (str): Ścieżka do pliku dziennika.
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Tworzy folder, jeśli nie istnieje
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except PermissionError as e:
            print(f"PermissionError: {e}. Make sure the file is not being used by another process.")
        sys.stdout = Tee(log_file, 'a')  # Przekierowanie wyjścia do pliku i terminala

    def log(self, message):
        """
        Zapisuje wiadomość do pliku dziennika z datą i czasem.

        Parameters:
        - message (str): Wiadomość do zapisania w pliku dziennika.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f'{timestamp} - {message}\n'
        sys.stdout.write(full_message)  # Wyświetla w terminalu

    def run_script(self, script_name, args=None, python_executable=sys.executable):
        """
        Uruchamia skrypt i przekierowuje jego wyjście do pliku dziennika.

        Parameters:
        - script_name (str): Nazwa skryptu do uruchomienia.
        - args (list): Lista argumentów do przekazania do skryptu.
        - python_executable (str): Ścieżka do interpretera Pythona.
        """
        command = [python_executable, script_name] + (args if args else [])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        if process.returncode == 0:
            self.log(f"{script_name} completed successfully.")
        else:
            self.log(f"{script_name} failed with error: {stderr}")


# Dokumentacja:
# Klasa Tee:

#     Opis: Klasa Tee przechwytuje wyjście i przekierowuje je zarówno do pliku, jak i do terminala.
#     Metody:
#         __init__(self, name, mode): Inicjalizuje klasę Tee, tworząc folder (jeśli nie istnieje) i otwierając plik w określonym trybie.
#         write(self, data): Zapisuje dane zarówno do pliku, jak i do terminala.
#         flush(self): Spłukuje buforowane dane do pliku i terminala.

# Klasa Logger:

#     Opis: Klasa Logger zarządza zapisywaniem komunikatów logów do pliku oraz ich wyświetlaniem w terminalu.
#     Metody:
#         __init__(self, log_file): Inicjalizuje klasę Logger, tworząc folder (jeśli nie istnieje) i ustawiając przekierowanie wyjścia do pliku i terminala.
#         log(self, message): Zapisuje wiadomość do pliku dziennika z datą i czasem.
#         run_script(self, script_name, args=None, python_executable=sys.executable): Uruchamia skrypt i przekierowuje jego wyjście do pliku dziennika.

# Funkcja log:

#     Opis: Zapisuje wiadomość do pliku dziennika z datą i czasem.
#     Parametry:
#         message (str): Wiadomość do zapisania w pliku dziennika.

# Funkcja run_script:

#     Opis: Uruchamia skrypt i przekierowuje jego wyjście do pliku dziennika.
#     Parametry:
#         script_name (str): Nazwa skryptu do uruchomienia.
#         args (list): Lista argumentów do przekazania do skryptu.
#         python_executable (str): Ścieżka do interpretera Pythona.