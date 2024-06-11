import os
import sys
from datetime import datetime
import subprocess

class Tee:
    def __init__(self, name, mode):
        log_dir = os.path.dirname(name)
        os.makedirs(log_dir, exist_ok=True)  # Tworzy folder logs, jeśli nie istnieje
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Tworzy folder logs, jeśli nie istnieje
        if os.path.exists(log_file):
            os.remove(log_file)
        sys.stdout = Tee(log_file, 'a')  # Przekierowanie wyjścia do pliku i terminala

    def log(self, message):
        """
        Zapisuje wiadomość do pliku dziennika z datą i czasem.
        
        Parameters:
        - message: Wiadomość do zapisania w pliku dziennika.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f'{timestamp} - {message}\n'
        sys.stdout.write(full_message)  # Wyświetla w terminalu
        with open(self.log_file, 'a') as f:
            f.write(full_message)  # Zapisuje w pliku dziennika

    def run_script(self, script_name, args=None, python_executable=sys.executable):
        """
        Uruchamia skrypt i przekierowuje jego wyjście do pliku dziennika.
        
        Parameters:
        - script_name: Nazwa skryptu do uruchomienia.
        - args: Lista argumentów do przekazania do skryptu.
        - python_executable: Ścieżka do interpretera Pythona.
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
