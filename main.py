#!/usr/bin/env python3

import subprocess
import sys
import os
from datetime import datetime
import argparse

python_executable = sys.executable  # dynamicznie pobiera ścieżkę do interpretera Pythona
log_dir = 'logs'
log_file = os.path.join(log_dir, 'progress_log.txt')

class Tee:
    def __init__(self, name, mode):
        os.makedirs(log_dir, exist_ok=True)  # Tworzy folder logs, jeśli nie istnieje
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def log(message):
    """
    Zapisuje wiadomość do pliku dziennika z datą i czasem.
    
    Parameters:
    - message: Wiadomość do zapisania w pliku dziennika.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f'{timestamp} - {message}\n'
    sys.stdout.write(full_message)  # Wyświetla w terminalu
    with open(log_file, 'a') as f:
        f.write(full_message)  # Zapisuje w pliku dziennika

def run_script(script_name, args=None):
    """
    Uruchamia skrypt i przekierowuje jego wyjście do pliku dziennika.
    
    Parameters:
    - script_name: Nazwa skryptu do uruchomienia.
    - args: Lista argumentów do przekazania do skryptu.
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
        log(f"{script_name} completed successfully.")
    else:
        log(f"{script_name} failed with error: {stderr}")

def main(bin_count,data_dir, zip_path):
    # Krok 1: Rozpakowanie danych
    log("Step 1: Extracting GTSRB data started.")
    run_script('data/GTSRB/extract_gtsrb.py', args=[data_dir, zip_path])

    # Krok 2: Przetwarzanie danych
    log("Step 2: Preprocessing data started.")
    run_script('scripts/preprocess_data.py', args=[data_dir])

    # Krok 3: Wizualizacja przykładowych danych
    log("Step 3: Visualizing sample data started.")
    run_script('scripts/debug_visualize_samples.py', args=[data_dir])

    # Krok 4: Uczenie parametrycznego klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego)
    log("Step 4: Training Gaussian Naive Bayes model started.")
    run_script('scripts/train_gaussian_bayes.py', args=[data_dir])

    # Krok 5: Uczenie nieparametrycznego klasyfikatora Bayesa (histogram wielowymiarowy)
    log("Step 5: Training Histogram Bayes model started.")
    run_script('scripts/train_histogram_bayes.py', args=[data_dir, str(bin_count)])

if __name__ == '__main__':
    # Tworzenie lub czyszczenie pliku dziennika na początku
    os.makedirs(log_dir, exist_ok=True)  # Tworzy folder logs, jeśli nie istnieje
    if os.path.exists(log_file):
        os.remove(log_file)
    sys.stdout = Tee(log_file, 'a')  # Przekierowanie wyjścia do pliku i terminala
    
    # Parser argumentów
    parser = argparse.ArgumentParser(description="Run the data processing and training pipeline.")
    parser.add_argument('--bin_count', type=int, default=20, help='Number of bins for histogram model.')
    parser.add_argument('--data_dir', type=str, default='data/GTSRB/Traffic_Signs/', help='Directory containing the data scripts.')
    parser.add_argument('--zip_path', type=str, default='data/GTSRB/gtsrb.zip', help='Path to the GTSRB zip file.')
    args = parser.parse_args()

    log("Process started.")
    main(args.bin_count, args.data_dir, args.zip_path)
    log("Process completed.")
