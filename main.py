#!/usr/bin/env python3

import subprocess
import sys

python_executable = sys.executable  # dynamicznie pobiera ścieżkę do interpretera Pythona

def main():
    # Krok 1: Rozpakowanie danych
    print("Step 1: Extracting GTSRB data...")
    subprocess.run([python_executable, 'data/GTSRB/extract_gtsrb.py'], check=True)

    # Krok 2: Przetwarzanie danych
    print("Step 2: Preprocessing data...")
    subprocess.run([python_executable, 'scripts/preprocess_data.py'], check=True)

    # Krok 3: Wizualizacja przykładowych danych
    print("Step 3: Visualizing sample data...")
    subprocess.run([python_executable, 'scripts/visualize_samples.py'], check=True)

if __name__ == '__main__':
    main()
