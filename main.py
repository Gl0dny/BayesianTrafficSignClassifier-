import os

def main():
    # Krok 1: Pobranie i rozpakowanie danych
    print("Step 1: Downloading and extracting GTSRB data...")
    os.system('python data/download_and_extract_gtsrb.py')

    # Krok 2: Przetwarzanie danych
    print("Step 2: Preprocessing data...")
    os.system('python scripts/preprocess_data.py')

    # Krok 3: Wizualizacja przykładowych danych
    print("Step 3: Visualizing sample data...")
    os.system('python scripts/visualize_samples.py')

if __name__ == '__main__':
    main()
