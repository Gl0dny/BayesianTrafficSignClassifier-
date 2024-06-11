import os
import zipfile

def extract_gtsrb(extract_folder='Traffic_Signs'):
    """
    Funkcja wypakowuje plik z danymi GTSRB (German Traffic Sign Recognition Benchmark) do określonego folderu.

    Parameters:
    - extract_folder (str): Nazwa folderu, do którego zostaną wypakowane dane. Domyślnie 'Traffic_Signs'.
    """
    extract_path = 'data/GTSRB'
    os.makedirs(extract_path, exist_ok=True)

    # Sprawdź, czy dane zostały już wypakowane
    if os.path.exists(os.path.join(extract_path, extract_folder)):
        print("The GTSRB dataset has already been extracted.")
        return

    zip_path = os.path.join(extract_path, 'gtsrb.zip')
    
    # Sprawdź, czy plik zip istnieje
    if not os.path.exists(zip_path):
        print("Error: The file 'gtsrb.zip' does not exist in the data/GTSRB directory.")
        return

    print(f"Extracting GTSRB dataset to {extract_folder} folder...")
    try:
        # Wypakuj zawartość pliku zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extract_path, extract_folder))
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: The file 'gtsrb.zip' is not a valid zip file.")

if __name__ == '__main__':
    extract_gtsrb()

# Opis funkcji:
# extract_gtsrb(extract_folder='Traffic_Signs')

# Funkcja wypakowuje plik z danymi GTSRB (German Traffic Sign Recognition Benchmark) do określonego folderu. Jeśli folder już istnieje, funkcja informuje, że dane zostały już wypakowane i kończy działanie. Jeśli plik zip nie istnieje, zgłaszany jest błąd.

#     Parameters:
#         extract_folder (str): Nazwa folderu, do którego zostaną wypakowane dane. Domyślnie 'Traffic_Signs'.

# Szczegóły funkcji:

#     Tworzenie folderu docelowego:
#         extract_path jest ustawiony na 'data/GTSRB'.
#         os.makedirs(extract_path, exist_ok=True) tworzy folder, jeśli nie istnieje.

#     Sprawdzanie, czy dane zostały już wypakowane:
#         Jeśli os.path.exists(os.path.join(extract_path, extract_folder)) zwróci True, funkcja informuje, że dane są już wypakowane i kończy działanie.

#     Sprawdzanie, czy plik zip istnieje:
#         Jeśli os.path.exists(zip_path) zwróci False, funkcja informuje, że plik zip nie istnieje i kończy działanie.

#     Wypakowanie pliku zip:
#         zipfile.ZipFile(zip_path, 'r') otwiera plik zip do odczytu.
#         zip_ref.extractall(os.path.join(extract_path, extract_folder)) wypakowuje wszystkie pliki do określonego folderu.
#         Jeśli plik zip jest uszkodzony, zgłaszany jest błąd zipfile.BadZipFile.

# Główna część skryptu:

#     if __name__ == '__main__': sprawdza, czy skrypt jest uruchamiany bezpośrednio, a nie importowany jako moduł.
#     extract_gtsrb() wywołuje funkcję wypakowywania danych GTSRB.