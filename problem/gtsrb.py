import os
import zipfile

class GTSRB:
    def __init__(self, extract_path, zip_path):
        """
        Inicjalizuje klasę GTSRB z określonymi ścieżkami do wypakowania i pliku zip.

        Parameters:
        - extract_path (str): Ścieżka do folderu, w którym dane mają być wypakowane.
        - zip_path (str): Ścieżka do pliku zip z danymi GTSRB.
        """
        self.extract_path = extract_path
        self.zip_path = zip_path

    def extract(self):
        """
        Funkcja wypakowuje plik z danymi GTSRB (German Traffic Sign Recognition Benchmark) do określonego folderu.

        - Sprawdza, czy dane zostały już wypakowane. Jeśli tak, wypisuje odpowiedni komunikat i kończy działanie.
        - Jeśli folder docelowy nie istnieje, tworzy go.
        - Sprawdza, czy plik zip istnieje. Jeśli nie, wypisuje komunikat o błędzie i kończy działanie.
        - Próbuje wypakować zawartość pliku zip. Jeśli plik zip jest uszkodzony, wypisuje komunikat o błędzie.
        """
        # Sprawdź, czy dane zostały już wypakowane
        if os.path.exists(self.extract_path):
            print("The GTSRB dataset has already been extracted.")
            return

        os.makedirs(self.extract_path, exist_ok=True)  # Tworzy folder, jeśli nie istnieje

        # Sprawdź, czy plik zip istnieje
        if not os.path.exists(self.zip_path):
            print(f"Error: The file '{self.zip_path}' does not exist.")
            return

        print(f"Extracting GTSRB dataset to {self.extract_path} folder...")
        try:
            # Wypakuj zawartość pliku zip
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"Error: The file '{self.zip_path}' is not a valid zip file.")


# Dokumentacja:
# Klasa GTSRB:

#     Opis: Klasa odpowiedzialna za zarządzanie wypakowywaniem danych GTSRB (German Traffic Sign Recognition Benchmark) z pliku zip.

# __init__(self, extract_path, zip_path):

#     Opis: Inicjalizuje klasę GTSRB z określonymi ścieżkami do wypakowania i pliku zip.
#     Parametry:
#         extract_path (str): Ścieżka do folderu, w którym dane mają być wypakowane.
#         zip_path (str): Ścieżka do pliku zip z danymi GTSRB.

# extract(self):

#     Opis: Funkcja wypakowuje plik z danymi GTSRB (German Traffic Sign Recognition Benchmark) do określonego folderu.
#     Działanie:
#         Sprawdza, czy dane zostały już wypakowane. Jeśli tak, wypisuje odpowiedni komunikat i kończy działanie.
#         Jeśli folder docelowy nie istnieje, tworzy go.
#         Sprawdza, czy plik zip istnieje. Jeśli nie, wypisuje komunikat o błędzie i kończy działanie.
#         Próbuje wypakować zawartość pliku zip. Jeśli plik zip jest uszkodzony, wypisuje komunikat o błędzie.
