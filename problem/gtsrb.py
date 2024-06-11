import os
import zipfile

class GTSRB:
    def __init__(self, extract_path, zip_path):
        self.extract_path = extract_path
        self.zip_path = zip_path

    def extract(self):
        """
        Funkcja wypakowuje plik z danymi GTSRB (German Traffic Sign Recognition Benchmark) do określonego folderu.
        """
        # Sprawdź, czy dane zostały już wypakowane
        if os.path.exists(self.extract_path):
            print("The GTSRB dataset has already been extracted.")
            return

        os.makedirs(self.extract_path, exist_ok=True)

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
