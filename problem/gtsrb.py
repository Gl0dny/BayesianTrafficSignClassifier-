import os
import zipfile

class GTSRB:
    def __init__(self, extract_path, zip_path):
        """
        Initializes the GTSRB class with specified paths for extraction and zip file.

        Parameters:
        - extract_path (str): Path to the folder where the data should be extracted.
        - zip_path (str): Path to the zip file containing the GTSRB data.
        """
        self.extract_path = extract_path
        self.zip_path = zip_path

    def extract(self):
        """
        Extracts the GTSRB (German Traffic Sign Recognition Benchmark) dataset to the specified folder.

        - Checks if the data has already been extracted. If so, prints a message and exits.
        - If the target folder does not exist, creates it.
        - Checks if the zip file exists. If not, prints an error message and exits.
        - Attempts to extract the contents of the zip file. If the zip file is corrupted, prints an error message.
        """
        if os.path.exists(self.extract_path):
            print("The GTSRB dataset has already been extracted.")
            return

        os.makedirs(self.extract_path, exist_ok=True)

        if not os.path.exists(self.zip_path):
            print(f"Error: The file '{self.zip_path}' does not exist.")
            return

        print(f"Extracting GTSRB dataset to {self.extract_path} folder...")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"Error: The file '{self.zip_path}' is not a valid zip file.")
