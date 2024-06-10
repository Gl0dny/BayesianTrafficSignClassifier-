import os
import zipfile

def extract_gtsrb(extract_folder='Traffic_Signs'):
    extract_path = 'data/GTSRB'
    os.makedirs(extract_path, exist_ok=True)

    # Sprawdź, czy dane zostały już wypakowane
    if os.path.exists(os.path.join(extract_path, extract_folder)):
        print("The GTSRB dataset has already been extracted.")
        return

    zip_path = os.path.join(extract_path, 'gtsrb.zip')
    
    if not os.path.exists(zip_path):
        print("Error: The file 'gtsrb.zip' does not exist in the data/GTSRB directory.")
        return

    print(f"Extracting GTSRB dataset to {extract_folder} folder...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extract_path, extract_folder))
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: The file 'gtsrb.zip' is not a valid zip file.")

if __name__ == '__main__':
    extract_gtsrb()

