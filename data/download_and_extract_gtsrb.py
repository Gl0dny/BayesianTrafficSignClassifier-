import os
import requests
import zipfile

def download_and_extract_gtsrb():
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    extract_path = 'data/GTSRB'
    os.makedirs(extract_path, exist_ok=True)
    
    zip_path = os.path.join(extract_path, 'GTSRB_Final_Training_Images.zip')
    if not os.path.exists(zip_path):
        print("Downloading GTSRB dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

    print("Extracting GTSRB dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

if __name__ == '__main__':
    download_and_extract_gtsrb()
