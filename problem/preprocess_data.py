import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import sys

def normalize_hu_moments(hu_moments):
    """
    Normalizuje momenty Hu, stosując skalę logarytmiczną.
    
    Parameters:
    - hu_moments: Tablica momentów Hu.
    
    Returns:
    - znormalizowane momenty Hu.
    """
    for i in range(len(hu_moments)):
        for j in range(len(hu_moments[i])):
            hu_moments[i][j] = -np.sign(hu_moments[i][j]) * np.log10(np.abs(hu_moments[i][j]))
    return hu_moments

def load_preprocess_gtsrb_data(data_dir):
    """
    Funkcja ładuje i przetwarza dane GTSRB, obliczając momenty Hu dla każdego obrazu.

    Parameters:
    - data_dir: ścieżka do katalogu z danymi GTSRB.

    Returns:
    - images: tablica z obrazami.
    - hu_moments: tablica z momentami Hu.
    - labels: tablica z etykietami klas.
    """
    images = []
    hu_moments = []
    labels = []
    classes = 8  # Liczba klas w zbiorze danych

    for class_id in range(classes):  # Iteracja przez każdą klasę
        class_dir = os.path.join(data_dir, 'train', str(class_id))
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):  # Iteracja przez każdy obraz w klasie
            try:
                image = Image.open(os.path.join(class_dir, img_name))
                # Konwersja obrazu do skali szarości
                image = image.convert('L')
                image = image.resize((64, 64))
                image_array = np.array(image)

                # Obliczanie momentów Hu
                moments = cv2.moments(image_array)
                hu_moments_image = cv2.HuMoments(moments).flatten()

                images.append(image_array)
                hu_moments.append(hu_moments_image)
                labels.append(class_id)
            except Exception as e:
                print("Error processing image:", e)

    # Normalizacja momentów Hu
    hu_moments = normalize_hu_moments(np.array(hu_moments))

    return np.array(images), hu_moments, np.array(labels)

def split_train_test_data(data, test_size=0.2, random_state=42):
    """
    Funkcja dzieli dane na zestawy treningowe i testowe oraz zapisuje je do plików .npy.

    Parameters:
    - data: ścieżka do katalogu z danymi.
    - test_size: ułamek danych do zestawu testowego.
    - random_state: losowy seed dla podziału danych.

    Returns:
    - X_train: obrazy treningowe.
    - X_test: obrazy testowe.
    - hu_train: momenty Hu dla zestawu treningowego.
    - hu_test: momenty Hu dla zestawu testowego.
    - y_train: etykiety dla zestawu treningowego.
    - y_test: etykiety dla zestawu testowego.
    """
    if os.path.exists(os.path.join(data, 'X_train.npy')):
        print("Dane zostały już przetworzone. Ładowanie z plików numpy...")
        X_train = np.load(os.path.join(data, 'X_train.npy'))
        X_test = np.load(os.path.join(data, 'X_test.npy'))
        hu_train = np.load(os.path.join(data, 'hu_train.npy'))
        hu_test = np.load(os.path.join(data, 'hu_test.npy'))
        y_train = np.load(os.path.join(data, 'y_train.npy'))
        y_test = np.load(os.path.join(data, 'y_test.npy'))
    else:
        # Wczytywanie i przetwarzanie danych
        images, hu_moments, labels = load_preprocess_gtsrb_data(data)
        print(f'Loaded {len(images)} images with {len(labels)} labels.')

        # Sprawdzanie kształtu danych
        print(images.shape, labels.shape)
        print(hu_moments.shape)

        # Podział na zestaw treningowy i testowy
        X_train, X_test, hu_train, hu_test, y_train, y_test = train_test_split(
            images, hu_moments, labels, test_size=test_size, random_state=random_state
        )
        print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

        # Zapisywanie przetworzonych danych
        np.save(os.path.join(data, 'X_train.npy'), X_train)
        np.save(os.path.join(data, 'X_test.npy'), X_test)
        np.save(os.path.join(data, 'hu_train.npy'), hu_train)
        np.save(os.path.join(data, 'hu_test.npy'), hu_test)
        np.save(os.path.join(data, 'y_train.npy'), y_train)
        np.save(os.path.join(data, 'y_test.npy'), y_test)

    return X_train, X_test, hu_train, hu_test, y_train, y_test

def log_hu_moments(hu_moments, labels, output_file):
    """
    Zapisuje momenty Hu dla każdej klasy do pliku tekstowego.
    
    Parameters:
    - hu_moments: tablica z momentami Hu.
    - labels: etykiety klas.
    - output_file: ścieżka do pliku wyjściowego.
    """
    with open(output_file, 'w') as f:
        last_sample_number = 0
        classes = np.unique(labels)
        for class_id in classes:
            f.write(f'Class {class_id} Hu Moments:\n')
            class_indices = np.where(labels == class_id)[0]
            for idx, moment_idx in enumerate(class_indices, start=1):
                f.write(f'Sample {last_sample_number + idx} Hu Moments: {hu_moments[moment_idx]}\n')
            last_sample_number += len(class_indices)
            f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    
    # Pobieranie ścieżki do katalogu z danymi z argumentu wiersza poleceń
    data_dir = sys.argv[1]

    # Podział na zestaw treningowy i testowy
    X_train, X_test, hu_train, hu_test, y_train, y_test = split_train_test_data(
        data=data_dir, test_size=0.2, random_state=42
    )

    # Upewnij się, że folder logs istnieje
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Logowanie momentów Hu do pliku
    log_file = os.path.join(log_dir, 'hu_moments_log.txt')
    log_hu_moments(hu_train, y_train, log_file)

    print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')
    print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
    print("Data preprocessing complete. Hu moments logged to", log_file)
    print("Data preprocessing complete.")

# Opis funkcji:
# normalize_hu_moments(hu_moments)

# Normalizuje momenty Hu, stosując skalę logarytmiczną, aby uczynić je bardziej porównywalnymi i skalowalnymi.

#     Parameters:
#         hu_moments (ndarray): Tablica momentów Hu.
#     Returns:
#         ndarray: Znormalizowane momenty Hu.

# load_preprocess_gtsrb_data(data_dir)

# Funkcja ładuje i przetwarza dane GTSRB, obliczając momenty Hu dla każdego obrazu.

#     Parameters:
#         data_dir (str): Ścieżka do katalogu z danymi GTSRB.
#     Returns:
#         ndarray: Tablica z obrazami.
#         ndarray: Tablica z momentami Hu.
#         ndarray: Tablica z etykietami klas.

# split_train_test_data(data, test_size=0.2, random_state=42)

# Funkcja dzieli dane na zestawy treningowe i testowe oraz zapisuje je do plików .npy.

#     Parameters:
#         data (str): Ścieżka do katalogu z danymi.
#         test_size (float): Ułamek danych do zestawu testowego.
#         random_state (int): Losowy seed dla podziału danych.
#     Returns:
#         ndarray: Obrazy treningowe.
#         ndarray: Obrazy testowe.
#         ndarray: Momenty Hu dla zestawu treningowego.
#         ndarray: Momenty Hu dla zestawu testowego.
#         ndarray: Etykiety dla zestawu treningowego.
#         ndarray: Etykiety dla zestawu testowego.

# log_hu_moments(hu_moments, labels, output_file)

# Zapisuje momenty Hu dla każdej klasy do pliku tekstowego.

#     Parameters:
#         hu_moments (ndarray): Tablica z momentami Hu.
#         labels (ndarray): Etykiety klas.
#         output_file (str): Ścieżka do pliku wyjściowego.