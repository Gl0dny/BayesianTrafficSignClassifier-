import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

class HuImageData:
    def __init__(self, data_dir, no_classes, no_features=7, test_size=0.2):
        """
        Inicjalizuje klasę HuImageData z określonymi parametrami.

        Parameters:
        - data_dir (str): Ścieżka do katalogu z danymi GTSRB.
        - no_classes (int): Liczba klas znaków drogowych.
        - no_features (int): Liczba cech do użycia z momentów Hu (domyślnie 7).
        - test_size (float): Ułamek danych przeznaczonych na zestaw testowy (domyślnie 0.2).
        """
        self.data_dir = data_dir
        self.no_classes = no_classes
        self.no_features = no_features
        self.test_size = test_size

    def _normalize_hu_moments(self, hu_moments):
        """
        Normalizuje momenty Hu, stosując skalę logarytmiczną.

        Parameters:
        - hu_moments (ndarray): Tablica momentów Hu.

        Returns:
        - ndarray: Znormalizowane momenty Hu.
        """
        for i in range(len(hu_moments)):
            for j in range(len(hu_moments[i])):
                hu_moments[i][j] = -np.sign(hu_moments[i][j]) * np.log10(np.abs(hu_moments[i][j]))
        return hu_moments

    def _extract_hu_moments_image_data(self):
        """
        Funkcja ładuje i przetwarza dane GTSRB, obliczając momenty Hu dla każdego obrazu.

        Returns:
        - tuple: (ndarray, ndarray, ndarray)
            - images: Tablica z obrazami.
            - hu_moments: Tablica z momentami Hu.
            - labels: Tablica z etykietami klas.
        """
        images = []
        hu_moments = []
        labels = []

        for class_id in range(self.no_classes):  # Iteracja przez każdą klasę
            class_dir = os.path.join(self.data_dir, 'train', str(class_id))
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
                    hu_moments.append(hu_moments_image[:self.no_features])  # Używa tylko określonej liczby cech
                    labels.append(class_id)
                except Exception as e:
                    print("Error processing image:", e)

        # Normalizacja momentów Hu
        hu_moments = self._normalize_hu_moments(np.array(hu_moments))

        return np.array(images), hu_moments, np.array(labels)

    def split_train_test_data(self, random_state=42):
        """
        Funkcja dzieli dane na zestawy treningowe i testowe oraz zapisuje je do plików .npy.

        Parameters:
        - random_state (int): Losowy seed dla podziału danych (domyślnie 42).

        Returns:
        - tuple: (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
            - X_train: Obrazy treningowe.
            - X_test: Obrazy testowe.
            - hu_train: Momentu Hu dla zestawu treningowego.
            - hu_test: Momentu Hu dla zestawu testowego.
            - y_train: Etykiety dla zestawu treningowego.
            - y_test: Etykiety dla zestawu testowego.
        """
        if os.path.exists(os.path.join(self.data_dir, 'hu_train.npy')):
            print("Dane zostały już przetworzone. Ładowanie z plików numpy...")
            X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
            hu_train = np.load(os.path.join(self.data_dir, 'hu_train.npy'))
            hu_test = np.load(os.path.join(self.data_dir, 'hu_test.npy'))
            y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(self.data_dir, 'y_test.npy'))
        else:
            # Wczytywanie i przetwarzanie danych
            images, hu_moments, labels = self._extract_hu_moments_image_data()
            print(f'Loaded {len(images)} images with {len(labels)} labels.')

            # Sprawdzanie kształtu danych
            print(images.shape, labels.shape)
            print(hu_moments.shape)

            # Podział na zestaw treningowy i testowy
            X_train, X_test, hu_train, hu_test, y_train, y_test = train_test_split(
                images, hu_moments, labels, test_size=self.test_size, random_state=random_state
            )
            print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

            # Zapisywanie przetworzonych danych
            np.save(os.path.join(self.data_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(self.data_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(self.data_dir, 'hu_train.npy'), hu_train)
            np.save(os.path.join(self.data_dir, 'hu_test.npy'), hu_test)
            np.save(os.path.join(self.data_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.data_dir, 'y_test.npy'), y_test)

        return X_train, X_test, hu_train, hu_test, y_train, y_test

    def log_hu_moments(self, hu_moments, labels, output_file):
        """
        Zapisuje momenty Hu dla każdej klasy do pliku tekstowego.

        Parameters:
        - hu_moments (ndarray): Tablica z momentami Hu.
        - labels (ndarray): Etykiety klas.
        - output_file (str): Ścieżka do pliku wyjściowego.
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

# Dokumentacja:
# Klasa HuImageData:

#     Opis: Klasa odpowiedzialna za przetwarzanie obrazów znaków drogowych, obliczanie momentów Hu oraz dzielenie danych na zestawy treningowe i testowe.

# __init__(self, data_dir, no_classes, no_features=7, test_size=0.2):

#     Opis: Inicjalizuje klasę HuImageData z określonymi parametrami.
#     Parametry:
#         data_dir (str): Ścieżka do katalogu z danymi GTSRB.
#         no_classes (int): Liczba klas znaków drogowych.
#         no_features (int): Liczba cech do użycia z momentów Hu (domyślnie 7).
#         test_size (float): Ułamek danych przeznaczonych na zestaw testowy (domyślnie 0.2).

# _normalize_hu_moments(self, hu_moments):

#     Opis: Normalizuje momenty Hu, stosując skalę logarytmiczną.
#     Parametry:
#         hu_moments (ndarray): Tablica momentów Hu.
#     Zwraca:
#         ndarray: Znormalizowane momenty Hu.

# _extract_hu_moments_image_data(self):

#     Opis: Funkcja ładuje i przetwarza dane GTSRB, obliczając momenty Hu dla każdego obrazu.
#     Zwraca:
#         tuple: (ndarray, ndarray, ndarray)
#             images: Tablica z obrazami.
#             hu_moments: Tablica z momentami Hu.
#             labels: Tablica z etykietami klas.

# split_train_test_data(self, random_state=42):

#     Opis: Funkcja dzieli dane na zestawy treningowe i testowe oraz zapisuje je do plików .npy.
#     Parametry:
#         random_state (int): Losowy seed dla podziału danych (domyślnie 42).
#     Zwraca:
#         tuple: (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
#             X_train: Obrazy treningowe.
#             X_test: Obrazy testowe.
#             hu_train: Momentu Hu dla zestawu treningowego.
#             hu_test: Momentu Hu dla zestawu testowego.
#             y_train: Etykiety dla zestawu treningowego.
#             y_test: Etykiety dla zestawu testowego.

# log_hu_moments(self, hu_moments, labels, output_file):

#     Opis: Zapisuje momenty Hu dla każdej klasy do pliku tekstowego.
#     Parametry:
#         hu_moments (ndarray): Tablica z momentami Hu.
#         labels (ndarray): Etykiety klas.
#         output_file (str): Ścieżka do pliku wyjściowego.