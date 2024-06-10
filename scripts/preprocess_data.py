import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

def load_preprocess_gtsrb_data(data_dir):
    images = []
    hu_moments = []
    labels = []
    classes = 43

    for class_id in range(classes):  # GTSRB ma 43 klasy znaków drogowych
        class_dir = os.path.join(data_dir, 'train', str(class_id))
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            try:
                image = Image.open(os.path.join(class_dir, img_name))
                # Convert the image to grayscale
                image = image.convert('L')
                image = image.resize((32, 32))
                image_array = np.array(image)

                # Compute Hu Moments
                moments = cv2.moments(image_array)
                hu_moments_image = cv2.HuMoments(moments).flatten()

                images.append(image_array)
                hu_moments.append(hu_moments_image)
                labels.append(class_id)
            except Exception as e:
                print("Error processing image:", e)

    return np.array(images), np.array(hu_moments), np.array(labels)

def split_train_test_data(data, test_size=0.2, random_state=42):
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

        # Checking data shape
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

if __name__ == '__main__':
    # Ścieżka do katalogu z danymi
    data_dir = 'data/GTSRB/Traffic_Signs/'

    # Podział na zestaw treningowy i testowy
    X_train, X_test, hu_train, hu_test, y_train, y_test = split_train_test_data(
        data=data_dir, test_size=0.2, random_state=42
    )

    print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')
    print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
    print("Data preprocessing complete.")