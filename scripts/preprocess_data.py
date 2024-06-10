import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocess_gtsrb_data(data_dir):
    images = []
    labels = []
    classes = 43

    for class_id in range(classes):  # GTSRB ma 43 klasy znaków drogowych
        class_dir = os.path.join(data_dir, 'train', str(class_id))
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            try:   
                image = Image.open(class_dir + '/'+ img_name)
                # Convert the image to grayscale
                image = image.convert('L')
                image = image.resize((32,32))
                image_array = np.array(image)
                images.append(image_array)
                labels.append(class_id)
            except Exception as e:
                print("Error processing image:", e)

    return np.array(images), np.array(labels)

if __name__ == '__main__':
    # Ścieżka do katalogu z danymi
    data_dir = 'data/GTSRB/Traffic_Signs/'

    # Wczytywanie danych i przetwarzanie danych
    images, labels = load_preprocess_gtsrb_data(data_dir)
    print(f'Loaded {len(images)} images with {len(labels)} labels.')

    # Checking data shape
    print(images.shape, labels.shape)

    # Podział na zestaw treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

    # Zapisywanie przetworzonych danych (opcjonalnie)
    np.save('data/GTSRB/Traffic_Signs/X_train.npy', X_train)
    np.save('data/GTSRB/Traffic_Signs/X_test.npy', X_test)
    np.save('data/GTSRB/Traffic_Signs/y_train.npy', y_train)
    np.save('data/GTSRB/Traffic_Signs/y_test.npy', y_test)

