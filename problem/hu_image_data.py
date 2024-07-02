import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

class HuImageData:
    def __init__(self, data_dir, no_classes, no_features=7, test_size=0.2):
        """
        Initializes the HuImageData class with specified parameters.

        Parameters:
        - data_dir (str): Path to the GTSRB data directory.
        - no_classes (int): Number of traffic sign classes.
        - no_features (int): Number of features to use from Hu moments (default is 7).
        - test_size (float): Fraction of data to be used as the test set (default is 0.2).
        """
        self.data_dir = data_dir
        self.no_classes = no_classes
        self.no_features = no_features
        self.test_size = test_size

    def _normalize_hu_moments(self, hu_moments):
        """
        Normalizes Hu moments using a logarithmic scale.

        Parameters:
        - hu_moments (ndarray): Array of Hu moments.

        Returns:
        - ndarray: Normalized Hu moments.
        """
        for i in range(len(hu_moments)):
            for j in range(len(hu_moments[i])):
                hu_moments[i][j] = -np.sign(hu_moments[i][j]) * np.log10(np.abs(hu_moments[i][j]))
        return hu_moments

    def _extract_hu_moments_image_data(self):
        """
        Loads and processes the GTSRB data, calculating Hu moments for each image.

        Returns:
        - tuple: (ndarray, ndarray, ndarray)
            - images: Array of images.
            - hu_moments: Array of Hu moments.
            - labels: Array of class labels.
        """
        images = []
        hu_moments = []
        labels = []

        for class_id in range(self.no_classes):
            class_dir = os.path.join(self.data_dir, 'train', str(class_id))
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                try:
                    image = Image.open(os.path.join(class_dir, img_name))
                    image = image.convert('L')
                    image = image.resize((64, 64))
                    image_array = np.array(image)

                    moments = cv2.moments(image_array)
                    hu_moments_image = cv2.HuMoments(moments).flatten()

                    images.append(image_array)
                    hu_moments.append(hu_moments_image[:self.no_features])
                    labels.append(class_id)
                except Exception as e:
                    print("Error processing image:", e)

        hu_moments = self._normalize_hu_moments(np.array(hu_moments))

        return np.array(images), hu_moments, np.array(labels)

    def split_train_test_data(self, random_state=42):
        """
        Splits the data into training and test sets and saves them to .npy files.

        Parameters:
        - random_state (int): Random seed for data splitting (default is 42).

        Returns:
        - tuple: (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
            - X_train: Training images.
            - X_test: Test images.
            - hu_train: Hu moments for the training set.
            - hu_test: Hu moments for the test set.
            - y_train: Labels for the training set.
            - y_test: Labels for the test set.
        """
        if os.path.exists(os.path.join(self.data_dir, 'hu_train.npy')):
            print("Data has already been processed. Loading from numpy files...")
            X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
            hu_train = np.load(os.path.join(self.data_dir, 'hu_train.npy'))
            hu_test = np.load(os.path.join(self.data_dir, 'hu_test.npy'))
            y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(self.data_dir, 'y_test.npy'))
        else:
            images, hu_moments, labels = self._extract_hu_moments_image_data()
            print(f'Loaded {len(images)} images with {len(labels)} labels.')

            print(images.shape, labels.shape)
            print(hu_moments.shape)

            X_train, X_test, hu_train, hu_test, y_train, y_test = train_test_split(
                images, hu_moments, labels, test_size=self.test_size, random_state=random_state
            )
            print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

            np.save(os.path.join(self.data_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(self.data_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(self.data_dir, 'hu_train.npy'), hu_train)
            np.save(os.path.join(self.data_dir, 'hu_test.npy'), hu_test)
            np.save(os.path.join(self.data_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.data_dir, 'y_test.npy'), y_test)

        return X_train, X_test, hu_train, hu_test, y_train, y_test

    def log_hu_moments(self, hu_moments, labels, output_file):
        """
        Saves Hu moments for each class to a text file.

        Parameters:
        - hu_moments (ndarray): Array of Hu moments.
        - labels (ndarray): Class labels.
        - output_file (str): Path to the output file.
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
