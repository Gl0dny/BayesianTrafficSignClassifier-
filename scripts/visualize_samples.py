import numpy as np
import matplotlib.pyplot as plt

def show_sample_images(images, labels, num_samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # plt.imshow(images[i].reshape(32, 32), cmap='gray')
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Wczytywanie przetworzonych danych
    X_train = np.load('data/GTSRB/Traffic_Signs/X_train.npy')
    y_train = np.load('data/GTSRB/Traffic_Signs/y_train.npy')

    # Wyświetlanie przykładowych obrazów z zestawu treningowego
    show_sample_images(X_train, y_train)
