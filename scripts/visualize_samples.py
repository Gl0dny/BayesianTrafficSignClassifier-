import numpy as np
import matplotlib.pyplot as plt

def show_sample_images(images, labels, hu_moments, num_samples=10):
    plt.figure(figsize=(10, 4))  # Zwiększ wysokość figury, aby zmieścić momenty Hu
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)  # Pierwszy rząd: obrazy
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
        
        plt.subplot(2, num_samples, num_samples + i + 1)  # Drugi rząd: momenty Hu
        hu_text = '\n'.join([f'{moment:.2e}' for moment in hu_moments[i]])  # Formatowanie momentów Hu
        plt.text(0.5, 0.5, hu_text, ha='center', va='center', wrap=True)
        plt.axis('off')
        
    plt.show()

if __name__ == '__main__':
    # Wczytywanie przetworzonych danych
    X_train = np.load('data/GTSRB/Traffic_Signs/X_train.npy')
    y_train = np.load('data/GTSRB/Traffic_Signs/y_train.npy')
    hu_train = np.load('data/GTSRB/Traffic_Signs/hu_train.npy')

    # Wyświetlanie przykładowych obrazów z zestawu treningowego wraz z momentami Hu
    show_sample_images(X_train, y_train, hu_train)
