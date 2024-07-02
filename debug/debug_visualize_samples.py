import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def show_sample_images(images, labels, hu_moments, num_samples=10):
    """
    Displays sample images with corresponding Hu moments.

    Parameters:
    - images (ndarray): Array of images.
    - labels (ndarray): Array of class labels.
    - hu_moments (ndarray): Array of Hu moments.
    - num_samples (int): Number of samples to display (default is 10).
    """
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
        
        plt.subplot(2, num_samples, num_samples + i + 1)
        hu_text = '\n'.join([f'Hu {j+1}: {moment:.2e}' for j, moment in enumerate(hu_moments[i])])
        plt.text(0.5, 0.5, hu_text, ha='center', va='center', wrap=True)
        plt.axis('off')
        
    plt.suptitle('Sample Images and their Hu Moments')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    hu_train = np.load(os.path.join(data_dir, 'hu_train.npy'))

    show_sample_images(X_train, y_train, hu_train)
