import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def show_sample_images(images, labels, hu_moments, num_samples=10):
    """
    Wyświetla próbki obrazów z odpowiadającymi im momentami Hu.

    Parameters:
    - images: Tablica z obrazami.
    - labels: Tablica z etykietami klas.
    - hu_moments: Tablica z momentami Hu.
    - num_samples: Liczba próbek do wyświetlenia (domyślnie 10).
    """
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
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    
    # Pobieranie ścieżki do katalogu z danymi z argumentu wiersza poleceń
    data_dir = sys.argv[1]

    # Wczytywanie przetworzonych danych
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    hu_train = np.load(os.path.join(data_dir, 'hu_train.npy'))

    # Wyświetlanie przykładowych obrazów z zestawu treningowego wraz z momentami Hu
    show_sample_images(X_train, y_train, hu_train)

# Opis funkcji:
# show_sample_images(images, labels, hu_moments, num_samples=10)

# Wyświetla próbki obrazów z odpowiadającymi im momentami Hu w formacie logarytmicznym.

#     Parameters:
#         images (ndarray): Tablica z obrazami.
#         labels (ndarray): Tablica z etykietami klas.
#         hu_moments (ndarray): Tablica z momentami Hu.
#         num_samples (int): Liczba próbek do wyświetlenia (domyślnie 10).

# Szczegóły implementacji:

#     Funkcja plt.figure(figsize=(10, 4)) ustawia wielkość figury, zwiększając jej wysokość, aby zmieścić dwa rzędy podwyświetla.
#     Pierwszy rząd (pierwsze plt.subplot) wyświetla obrazy w skali szarości wraz z etykietami.
#     Drugi rząd (drugie plt.subplot) wyświetla znormalizowane momenty Hu w formacie logarytmicznym dla każdego obrazu.
#     Moment Hu jest sformatowany do notacji wykładniczej z dwoma miejscami po przecinku (f'{moment:.2e}'), co jest odpowiednie do wyświetlania małych wartości.
#     plt.axis('off') wyłącza osie dla lepszego wyglądu wyświetlania.

# Sposób użycia:

#     Wczytaj przetworzone dane (obrazy, etykiety, momenty Hu) z plików .npy.
#     Wywołaj funkcję show_sample_images, aby wyświetlić próbki obrazów z zestawu treningowego wraz z odpowiadającymi im momentami Hu.