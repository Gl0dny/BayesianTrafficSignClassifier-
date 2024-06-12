import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def show_sample_images(images, labels, hu_moments, num_samples=10):
    """
    Wyświetla próbki obrazów z odpowiadającymi im momentami Hu.

    Parameters:
    - images (ndarray): Tablica z obrazami.
    - labels (ndarray): Tablica z etykietami klas.
    - hu_moments (ndarray): Tablica z momentami Hu.
    - num_samples (int): Liczba próbek do wyświetlenia (domyślnie 10).
    """
    plt.figure(figsize=(15, 6))  # Ustawienie rozmiaru figury, aby zmieścić więcej informacji
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)  # Pierwszy rząd: obrazy
        plt.imshow(images[i], cmap='gray')  # Wyświetlenie obrazu w skali szarości
        plt.title(f'Label: {labels[i]}')  # Dodanie tytułu z etykietą klasy
        plt.axis('off')  # Wyłączenie osi dla lepszej czytelności
        
        plt.subplot(2, num_samples, num_samples + i + 1)  # Drugi rząd: momenty Hu
        hu_text = '\n'.join([f'Hu {j+1}: {moment:.2e}' for j, moment in enumerate(hu_moments[i])])  # Formatowanie momentów Hu z numerami
        plt.text(0.5, 0.5, hu_text, ha='center', va='center', wrap=True)  # Wyświetlenie momentów Hu w centrum
        plt.axis('off')  # Wyłączenie osi dla lepszej czytelności
        
    plt.suptitle('Przykładowe próbki i ich momenty Hu')  # Dodanie tytułu wykresu
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Zarezerwowanie miejsca na tytuł
    plt.show()  # Wyświetlenie wykresu

if __name__ == '__main__':
    # Sprawdzenie, czy podano odpowiednią liczbę argumentów wiersza poleceń
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

# Dokumentacja:
# Funkcja show_sample_images(images, labels, hu_moments, num_samples=10):

#     Opis: Wyświetla próbki obrazów z odpowiadającymi im momentami Hu.
#     Parametry:
#         images (ndarray): Tablica z obrazami.
#         labels (ndarray): Tablica z etykietami klas.
#         hu_moments (ndarray): Tablica z momentami Hu.
#         num_samples (int): Liczba próbek do wyświetlenia (domyślnie 10).
#     Opis działania:
#         Funkcja ustawia rozmiar figury, aby zmieścić więcej informacji.
#         W pierwszym rzędzie wyświetlane są obrazy w skali szarości z tytułem zawierającym etykietę klasy.
#         W drugim rzędzie wyświetlane są odpowiednie momenty Hu w formacie tekstowym, wraz z ich numerami.
#         Dodany jest tytuł wykresu "Przykładowe próbki i ich momenty Hu".

# Sekcja if __name__ == '__main__'::

#     Opis: Sekcja główna skryptu, która jest wykonywana, gdy skrypt jest uruchamiany bezpośrednio.
#     Działanie:
#         Sprawdza, czy podano odpowiednią liczbę argumentów wiersza poleceń.
#         Pobiera ścieżkę do katalogu z danymi z argumentu wiersza poleceń.
#         Wczytuje przetworzone dane (obrazy, etykiety, momenty Hu) z plików .npy.
#         Wywołuje funkcję show_sample_images, aby wyświetlić próbki obrazów z zestawu treningowego wraz z odpowiadającymi im momentami Hu.
