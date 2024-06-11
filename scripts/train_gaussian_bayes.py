import sys
import numpy as np
from sklearn.metrics import classification_report

def load_data(data_dir):
    """
    Funkcja ładuje przetworzone dane Hu z plików .npy.
    
    Parameters:
    - data_dir (str): Ścieżka do katalogu z danymi.

    Returns:
    - hu_train (numpy.ndarray): Moment Hu dla danych treningowych.
    - hu_test (numpy.ndarray): Moment Hu dla danych testowych.
    - y_train (numpy.ndarray): Etykiety klas dla danych treningowych.
    - y_test (numpy.ndarray): Etykiety klas dla danych testowych.
    """
    hu_train = np.load(f'{data_dir}hu_train.npy')
    hu_test = np.load(f'{data_dir}hu_test.npy')
    y_train = np.load(f'{data_dir}y_train.npy')
    y_test = np.load(f'{data_dir}y_test.npy')
    return hu_train, hu_test, y_train, y_test

class MaximumLikelihoodBayesClassifier:
    def __init__(self):
        """
        Konstruktor klasy MaximumLikelihoodBayesClassifier.
        Inicjalizuje struktury do przechowywania priorytetów klas, średnich oraz wariancji.
        """
        self.class_prior = {}
        self.mean = {}
        self.variance = {}
    
    def fit(self, X, y):
        """
        Trenuje klasyfikator parametryczny Bayesa przy użyciu metody maksymalnego prawdopodobieństwa.
        
        Parameters:
        - X (numpy.ndarray): Tablica z cechami treningowymi.
        - y (numpy.ndarray): Tablica z etykietami klas treningowych.
        """
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
    
    def _calculate_likelihood(self, class_idx, x):
        """
        Oblicza prawdopodobieństwo warunkowe dla danej klasy i przykładu.
        
        Parameters:
        - class_idx (int): Indeks klasy.
        - x (numpy.ndarray): Pojedynczy przykład.

        Returns:
        - numpy.ndarray: Prawdopodobieństwo warunkowe dla każdej cechy.
        """
        mean = self.mean[class_idx]
        var = self.variance[class_idx]
        numerator = np.exp(- ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _calculate_posterior(self, x):
        """
        Oblicza prawdopodobieństwo a posteriori dla każdej klasy i wybiera klasę z najwyższym prawdopodobieństwem.
        
        Parameters:
        - x (numpy.ndarray): Pojedynczy przykład.

        Returns:
        - int: Klasa o najwyższym prawdopodobieństwie a posteriori.
        """
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_prior[c])
            conditional = np.sum(np.log(self._calculate_likelihood(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Przewiduje klasy dla zbioru danych.
        
        Parameters:
        - X (numpy.ndarray): Tablica z cechami testowymi.

        Returns:
        - numpy.ndarray: Przewidywane etykiety klas dla zbioru testowego.
        """
        y_pred = [self._calculate_posterior(x) for x in X]
        return np.array(y_pred)

def train_and_evaluate_bayes(hu_train, y_train, hu_test, y_test):
    """
    Trenuje i ocenia klasyfikator parametryczny Bayesa.
    
    Parameters:
    - hu_train (numpy.ndarray): Moment Hu dla danych treningowych.
    - y_train (numpy.ndarray): Etykiety klas dla danych treningowych.
    - hu_test (numpy.ndarray): Moment Hu dla danych testowych.
    - y_test (numpy.ndarray): Etykiety klas dla danych testowych.
    """
    classifier = MaximumLikelihoodBayesClassifier()
    classifier.fit(hu_train, y_train)
    y_pred = classifier.predict(hu_test)
    print("Maximum Likelihood Bayes Classifier Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    
    # Ścieżka do katalogu z danymi
    data_dir = sys.argv[1]
    
    # Ładowanie danych
    hu_train, hu_test, y_train, y_test = load_data(data_dir)
    
    # Trenowanie i ewaluacja klasyfikatora Bayesa
    train_and_evaluate_bayes(hu_train, y_train, hu_test, y_test)


# Opis implementacji:

#     load_data: Funkcja ładuje przetworzone dane Hu oraz etykiety klas z plików .npy. Zwraca cztery tablice: hu_train, hu_test, y_train, y_test.

#     MaximumLikelihoodBayesClassifier: Klasa klasyfikatora Bayesa.
#         __init__: Inicjalizuje puste struktury do przechowywania priorytetów klas, średnich oraz wariancji.
#         fit: Trenuje klasyfikator, obliczając priorytety klas, średnie oraz wariancje dla każdej klasy.
#         _calculate_likelihood: Oblicza prawdopodobieństwo warunkowe dla danej klasy i przykładu.
#         _calculate_posterior: Oblicza prawdopodobieństwo a posteriori dla każdej klasy i wybiera klasę z najwyższym prawdopodobieństwem.
#         predict: Przewiduje klasy dla zbioru danych testowych.

#     train_and_evaluate_bayes: Funkcja trenuje i ocenia klasyfikator Bayesa. Wyświetla raport klasyfikacji.

#     Główna część skryptu: Ładuje dane, trenuje klasyfikator oraz ocenia jego wydajność.

