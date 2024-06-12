import numpy as np
from sklearn.metrics import classification_report

class GaussianBayesClassifier:
    """
    Klasyfikator Bayesa z wykorzystaniem rozkładów Gaussa do modelowania rozkładów cech.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Inicjalizuje klasyfikator na podstawie danych treningowych i testowych.

        Parameters:
        - X_train (numpy.ndarray): Tablica z cechami treningowymi.
        - y_train (numpy.ndarray): Tablica z etykietami klas treningowych.
        - X_test (numpy.ndarray): Tablica z cechami testowymi.
        - y_test (numpy.ndarray): Tablica z etykietami klas testowych.
        """
        self.classes = np.unique(y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.mean = {}
        self.variance = {}
        self.class_prior = {}

    def fit(self):
        """
        Trenuje klasyfikator na podstawie danych treningowych, obliczając średnie, wariancje i priorytety klas.
        """
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.class_prior[c] = len(X_c) / len(self.X_train)

    def predict(self, predict_log_file):
        """
        Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.

        Parameters:
        - predict_log_file (str): Ścieżka do pliku, w którym będą zapisywane szczegółowe informacje o predykcji.

        Returns:
        - numpy.ndarray: Przewidywane etykiety klas dla zbioru testowego.
        """
        y_pred = []
        with open(predict_log_file, 'w') as f:
            for i, x in enumerate(self.X_test):
                predicted_class = self._calculate_posterior(x)
                y_pred.append(predicted_class)
                class_probs = {cls: self._calculate_posterior(x) for cls in self.classes}
                f.write(f'Sample {i}: {x}\nPredicted class: {predicted_class}\nClass probabilities: {class_probs}\n\n')
        return np.array(y_pred)

    def print_classification_report(self, y_pred):
        """
        Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.

        Parameters:
        - y_pred (numpy.ndarray): Przewidywane etykiety klas.
        """
        print("Gaussian Bayes Classification Report:")
        print(classification_report(self.y_test, y_pred))

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

# Dokumentacja:
# Klasa GaussianBayesClassifier:

#     Opis: Klasyfikator Bayesa z wykorzystaniem rozkładów Gaussa do modelowania rozkładów cech.
#     Metody:
#         __init__(self, X_train, y_train, X_test, y_test): Inicjalizuje klasyfikator na podstawie danych treningowych i testowych.
#         fit(self): Trenuje klasyfikator na podstawie danych treningowych, obliczając średnie, wariancje i priorytety klas.
#         predict(self, predict_log_file): Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.
#         print_classification_report(self, y_pred): Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.
#         _calculate_posterior(self, x): Oblicza prawdopodobieństwo a posteriori dla każdej klasy i wybiera klasę z najwyższym prawdopodobieństwem.
#         _calculate_likelihood(self, class_idx, x): Oblicza prawdopodobieństwo warunkowe dla danej klasy i przykładu.

# Funkcja fit:

#     Opis: Trenuje klasyfikator na podstawie danych treningowych, obliczając średnie, wariancje i priorytety klas.

# Funkcja predict:

#     Opis: Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.
#     Parametry:
#         predict_log_file (str): Ścieżka do pliku, w którym będą zapisywane szczegółowe informacje o predykcji.
#     Zwraca:
#         numpy.ndarray: Przewidywane etykiety klas dla zbioru testowego.

# Funkcja print_classification_report:

#     Opis: Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.
#     Parametry:
#         y_pred (numpy.ndarray): Przewidywane etykiety klas.

# Funkcja _calculate_posterior:

#     Opis: Oblicza prawdopodobieństwo a posteriori dla każdej klasy i wybiera klasę z najwyższym prawdopodobieństwem.
#     Parametry:
#         x (numpy.ndarray): Pojedynczy przykład.
#     Zwraca:
#         int: Klasa o najwyższym prawdopodobieństwie a posteriori.

# Funkcja _calculate_likelihood:

#     Opis: Oblicza prawdopodobieństwo warunkowe dla danej klasy i przykładu.
#     Parametry:
#         class_idx (int): Indeks klasy.
#         x (numpy.ndarray): Pojedynczy przykład.
#     Zwraca:
#         numpy.ndarray: Prawdopodobieństwo warunkowe dla każdej cechy.