import numpy as np
from sklearn.metrics import classification_report
import os

class HistogramBayesClassifier:
    """
    Klasyfikator Bayesa z wykorzystaniem histogramów do modelowania rozkładów cech.
    """
    
    def __init__(self, bins=10):
        """
        Inicjalizuje klasyfikator z określoną liczbą przedziałów (binów) dla histogramów.

        Parameters:
        - bins (int): Liczba przedziałów dla histogramów.
        """
        self.bins = bins
        self.histograms = {}
        self.classes = None

    def fit(self, X, y, train_log_file):
        """
        Trenuje klasyfikator na podstawie danych treningowych, obliczając histogramy dla każdej klasy i cechy.
        
        Parameters:
        - X (numpy.ndarray): Tablica z cechami treningowymi.
        - y (numpy.ndarray): Tablica z etykietami klas treningowych.
        - train_log_file (str): Ścieżka do pliku, w którym będą zapisywane histogramy.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            self.histograms[cls] = []
            for feature in range(X.shape[1]):
                hist, bin_edges = np.histogram(X[y == cls, feature], bins=self.bins)
                self.histograms[cls].append((hist, bin_edges))
        self.log_histograms(train_log_file)

    def log_histograms(self, log_file):
        """
        Zapisuje histogramy do pliku tekstowego.

        Parameters:
        - log_file (str): Ścieżka do pliku, w którym będą zapisywane histogramy.
        """
        with open(log_file, 'w') as f:
            for cls, hists in self.histograms.items():
                f.write(f'Class {cls} histograms:\n')
                for i, (hist, bin_edges) in enumerate(hists):
                    f.write(f'Feature {i}: histogram: {hist}, bin_edges: {bin_edges}\n')
                f.write('\n')

    def predict(self, X, predict_log_file):
        """
        Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.

        Parameters:
        - X (numpy.ndarray): Tablica z cechami testowymi.
        - predict_log_file (str): Ścieżka do pliku, w którym będą zapisywane szczegółowe informacje o predykcji.

        Returns:
        - numpy.ndarray: Przewidywane etykiety klas dla zbioru testowego.
        """
        y_pred = []
        with open(predict_log_file, 'w') as f:
            for x in X:
                class_probs = self.calculate_class_probabilities(x)
                predicted_class = max(class_probs, key=class_probs.get)
                y_pred.append(predicted_class)
                f.write(f'Sample: {x}\nPredicted class: {predicted_class}\nClass probabilities: {class_probs}\n\n')
        return np.array(y_pred)

    def calculate_class_probabilities(self, x):
        """
        Oblicza prawdopodobieństwa klas dla pojedynczego przykładu na podstawie histogramów.

        Parameters:
        - x (numpy.ndarray): Pojedynczy przykład.

        Returns:
        - dict: Słownik z klasami i ich odpowiadającymi prawdopodobieństwami.
        """
        class_probs = {}
        for cls in self.classes:
            class_prob = 1.0
            for feature in range(len(x)):
                hist, bin_edges = self.histograms[cls][feature]
                bin_index = np.digitize(x[feature], bin_edges) - 1
                bin_index = min(max(bin_index, 0), len(hist) - 1)
                prob = hist[bin_index] / np.sum(hist)
                class_prob *= prob
            class_probs[cls] = class_prob
        return class_probs

if __name__ == '__main__':
    # Załaduj dane treningowe i testowe z plików .npy
    data_dir = 'data/GTSRB/Traffic_Signs/'
    hu_train = np.load(os.path.join(data_dir, 'hu_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    hu_test = np.load(os.path.join(data_dir, 'hu_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Trenowanie klasyfikatora Histogram Bayes
    bins = 20
    h_classifier = HistogramBayesClassifier(bins=bins)
    h_classifier.fit(hu_train, y_train, train_log_file='logs/train_histograms.txt')

    # Predykcja i ocena modelu
    y_pred = h_classifier.predict(hu_test, predict_log_file='logs/predict_probs.txt')
    print("Histogram Bayes Classification Report:")
    print(classification_report(y_test, y_pred))

# Opis implementacji:

#     HistogramBayesClassifier: Klasa klasyfikatora Bayesa z wykorzystaniem histogramów.
#         __init__: Inicjalizuje klasę z określoną liczbą przedziałów dla histogramów.
#         fit: Trenuje klasyfikator, obliczając histogramy dla każdej klasy i cechy.
#         log_histograms: Zapisuje histogramy do pliku tekstowego.
#         predict: Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.
#         calculate_class_probabilities: Oblicza prawdopodobieństwa klas dla pojedynczego przykładu na podstawie histogramów.

#     Główna część skryptu:
#         load_data: Funkcja ładuje dane Hu oraz etykiety klas z plików .npy.
#         Trenowanie klasyfikatora Histogram Bayes: Tworzy instancję klasyfikatora, trenuje go na danych treningowych i zapisuje histogramy do pliku.
#         Predykcja i ocena modelu: Przewiduje klasy dla danych testowych, zapisuje szczegółowe informacje o predykcji do pliku i drukuje raport klasyfikacji.

# Zawartość plików logów:

#     train_histograms.txt: Plik zawiera histogramy dla każdej klasy i cechy, obliczone na danych treningowych.
#     predict_probs.txt: Plik zawiera szczegółowe informacje o predykcji dla każdej próbki testowej, w tym przewidywaną klasę oraz prawdopodobieństwa klas.

