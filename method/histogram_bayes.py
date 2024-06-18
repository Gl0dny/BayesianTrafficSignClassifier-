import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class HistogramBayesClassifier:
    """
    Klasyfikator Bayesa z wykorzystaniem histogramów do modelowania rozkładów cech.
    """
    
    def __init__(self, bins, X_train, y_train, X_test, y_test):
        """
        Inicjalizuje klasyfikator z określoną liczbą przedziałów (binów) dla histogramów.

        Parameters:
        - bins (int): Liczba przedziałów dla histogramów.
        - X_train (numpy.ndarray): Tablica z cechami treningowymi.
        - y_train (numpy.ndarray): Tablica z etykietami klas treningowych.
        - X_test (numpy.ndarray): Tablica z cechami testowymi.
        - y_test (numpy.ndarray): Tablica z etykietami klas testowych.
        """
        self.bins = bins
        self.histograms = {}
        self.classes = np.unique(y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        """
        Trenuje klasyfikator na podstawie danych treningowych, obliczając histogramy dla każdej klasy i cechy.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Brak danych treningowych. Ustaw dane treningowe przy inicjalizacji klasy.")
        
        self.classes = np.unique(self.y_train)
        for cls in self.classes:
            self.histograms[cls] = []
            for feature in range(self.X_train.shape[1]):
                hist, bin_edges = np.histogram(self.X_train[self.y_train == cls, feature], bins=self.bins)
                self.histograms[cls].append((hist, bin_edges))

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
                class_probs = self._calculate_class_probabilities(x)
                predicted_class = max(class_probs, key=class_probs.get)
                y_pred.append(predicted_class)
                f.write(f'Sample {i}: {x}\nPredicted class: {predicted_class}\nClass probabilities: {class_probs}\n\n')
        return np.array(y_pred)
    
    def _calculate_class_probabilities(self, x):
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
    
    def print_classification_report(self, y_pred):
        """
        Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.

        Parameters:
        - y_pred (numpy.ndarray): Przewidywane etykiety klas.
        """
        print("Histogram Bayes Classification Report:")
        print(classification_report(self.y_test, y_pred))
    
    def print_histograms_for_class(self, cls):
        """
        Drukuje histogramy dla wszystkich cech dla określonej klasy.

        Parameters:
        - cls (int): Klasa, dla której mają być wyświetlone histogramy.
        """
        if cls not in self.histograms:
            raise ValueError(f"Class {cls} not found in the trained model.")
        
        hists = self.histograms[cls]
        num_features = len(hists)
        
        fig, axes = plt.subplots(num_features, 1, figsize=(12, num_features * 4))

        if num_features == 1:
            axes = [axes]

        for i, (hist, bin_edges) in enumerate(hists):
            ax = axes[i]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
            ax.set_title(f'Class {cls} - Feature {i}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            
            # Adding the values on top of the bars
            for j in range(len(hist)):
                ax.text(bin_centers[j], hist[j], str(hist[j]), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
