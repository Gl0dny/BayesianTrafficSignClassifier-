import os
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class HistogramBayesClassifier:
    def __init__(self, bins=10):
        self.bins = bins
        self.histograms = {}
        self.classes = None

    def fit(self, X, y, train_log_file):
        self.classes = np.unique(y)
        for cls in self.classes:
            self.histograms[cls] = []
            for feature in range(X.shape[1]):
                hist, bin_edges = np.histogram(X[y == cls, feature], bins=self.bins)
                self.histograms[cls].append((hist, bin_edges))
        self.log_histograms(train_log_file)

    def log_histograms(self, log_file):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

        with open(log_file, 'w') as f:
            for cls in self.classes:
                f.write(f'Class {cls} Histograms:\n')
                for feature_idx, (hist, bin_edges) in enumerate(self.histograms[cls], start=1):
                    f.write(f'  Feature {feature_idx}:\n')
                    f.write(f'    Histogram: {hist}\n')
                    f.write(f'    Bin edges: {bin_edges}\n')
                f.write('\n')

    def log_probs(self, probs, log_file):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

        with open(log_file, 'w') as f:
            for idx, prob_list in enumerate(probs):
                f.write(f'Sample {idx + 1} Probabilities:\n')
                for cls, prob in zip(self.classes, prob_list):
                    f.write(f'  Class {cls}: {prob}\n')
                f.write('\n')

    def predict(self, X, predict_log_file):
        log_probs = []  # Lista do przechowywania prawdopodobieństw dla każdej próbki
        predictions = []
        for x in X:
            class_probs = []
            for cls in self.classes:
                prob = 1.0
                for feature in range(len(x)):
                    hist, bin_edges = self.histograms[cls][feature]
                    bin_index = np.digitize(x[feature], bin_edges) - 1
                    bin_index = min(max(bin_index, 0), len(hist) - 1)
                    prob *= hist[bin_index]
                class_probs.append(prob)
            log_probs.append(class_probs)  # Dodajemy prawdopodobieństwa dla danej próbki do listy log_probs
            predictions.append(self.classes[np.argmax(class_probs)])
        
        # Zapisujemy log do pliku
        self.log_probs(log_probs, predict_log_file)
        
        return np.array(predictions)

def train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test, bins=5):
    h_classifier = HistogramBayesClassifier(bins=bins)
    h_classifier.fit(hu_train, y_train, train_log_file='train_histograms.txt')
    y_pred = h_classifier.predict(hu_test, predict_log_file='predict_probs.txt')
    print("Histogram Bayes Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    data_dir = 'data/GTSRB/Traffic_Signs/'
    hu_train = np.load(f'{data_dir}hu_train.npy')
    hu_test = np.load(f'{data_dir}hu_test.npy')
    y_train = np.load(f'{data_dir}y_train.npy')
    y_test = np.load(f'{data_dir}y_test.npy')

    train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test)
