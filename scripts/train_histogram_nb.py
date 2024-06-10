import numpy as np
from sklearn.metrics import classification_report
import os

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
        with open(log_file, 'w') as f:
            for cls, hists in self.histograms.items():
                f.write(f'Class {cls} histograms:\n')
                for i, (hist, bin_edges) in enumerate(hists):
                    f.write(f'Feature {i}: histogram: {hist}, bin_edges: {bin_edges}\n')
                f.write('\n')

    def predict(self, X, predict_log_file):
        y_pred = []
        with open(predict_log_file, 'w') as f:
            for x in X:
                class_probs = self.calculate_class_probabilities(x)
                predicted_class = max(class_probs, key=class_probs.get)
                y_pred.append(predicted_class)
                f.write(f'Sample: {x}\nPredicted class: {predicted_class}\nClass probabilities: {class_probs}\n\n')
        return np.array(y_pred)

    def calculate_class_probabilities(self, x):
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
