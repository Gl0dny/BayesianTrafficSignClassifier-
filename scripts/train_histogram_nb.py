import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class HistogramBayesClassifier:
    def __init__(self, bins=10):
        self.bins = bins
        self.histograms = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            self.histograms[cls] = []
            for feature in range(X.shape[1]):
                hist, bin_edges = np.histogram(X[y == cls, feature], bins=self.bins, density=True)
                self.histograms[cls].append((hist, bin_edges))

    def predict(self, X):
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
            predictions.append(self.classes[np.argmax(class_probs)])
        return np.array(predictions)

def train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test, bins=2):
    h_classifier = HistogramBayesClassifier(bins=bins)
    h_classifier.fit(hu_train, y_train)
    y_pred = h_classifier.predict(hu_test)
    print("Histogram Bayes Classification Report:")
    print(classification_report(y_test, y_pred))
    # Przykładowe użycie:
    # Załóżmy, że masz już wytrenowany klasyfikator hnb i chcesz zobaczyć histogram dla klasy 0 i cechy 0.
    visualize_histograms(h_classifier, hu_train, y_train, class_idx=0)

def visualize_histograms(classifier, X, y, class_idx, num_features=7):
    # Ustawienie układu subplotów
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(18, 4))

    # Iteracja po cechach
    for feature_idx in range(1, num_features + 1):  # Rozpoczynamy od 1, nie od 0
        # Wybierz histogram dla danej klasy i cechy
        hist, bin_edges = classifier.histograms[class_idx][feature_idx - 1]  # Odwołujemy się do indeksu - 1

        # Ustal wartości osi x dla środków kubków
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Wykreśl histogram
        axes[feature_idx - 1].bar(bin_centers, hist, width=0.7*(bin_centers[1] - bin_centers[0]))

        # Dodaj tytuł dla subplotu
        axes[feature_idx - 1].set_title(f'Moment Hu {feature_idx}')

        # Dodaj etykietę oznaczającą, że to jest moment Hu
        axes[feature_idx - 1].set_xlabel('Value')
        axes[feature_idx - 1].set_ylabel('Density')

    # Wyświetl wykresy
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_dir = 'data/GTSRB/Traffic_Signs/'
    hu_train = np.load(f'{data_dir}hu_train.npy')
    hu_test = np.load(f'{data_dir}hu_test.npy')
    y_train = np.load(f'{data_dir}y_train.npy')
    y_test = np.load(f'{data_dir}y_test.npy')

    train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test)