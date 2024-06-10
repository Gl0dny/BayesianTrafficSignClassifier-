import numpy as np
from sklearn.metrics import classification_report

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

def train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test, bins=10):
    hnb = HistogramBayesClassifier(bins=bins)
    hnb.fit(hu_train, y_train)
    y_pred = hnb.predict(hu_test)
    print("Histogram Bayes Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    data_dir = 'data/GTSRB/Traffic_Signs/'
    hu_train = np.load(f'{data_dir}hu_train.npy')
    hu_test = np.load(f'{data_dir}hu_test.npy')
    y_train = np.load(f'{data_dir}y_train.npy')
    y_test = np.load(f'{data_dir}y_test.npy')

    train_and_evaluate_histogram_nb(hu_train, y_train, hu_test, y_test)
