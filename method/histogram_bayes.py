import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class HistogramBayesClassifier:
    """
    Bayes Classifier using histograms to model feature distributions.
    """
    
    def __init__(self, bins, X_train, y_train, X_test, y_test):
        """
        Initializes the classifier with a specified number of bins for the histograms.

        Parameters:
        - bins (int): Number of bins for the histograms.
        - X_train (numpy.ndarray): Array with training features.
        - y_train (numpy.ndarray): Array with training class labels.
        - X_test (numpy.ndarray): Array with testing features.
        - y_test (numpy.ndarray): Array with testing class labels.
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
        Trains the classifier based on the training data by calculating histograms for each class and feature.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data. Set the training data during class initialization.")
        
        self.classes = np.unique(self.y_train)
        for cls in self.classes:
            self.histograms[cls] = []
            for feature in range(self.X_train.shape[1]):
                hist, bin_edges = np.histogram(self.X_train[self.y_train == cls, feature], bins=self.bins)
                self.histograms[cls].append((hist, bin_edges))

    def log_histograms(self, log_file):
        """
        Saves histograms to a text file.

        Parameters:
        - log_file (str): Path to the file where histograms will be saved.
        """
        with open(log_file, 'w') as f:
            for cls, hists in self.histograms.items():
                f.write(f'Class {cls} histograms:\n')
                for i, (hist, bin_edges) in enumerate(hists):
                    f.write(f'Feature {i}: histogram: {hist}, bin_edges: {bin_edges}\n')
                f.write('\n')
    
    def predict(self, predict_log_file):
        """
        Predicts classes for the test data and saves detailed prediction information to a file.

        Parameters:
        - predict_log_file (str): Path to the file where detailed prediction information will be saved.

        Returns:
        - numpy.ndarray: Predicted class labels for the test set.
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
        Calculates class probabilities for a single example based on the histograms.

        Parameters:
        - x (numpy.ndarray): Single example.

        Returns:
        - dict: Dictionary with classes and their corresponding probabilities.
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
        Prints the classification report based on the test data and predictions.

        Parameters:
        - y_pred (numpy.ndarray): Predicted class labels.
        """
        print("Histogram Bayes Classification Report:")
        print(classification_report(self.y_test, y_pred))
    
    def print_histograms_for_class(self, cls):
        """
        Prints histograms for all features for a specified class.

        Parameters:
        - cls (int): Class for which histograms should be displayed.
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
