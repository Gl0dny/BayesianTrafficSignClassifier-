import numpy as np
from sklearn.metrics import classification_report

class GaussianBayesClassifier:
    """
    Bayes classifier using Gaussian distributions to model feature distributions.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the classifier based on training and test data.

        Parameters:
        - X_train (numpy.ndarray): Array with training features.
        - y_train (numpy.ndarray): Array with training class labels.
        - X_test (numpy.ndarray): Array with test features.
        - y_test (numpy.ndarray): Array with test class labels.
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
        Trains the classifier based on training data by calculating class means, variances, and priors.
        """
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.class_prior[c] = len(X_c) / len(self.X_train)

    def predict(self, predict_log_file):
        """
        Predicts classes for test data and logs detailed prediction information to a file.

        Parameters:
        - predict_log_file (str): Path to the file where detailed prediction information will be logged.

        Returns:
        - numpy.ndarray: Predicted class labels for the test set.
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
        Prints the classification report based on test data and predictions.

        Parameters:
        - y_pred (numpy.ndarray): Predicted class labels.
        """
        print("Gaussian Bayes Classification Report:")
        print(classification_report(self.y_test, y_pred))

    def _calculate_posterior(self, x):
        """
        Calculates the posterior probability for each class and selects the class with the highest probability.

        Parameters:
        - x (numpy.ndarray): Single example.

        Returns:
        - int: Class with the highest posterior probability.
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
        Calculates the conditional probability for a given class and example.

        Parameters:
        - class_idx (int): Class index.
        - x (numpy.ndarray): Single example.

        Returns:
        - numpy.ndarray: Conditional probability for each feature.
        """
        mean = self.mean[class_idx]
        var = self.variance[class_idx]
        numerator = np.exp(- ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
