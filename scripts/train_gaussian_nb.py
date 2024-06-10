import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def load_data(data_dir):
    hu_train = np.load(f'{data_dir}hu_train.npy')
    hu_test = np.load(f'{data_dir}hu_test.npy')
    y_train = np.load(f'{data_dir}y_train.npy')
    y_test = np.load(f'{data_dir}y_test.npy')
    return hu_train, hu_test, y_train, y_test

def train_and_evaluate_gaussian_nb(hu_train, y_train, hu_test, y_test):
    gnb = GaussianNB()
    gnb.fit(hu_train, y_train)
    y_pred = gnb.predict(hu_test)
    print("Gaussian Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    data_dir = 'data/GTSRB/Traffic_Signs/'
    hu_train, hu_test, y_train, y_test = load_data(data_dir)
    train_and_evaluate_gaussian_nb(hu_train, y_train, hu_test, y_test)
