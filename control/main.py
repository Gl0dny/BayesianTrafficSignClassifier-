#!/usr/bin/env python3

import sys
import os
import argparse
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from control import Logger
from problem import GTSRB, HuImageData
from method import GaussianBayesClassifier, HistogramBayesClassifier

def clean_pipeline_data():
    """
    Function to clean unnecessary directories.
    """
    directories_to_remove = [
        'debug/logs',
        'problem/data/GTSRB/Traffic_Signs/',
    ]
    for directory in directories_to_remove:
        if os.path.exists(directory):
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            else:
                os.remove(directory)
            print(f"Removing {directory}")

def main(bin_count, data_dir, zip_path, debug, no_classes, no_features, test_size):
    """
    Main function to start the data processing and model training pipeline.

    Parameters:
    - bin_count (int): Number of bins for the histogram model.
    - data_dir (str): Path to the data directory.
    - zip_path (str): Path to the GTSRB zip file.
    - debug (bool): Flag to enable debug mode.
    - no_classes (int): Number of traffic sign classes.
    - no_features (int): Number of features to use from Hu moments.
    - test_size (float): Fraction of data to be used for the test set.
    """
    
    logger.log("Step 1: Extracting GTSRB data started.")
    gtsrb = GTSRB(data_dir, zip_path)
    gtsrb.extract()

    logger.log("Step 2: Preprocessing data started.")
    hu_image_data = HuImageData(data_dir, no_classes, no_features, test_size)
    X_train, X_test, hu_train, hu_test, y_train, y_test = hu_image_data.split_train_test_data()
    hu_image_data.log_hu_moments(hu_train, y_train, os.path.join(log_dir, 'hu_moments.log'))
    print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
    print("Data preprocessing complete. Hu moments logged to", log_file)

    if debug:
        logger.log("Optional Step: Visualizing sample data started.")
        logger.run_script('debug/debug_visualize_samples.py', args=[data_dir])

    logger.log("Step 3: Training Gaussian Bayes model started.")
    g_classifier = GaussianBayesClassifier(X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
    g_classifier.fit()

    logger.log("Step 4: Training Histogram Bayes model started.")
    h_classifier = HistogramBayesClassifier(bins=bin_count, X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
    h_classifier.fit()
    h_classifier.log_histograms(log_file=os.path.join(log_dir, 'histograms.log'))
    
    if debug:
        logger.log("Optional Step: Visualizing sample histogram started.")
        h_classifier.print_histograms_for_class(1)

    y_pred = g_classifier.predict(predict_log_file=os.path.join(log_dir, 'g_classifier_predict_predictions.log'))
    g_classifier.print_classification_report(y_pred)
    
    y_pred = h_classifier.predict(predict_log_file=os.path.join(log_dir, 'h_classifier_predict_predictions.log'))
    h_classifier.print_classification_report(y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the data processing and training pipeline.")
    parser.add_argument('--data_dir', type=str, default='problem/data/GTSRB/Traffic_Signs/', help='Directory containing the data.')
    parser.add_argument('--zip_path', type=str, default='problem/data/GTSRB/gtsrb.zip', help='Path to the GTSRB zip file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to visualize sample data.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used for testing (between 0.01 and 0.99).')
    parser.add_argument('--no_classes', type=int, default=5, help='Number of classes.')
    parser.add_argument('--no_features', type=int, default=7, help='Number of features (Hu moments) to use (between 1 and 7).')
    parser.add_argument('--bin_count', type=int, default=10, help='Number of bins for the histogram model.')
    parser.add_argument('--clean', action='store_true', help='Optionally clean unnecessary directories before starting.')

    args = parser.parse_args()

    if not (2 < args.no_classes):
        raise ValueError("The number of classes must be at least 2.")
    if not (0.01 <= args.test_size <= 0.99):
        raise ValueError("test_size must be between 0.01 and 0.99")
    if not (1 <= args.no_features <= 7):
        raise ValueError("no_features must be between 1 and 7")
    
    if args.clean:
        print("Cleaning previously processed files.")
        clean_pipeline_data()

    log_dir = 'debug/logs'
    log_file = os.path.join(log_dir, 'main.log')
    logger = Logger(log_file)

    logger.log("Process started.")
    main(args.bin_count, args.data_dir, args.zip_path, args.debug, args.no_classes, args.no_features, args.test_size)
    logger.log("Process completed.")
