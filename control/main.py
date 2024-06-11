#!/usr/bin/env python3

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from problem import GTSRB
from problem import HuImageData
from logger_utils import Logger

log_dir = 'debug/logs'
log_file = os.path.join(log_dir, 'progress_log.txt')

def main(bin_count, data_dir, zip_path, debug):
    logger = Logger(log_file)

    # Krok 1: Rozpakowanie danych
    logger.log("Step 1: Extracting GTSRB data started.")
    gtsrb=GTSRB(data_dir,zip_path)
    gtsrb.extract()

    # Krok 2: Przetwarzanie danych
    logger.log("Step 2: Preprocessing data started.")
    # run_script('problem/preprocess_data.py', args=[data_dir])
    hu_image_data=HuImageData(data_dir, 8)
    X_train, X_test, hu_train, hu_test, y_train, y_test = hu_image_data.split_train_test_data()
    hu_image_data.log_hu_moments(hu_train, y_train, os.path.join(log_dir, 'hu_moments_log.txt'))
    print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
    print("Data preprocessing complete. Hu moments logged to", log_file)
    print("Data preprocessing complete.")


    # # Krok (opcjonalny): Wizualizacja przykładowych danych
    # if debug:
    #     log("Optional Step: Visualizing sample data started.")
    #     run_script('debug/debug_visualize_samples.py', args=[data_dir])

    # # Krok 3: Uczenie parametrycznego klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego)
    # log("Step 3: Training Gaussian Naive Bayes model started.")
    # run_script('method/train_gaussian_bayes.py', args=[data_dir])

    # # Krok 4: Uczenie nieparametrycznego klasyfikatora Bayesa (histogram wielowymiarowy)
    # log("Step 4: Training Histogram Bayes model started.")
    # run_script('method/train_histogram_bayes.py', args=[data_dir, str(bin_count)])

if __name__ == '__main__':
    # Parser argumentów
    parser = argparse.ArgumentParser(description="Run the data processing and training pipeline.")
    parser.add_argument('--bin_count', type=int, default=20, help='Number of bins for histogram model.')
    parser.add_argument('--data_dir', type=str, default='problem/data/GTSRB/Traffic_Signs/', help='Directory containing the data scripts.')
    parser.add_argument('--zip_path', type=str, default='problem/data/GTSRB/gtsrb.zip', help='Path to the GTSRB zip file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to visualize sample data.')
    args = parser.parse_args()

    logger=Logger(log_file)
    logger.log("Process started.")
    main(args.bin_count, args.data_dir, args.zip_path, args.debug)
    logger.log("Process completed.")
