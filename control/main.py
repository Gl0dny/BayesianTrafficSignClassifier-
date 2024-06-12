#!/usr/bin/env python3

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from control import Logger
from problem import GTSRB
from problem import HuImageData
from method import GaussianBayesClassifier
from method import HistogramBayesClassifier

log_dir = 'debug/logs'
log_file = os.path.join(log_dir, 'progress_log.txt')

def main(bin_count, data_dir, zip_path, debug, no_classes, no_features, test_size):
    """
    Główna funkcja uruchamiająca proces przetwarzania danych i trenowania modeli klasyfikacyjnych.

    Parameters:
    - bin_count (int): Liczba koszyków dla modelu histogramowego.
    - data_dir (str): Ścieżka do katalogu z danymi.
    - zip_path (str): Ścieżka do pliku zip z danymi GTSRB.
    - debug (bool): Flaga włączająca tryb debugowania.
    - no_classes (int): Liczba klas znaków drogowych.
    - no_features (int): Liczba cech do użycia z momentów Hu.
    - test_size (float): Ułamek danych przeznaczonych na zestaw testowy.
    """
    logger = Logger(log_file)

    # Krok 1: Rozpakowanie danych
    logger.log("Step 1: Extracting GTSRB data started.")
    gtsrb = GTSRB(data_dir, zip_path)
    gtsrb.extract()

    # Krok 2: Przetwarzanie danych
    logger.log("Step 2: Preprocessing data started.")
    hu_image_data = HuImageData(data_dir, no_classes, no_features, test_size)
    X_train, X_test, hu_train, hu_test, y_train, y_test = hu_image_data.split_train_test_data()
    hu_image_data.log_hu_moments(hu_train, y_train, os.path.join(log_dir, 'hu_moments_log.txt'))
    print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
    print("Data preprocessing complete. Hu moments logged to", log_file)

    # Krok (opcjonalny): Wizualizacja przykładowych danych
    if debug:
        logger.log("Optional Step: Visualizing sample data started.")
        logger.run_script('debug/debug_visualize_samples.py', args=[data_dir])

    # Krok 3: Uczenie parametrycznego klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego)
    logger.log("Step 3: Training Gaussian Bayes model started.")
    g_classifier = GaussianBayesClassifier(X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
    g_classifier.fit()

    # Krok 4: Uczenie nieparametrycznego klasyfikatora Bayesa (histogram wielowymiarowy)
    logger.log("Step 4: Training Histogram Bayes model started.")
    h_classifier = HistogramBayesClassifier(bins=bin_count, X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
    h_classifier.fit()
    h_classifier.log_histograms(log_file=os.path.join(log_dir, 'train_histograms.txt'))
    
    # Krok (opcjonalny): Wizualizacja histogramów danej klasy
    if debug:
        logger.log("Optional Step: Visualizing sample histogram started.")
        h_classifier.print_histograms_for_class(1)

    # Krok 5: Klasyfikacja - Uruchomienie parametrycznego klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego) na zbiorze testowym
    y_pred = g_classifier.predict(predict_log_file=os.path.join(log_dir, 'g_classifier_predict_probs.txt'))
    g_classifier.print_classification_report(y_pred)
    
    # Krok 6: Klasyfikacja - Uruchomienie nieparametrycznego klasyfikatora Bayesa (histogram wielowymiarowy) na zbiorze testowym
    y_pred = h_classifier.predict(predict_log_file=os.path.join(log_dir, 'h_classifier_predict_probs.txt'))
    h_classifier.print_classification_report(y_pred)

if __name__ == '__main__':
    # Parser argumentów
    parser = argparse.ArgumentParser(description="Run the data processing and training pipeline.")
    parser.add_argument('--data_dir', type=str, default='problem/data/GTSRB/Traffic_Signs/', help='Directory containing the data scripts.')
    parser.add_argument('--zip_path', type=str, default='problem/data/GTSRB/gtsrb.zip', help='Path to the GTSRB zip file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to visualize sample data.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used for testing (between 0.01 and 0.99).')
    parser.add_argument('--no_classes', type=int, default=8, help='Number of classes.')
    parser.add_argument('--no_features', type=int, default=7, help='Number of features (Hu moments) to use (between 1 and 7).')
    parser.add_argument('--bin_count', type=int, default=10, help='Number of bins for histogram model.')
    args = parser.parse_args()

    # Walidacja argumentów
    if not (2 < args.no_classes):
        raise ValueError("The number of classes must be at least 2.")
    if not (0.01 <= args.test_size <= 0.99):
        raise ValueError("test_size must be between 0.01 and 0.99")
    if not (1 <= args.no_features <= 7):
        raise ValueError("no_features must be between 1 and 7")
    
    # Logowanie procesu
    logger = Logger(log_file)
    logger.log("Process started.")
    main(args.bin_count, args.data_dir, args.zip_path, args.debug, args.no_classes, args.no_features, args.test_size)
    logger.log("Process completed.")

# Dokumentacja:
# Funkcja main:

#     Opis: Główna funkcja uruchamiająca proces przetwarzania danych i trenowania modeli klasyfikacyjnych.
#     Parametry:
#         bin_count (int): Liczba koszyków dla modelu histogramowego.
#         data_dir (str): Ścieżka do katalogu z danymi.
#         zip_path (str): Ścieżka do pliku zip z danymi GTSRB.
#         debug (bool): Flaga włączająca tryb debugowania.
#         no_classes (int): Liczba klas znaków drogowych.
#         no_features (int): Liczba cech do użycia z momentów Hu.
#         test_size (float): Ułamek danych przeznaczonych na zestaw testowy.

# Sekcja if __name__ == '__main__':

#     Opis: Parser argumentów i walidacja wejściowych parametrów. Uruchamia główną funkcję przetwarzania danych i trenowania modeli.
#     Argumenty:
#         --data_dir: Katalog zawierający dane.
#         --zip_path: Ścieżka do pliku zip z danymi GTSRB.
#         --debug: Włącza tryb debugowania do wizualizacji danych.
#         --test_size: Ułamek danych do zestawu testowego (między 0.01 a 0.99).
#         --no_classes: Liczba klas.
#         --no_features: Liczba cech (momentów Hu) do użycia (między 1 a 7).
#         --bin_count: Liczba koszyków dla modelu histogramowego.