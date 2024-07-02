# Traffic Sign Classification Project Using Bayes Classifier

The purpose of this program is to process data from the German Traffic Sign Recognition Benchmark (GTSRB) and train two classification models: Gaussian Bayes and Histogram Bayes. The models are evaluated and their performance is logged.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation and Environment Setup](#installation-and-environment-setup)
  - [Simulator](#simulator)
  - [Micromouse](#micromouse)
  - [Maze](#maze)
  - [Logger](#logger)
  - [Sensor](#sensor)
  - [CommandQueue](#commandqueue)
  - [Main Function](#main-function)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Compilation and Execution](#compilation-and-execution)

## Project Structure

    BayesianTrafficSignClassifier
    │
    ├── control
    │   ├── __init__.py
    │   ├── logger_utils.py
    │   └── main.py
    │
    ├── debug
    │   └── debug_visualize_samples.py
    │
    ├── method
    │   ├── __init__.py
    │   ├── gaussian_bayes.py
    │   └── histogram_bayes.py
    │
    ├── problem
    │   ├── __init__.py
    │   ├── gtsrb.py
    │   ├── hu_image_data.py
    │   └── data
    │       └── GTSRB
    │           └── gtsrb.zip
    │
    ├── setup
    │   ├── requirements.txt
    │   ├── setup.bat
    │   └── setup.sh
    │
    ├── .gitattributes
    ├── .gitignore
    └── README.md

Below is a brief description of the main directories and files:

- **control**: Contains the main control scripts of the project.
  - `__init__.py`: Initialization file for the control module.
  - `logger_utils.py`: Helper functions for logging, setting log formatting, log levels, and mechanisms for saving logs to files or displaying them on the console.
  - `main.py`: Main script for running the classifier. Includes logic for initialization, data loading, model training, and evaluation of results.

- **debug**: Contains scripts for debugging.
  - `debug_visualize_samples.py`: Script for visualizing data samples for debugging purposes. Helps understand the input data and verify the correctness of data processing.

- **method**: Contains the implementation of Bayes methods.
  - `__init__.py`: Initialization file for the method module.
  - `gaussian_bayes.py`: Contains the implementation of the parametric ML Bayes classifier (assuming normal distribution). This classifier uses the assumption that features follow a normal distribution to calculate class probabilities.
  - `histogram_bayes.py`: Contains the implementation of the non-parametric Bayes classifier (multidimensional histogram). This classifier is based on histograms of features, using feature distributions to calculate class probabilities.

- **problem**: Contains files and data specific to the problem.
  - `__init__.py`: Initialization file for the problem module.
  - `gtsrb.py`: Methods for handling GTSRB data.
  - `hu_image_data.py`: Methods for handling image data with Hu moments, extracting Hu moments from traffic sign images, and splitting the data into training and test sets.
  - **data/GTSRB**: Directory containing the GTSRB (German Traffic Sign Recognition Benchmark) data. This data is used to train and test the classification models.
    - `gtsrb.zip`: Compressed file with the GTSRB dataset.

- **setup**: Contains installation scripts and the requirements file.
  - `requirements.txt`: List of dependencies required for the project.
  - `setup.bat`: Batch script for Windows to install dependencies and set up the environment.
  - `setup.sh`: Shell script for Unix/Linux to install dependencies and set up the environment.


## Installation and Environment Setup

### Required Modules
The `requirements.txt` file contains all the required packages and their versions necessary to run the project.

To handle the setup of a virtual environment along with the installation of the necessary modules, use the `setup.bat` or `setup.sh` scripts.

### Windows

1. Run `setup/setup.bat`.
2. The script will check if a virtual environment exists, and if not, it will create a new one.
3. The virtual environment will be activated.
4. A list of installed packages will be displayed before and after the installation of new packages.
5. Packages from `requirements.txt` will be installed.
6. Instructions for running the main script and deactivating the virtual environment will be displayed.

### Unix/Linux

1. Run `setup/setup.sh`.
2. The script will check if a virtual environment exists, and if not, it will create a new one using:
    ```
    python3 -m venv
    ```
3. The virtual environment will be activated via:
    ```
    source venv/bin/activate.
    ```
4. A list of installed packages will be displayed before and after the installation of new packages.
5. Packages from `requirements.txt` will be installed.
6. Instructions for running the main script and deactivating the virtual environment will be displayed.

Alternatively, you can manually install the required libraries using pip:
```
pip install -r setup/requirements.txt
```

## Running the Project

Run the main script, which will execute all the project steps:
```
python control/main.py
```

### control/main.py

The main script initiates the data processing and model training process.

#### Parameters:

- `bin_count` (int): Number of bins for the histogram model.
- `data_dir` (str): Path to the data directory.
- `zip_path` (str): Path to the zip file containing the data in the format train/class_dir.
- `debug` (bool): Flag to enable debugging mode.
- `no_classes` (int): Number of traffic sign classes. (default: 5 - due to the project's size limit of 20 MB)
- `no_features` (int): Number of features to use from Hu moments. (default: 7 - number of Hu moments)
- `test_size` (float): Fraction of data reserved for the test set. (default: 0.2)
- `bin_count` (int): Number of bins for the histogram model. (default: 10)
- `clean` (bool): Flag to enable cleaning mode.

#### Arguments:

- `--data_dir`: Directory containing the data.
- `--zip_path`: Path to the zip file with compressed data in the format train/class_dir.
- `--debug`: Enables debugging mode for data visualization.
- `--test_size`: Fraction of data for the test set (between 0.01 and 0.99).
- `--no_classes`: Number of classes. (dependent on the amount of available data/classes)
- `--no_features`: Number of features (Hu moments) to use (between 1 and 7).
- `--bin_count`: Number of bins for the histogram model.
- `--clean`: Enables cleaning mode for files generated during data processing.

Here is an example of running with arguments:
```
python control/main.py --clean --debug --test_size 0.2 --no_classes 5 --no_features 7 --bin_count 10
```

## Detailed Program Operation

The program processes the data and trains two classification models, monitoring the entire process through logging. Below is a detailed description of each step.

### Step 1: Extracting Data

```
logger.log("Step 1: Extracting GTSRB data started.")
gtsrb = GTSRB(data_dir, zip_path)
gtsrb.extract()
```

**Description:** Extracting data from the zip file containing the German Traffic Sign Recognition Benchmark (GTSRB) resources

**Actions:**

- Initialize the GTSRB object with the data directory path and the zip file path.
- Extract the zip file to the specified directory.

**Logging:** Record the start of the data extraction operation.

### Step 2: Data Processing

```
logger.log("Step 2: Preprocessing data started.")
hu_image_data = HuImageData(data_dir, no_classes, no_features, test_size)
X_train, X_test, hu_train, hu_test, y_train, y_test = hu_image_data.split_train_test_data()
hu_image_data.log_hu_moments(hu_train, y_train, os.path.join(log_dir, 'hu_moments_log.txt'))
print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
print("Data preprocessing complete. Hu moments logged to", log_file)
```

**Description:** Processing data, including calculating Hu moments, splitting into training and test sets, and logging results.

**Actions:**

- Initialize the HuImageData object with data from the directory, number of classes, number of features, and test set size.
- Split the data into training and test sets and calculate Hu moments.
- Log the Hu moments for the training set.
- Display information about the sizes of the training and test sets.

**Logging:** Record the start of data processing and log Hu moments to the `hu_moments_log.txt` file.

### Optional Step: Visualizing Sample Data

```
if debug:
    logger.log("Optional Step: Visualizing sample data started.")
    logger.run_script('debug/debug_visualize_samples.py', args=[data_dir])
```

**Description:** Visualize sample data (optional, depending on debugging mode).

**Actions:**

- Record the start of data visualization.
- Run the debug/debug_visualize_samples.py script to visualize data samples.

**Logging**: Record the start of the optional visualization step.

### Step 3: Training Gaussian Bayes Classifier (assuming normal distribution)

```
logger.log("Step 3: Training Gaussian Bayes model started.")
g_classifier = GaussianBayesClassifier(X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
g_classifier.fit()
```

**Description:** Train the Gaussian Bayes classifier assuming normal distribution.

**Actions:**

- Initialize the GaussianBayesClassifier object with training and test data.
- Train the model on the training set.

**Logowanie:** Rejestrowanie rozpoczęcia treningu klasyfikatora Gaussian Bayes.

### Krok 4: Uczenie Nieparametrycznego Klasyfikatora Bayesa (histogram wielowymiarowy)

```
logger.log("Step 4: Training Histogram Bayes model started.")
h_classifier = HistogramBayesClassifier(bins=bin_count, X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
h_classifier.fit()
h_classifier.log_histograms(log_file=os.path.join(log_dir, 'train_histograms.txt'))
```

**Opis:** Trening nieparametrycznego klasyfikatora Bayesa z wykorzystaniem histogramów.

**Czynności:**
- Inicjalizacja obiektu HistogramBayesClassifier z danymi treningowymi i testowymi oraz liczbą binów.
- Trening modelu na zbiorze treningowym.

**Logowanie:** Rejestrowanie rozpoczęcia treningu klasyfikatora Histogram Bayes oraz zapis histogramów.

### Krok (opcjonalny): Wizualizacja Histogramów Danej Klasy

```
if debug:
    logger.log("Optional Step: Visualizing sample histogram started.")
    h_classifier.print_histograms_for_class(1)
```

**Opis:** Wizualizacja histogramów dla wybranej klasy (opcjonalne, w zależności od trybu debugowania).

**Czynności:**
- Rejestrowanie rozpoczęcia wizualizacji histogramów.
- Wyświetlenie histogramów dla klasy 1.

**Logowanie:** Rejestrowanie rozpoczęcia opcjonalnego kroku wizualizacji histogramów.

### Krok 5: Klasyfikacja - Parametryczny Klasyfikator Bayesa ML

```
y_pred = g_classifier.predict(predict_log_file=os.path.join(log_dir, 'g_classifier_predict_predictions.txt'))
g_classifier.print_classification_report(y_pred)
```

**Opis:** Przewidywanie klas na zbiorze testowym za pomocą parametrycznego klasyfikatora Bayesa.

**Czynności:**
- Przewidywanie klas dla zbioru testowego i zapis wyników do pliku g_classifier_predict_predictions.txt.
- Wyświetlenie raportu z klasyfikacji.

**Logowanie:** Zapis wyników predykcji i raportu z klasyfikacji.

### Krok 6: Klasyfikacja - Nieparametryczny Klasyfikator Bayesa

**Opis:** Przewidywanie klas na zbiorze testowym za pomocą nieparametrycznego klasyfikatora Bayesa.

**Czynności:**
- Przewidywanie klas dla zbioru testowego i zapis wyników do pliku h_classifier_predict_predictions.txt.
- Wyświetlenie raportu z klasyfikacji.

**Logowanie:** Zapis wyników predykcji i raportu z klasyfikacji.

## Podsumowanie

Program składa się z kilku kroków, od rozpakowania danych, przez przetwarzanie i uczenie modeli klasyfikacyjnych, aż po przewidywanie klas na zbiorze testowym. Każdy krok jest starannie logowany, co umożliwia śledzenie postępu i diagnozowanie ewentualnych problemów. Dodatkowe kroki wizualizacji mogą być wykonywane w trybie debugowania, co pomaga w analizie i zrozumieniu danych.


# Plik Logów Monitorujący Postęp: `main.log`

## Opis

`main.log` jest głównym plikiem logów, który monitoruje postęp całego procesu przetwarzania danych, trenowania modeli oraz inne istotne operacje wykonywane przez program. Jest to kluczowy plik do śledzenia, ponieważ zawiera chronologiczny zapis wszystkich ważnych kroków i ewentualnych błędów, które wystąpiły podczas działania programu.

## Przykład Zawartości `main.log`

```
2024-06-13 11:45:12 - Process started.
2024-06-13 11:45:12 - Step 1: Extracting GTSRB data started.
Extracting GTSRB dataset to problem/data/GTSRB/Traffic_Signs/ folder...
Extraction complete.
2024-06-13 11:45:14 - Step 2: Preprocessing data started.
Loaded 2879 images with 2879 labels.
(2879, 64, 64) (2879,)
(2879, 7)
Train set size: 2303, Test set size: 576
Train Hu moments size: 2303, Test Hu moments size: 576
Data preprocessing complete. Hu moments logged to debug/logs\main.log
2024-06-13 11:45:42 - Step 3: Training Gaussian Bayes model started.
2024-06-13 11:45:42 - Step 4: Training Histogram Bayes model started.

Histogram Bayes Classification Report:
              precision    recall  f1-score   support

           0       0.47      0.68      0.56       142
           1       0.54      0.60      0.57       181
           2       0.19      0.27      0.22        48
           3       0.22      0.13      0.16        60
           4       0.44      0.19      0.27       145

    accuracy                           0.44       576
   macro avg       0.37      0.38      0.36       576
weighted avg       0.43      0.44      0.42       576

2024-06-13 11:45:42 - Process completed.
```

# Raport klasyfikacji - przykład

    Gaussian Bayes Classification Report:
                precision    recall  f1-score   support

            0       0.51      0.68      0.58       142
            1       0.51      0.71      0.59       181
            2       0.28      0.31      0.30        48
            3       0.37      0.22      0.27        60
            4       0.35      0.11      0.17       145

        accuracy                           0.47       576
    macro avg       0.40      0.41      0.38       576
    weighted avg       0.44      0.47      0.43       576


## Ogólna wydajność modelu
- **Accuracy (dokładność):** 0.47
  - Dokładność oznacza odsetek poprawnie sklasyfikowanych próbek spośród wszystkich próbek. W tym przypadku model prawidłowo sklasyfikował 47% wszystkich próbek.

## Wydajność dla poszczególnych klas

### Klasa 0:
- **Precision (precyzja):** 0.51
  - Precyzja to stosunek liczby prawdziwie pozytywnych wyników do sumy prawdziwie pozytywnych i fałszywie pozytywnych wyników. Oznacza to, że z wszystkich próbek sklasyfikowanych jako klasa 0, 51% było poprawnie sklasyfikowanych.
- **Recall (czułość):** 0.68
  - Czułość to stosunek liczby prawdziwie pozytywnych wyników do sumy prawdziwie pozytywnych i fałszywie negatywnych wyników. Oznacza to, że z wszystkich prawdziwych próbek klasy 0, 68% zostało prawidłowo sklasyfikowanych jako klasa 0.
- **F1-score:** 0.58
  - F1-score to średnia harmoniczna precyzji i czułości. Jest to miara ogólnej wydajności modelu dla tej klasy.
- **Support:** 142
  - Support to liczba rzeczywistych wystąpień danej klasy w zbiorze danych testowych.

# Pliki Logów

## control/logger_utils.py

Plik `logger_utils.py` zawiera klasy Tee i Logger, które zarządzają zapisywaniem komunikatów logów do plików oraz ich wyświetlaniem w terminalu.

## Generowane Pliki Logów

### `g_classifier_predictions.log`

**Opis:**
Plik zawiera szczegółowe informacje dotyczące predykcji dokonanych przez klasyfikator Gaussian Bayes.

**Zawartość:**
- Szczegóły predykcji dla każdej próbki w zestawie testowym.
- Prawdopodobieństwa a posteriori dla każdej klasy.
- Prawdopodobieństwa warunkowe dla każdej cechy i klasy.

**Format:**
Każda linia zawiera informacje dla jednej próbki:
- Identyfikator próbki, prawdziwa klasa, przewidywana klasa, prawdopodobieństwa a posteriori, prawdopodobieństwa warunkowe.

### `h_classifier_predictions.log`

**Opis:**
Plik zawiera szczegółowe informacje dotyczące predykcji dokonanych przez klasyfikator Histogram Bayes.

**Zawartość:**
- Szczegóły predykcji dla każdej próbki w zestawie testowym.
- Prawdopodobieństwa klas dla każdej próbki na podstawie histogramów.

**Format:**
Każda linia zawiera informacje dla jednej próbki:
- Identyfikator próbki, prawdziwa klasa, przewidywana klasa, prawdopodobieństwa klas.

### `hu_moments.log`

**Opis:**
Pliki zawierają obliczone momenty Hu dla próbek należących do określonej klasy.

**Zawartość:**
- Obliczone momenty Hu dla próbek.
- Zapisane momenty Hu dla analizy i debugowania.

**Format:**
Każda linia zawiera momenty Hu dla jednej próbki:
- Identyfikator próbki, wartości momentów Hu.

### `histograms.log`

**Opis:**
Pliki zawierają histogramy dla każdej cechy i klasy.

**Zawartość:**
- Histogramy dla każdej cechy i klasy.
- Zapisane histogramy do analizy i debugowania.

**Format:**
Każda linia zawiera wartości histogramu dla jednej cechy:
- Indeks cechy, wartości histogramu dla tej cechy.

## Przykłady Plików Logów

### Przykład wpisu w `g_classifier_predictions.log`:
```
Sample 0: [  2.16200585   6.99546412   8.97868627   8.7388965   17.76683571
  12.90594896 -17.7310433 ]
Predicted class: 0
Class probabilities: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

Sample 1: [  2.67146131   7.13761203   9.67750212   9.70987065 -19.40559961
  13.92897663 -20.4178654 ]
Predicted class: 3
Class probabilities: {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
```
### Przykład wpisu w `h_classifier_predictions.log`:
```
Sample 0: [  2.16200585   6.99546412   8.97868627   8.7388965   17.76683571
  12.90594896 -17.7310433 ]
Predicted class: 0
Class probabilities: {0: 0.0001115832769177358, 1: 7.402632433244765e-09, 2: 0.0, 3: 1.2183058381188396e-07, 4: 1.914566511006075e-07}

Sample 1: [  2.67146131   7.13761203   9.67750212   9.70987065 -19.40559961
  13.92897663 -20.4178654 ]
Predicted class: 0
Class probabilities: {0: 6.810747944179981e-05, 1: 9.922832670994054e-06, 2: 2.1797952038828098e-05, 3: 2.6278723219514598e-05, 4: 3.2232811716693285e-06}
```
### Przykład wpisu w `hu_moments.log`:
```
Class 0 Hu Moments:
Sample 1 Hu Moments: [  2.43038677   6.09687194   9.14487504   9.12304103 -18.35530063
  12.49486787 -18.47639574]
Sample 2 Hu Moments: [  2.58046413   8.72705975  12.73975729  11.98774375  24.67979376
  16.69358992 -24.40558514]
```
### Przykład wpisu w `histograms.log`:
```
Class 0 histograms:
Feature 0: histogram: [ 6  5 71 99 69 62 74 59 36 43], bin_edges: [1.79549735 1.9165121  2.03752686 2.15854161 2.27955636 2.40057112
 2.52158587 2.64260062 2.76361538 2.88463013 3.00564488]
Feature 1: histogram: [ 40  86 111 140  75  38  21   5   5   3], bin_edges: [ 5.74120241  6.24296646  6.74473052  7.24649458  7.74825864  8.2500227
  8.75178675  9.25355081  9.75531487 10.25707893 10.75884299]
```

## Uwagi

- **Pliki logów są generowane w odpowiednich momentach pracy programu, w zależności od wykonywanych operacji.**
- **Pliki `g_classifier_predictions.log` i `h_classifier_predictions.log` są generowane po wykonaniu predykcji przez odpowiednie klasyfikatory.**
- **Pliki `hu_moments.log` są generowane podczas obliczania momentów Hu.**
- **Pliki `histograms.log` są generowane podczas tworzenia histogramów dla cech i klas.**


## Diagram klas

    Main: Główna klasa odpowiedzialna za inicjalizację projektu, wczytywanie danych, trenowanie modeli i ewaluację wyników.
    LoggerUtils: Klasa zawierająca metody do logowania oraz uruchamiania skryptów pomocniczych.
    GaussianBayesClassifier: Klasa implementująca parametryczny klasyfikator Bayesa przy założeniu rozkładu normalnego.
    HistogramBayesClassifier: Klasa implementująca nieparametryczny klasyfikator Bayesa oparty na histogramach.
    GTSRB: Klasa odpowiedzialna za zarządzanie danymi GTSRB, w tym ich rozpakowywanie.
    HuImageData: Klasa odpowiedzialna za przetwarzanie danych obrazowych, w tym obliczanie momentów Hu i podział danych na zbiory treningowy i testowy.

```mermaid
classDiagram
    class Main {
        -data_dir : str
        -zip_path : str
        -bin_count : int
        -no_classes : int
        -no_features : int
        -test_size : float
        -debug : bool
        +main()
    }

    class LoggerUtils {
        +log(message: str)
        +run_script(script: str, args: list)
    }

    class GaussianBayesClassifier {
        -X_train : array
        -y_train : array
        -X_test : array
        -y_test : array
        +fit()
        +predict() : array
        +print_classification_report(y_pred: array)
    }

    class HistogramBayesClassifier {
        -bins : int
        -X_train : array
        -y_train : array
        -X_test : array
        -y_test : array
        +fit()
        +predict() : array
        +print_classification_report(y_pred: array)
        +log_histograms(log_file: str)
        +print_histograms_for_class(class_index: int)
    }

    class GTSRB {
        -data_dir : str
        -zip_path : str
        +extract()
    }

    class HuImageData {
        -data_dir : str
        -no_classes : int
        -no_features : int
        -test_size : float
        +split_train_test_data() : tuple
        +log_hu_moments(hu_data: array, labels: array, log_file: str)
    }

    Main --> LoggerUtils : uses
    Main --> GaussianBayesClassifier : uses
    Main --> HistogramBayesClassifier : uses
    Main --> GTSRB : uses
    Main --> HuImageData : uses
```

## Opis Głównych Plików

### control/logger_utils.py

Zarządza zapisywaniem komunikatów logów do pliku oraz ich wyświetlaniem w terminalu.

#### Klasa Tee:
- Przechwytuje wyjście i przekierowuje je zarówno do pliku, jak i do terminala.

#### Klasa Logger:
- Zarządza zapisywaniem komunikatów logów do pliku oraz ich wyświetlaniem w terminalu.

### debug/debug_visualize_samples.py

Wyświetla próbki obrazów z odpowiadającymi im momentami Hu w trybie debugowania.

#### Funkcja show_sample_images:
- Wyświetla próbki obrazów z momentami Hu.
- Parametry:
  - `images` (ndarray): Tablica z obrazami.
  - `labels` (ndarray): Tablica z etykietami klas.
  - `hu_moments` (ndarray): Tablica z momentami Hu.
  - `num_samples` (int): Liczba próbek do wyświetlenia (domyślnie 10).

### problem/gtsrb.py

Zarządza wypakowywaniem danych GTSRB (German Traffic Sign Recognition Benchmark) z pliku zip.

#### Klasa GTSRB:
- `__init__`: Inicjalizuje klasę z określonymi ścieżkami do wypakowania i pliku zip.
- `extract`: Wypakowuje plik z danymi GTSRB do określonego folderu.

### problem/hu_image_data.py

Przetwarza obrazy znaków drogowych, oblicza momenty Hu oraz dzieli dane na zestawy treningowe i testowe.

#### Klasa HuImageData:
- `__init__`: Inicjalizuje klasę z określonymi parametrami.
- `_normalize_hu_moments`: Normalizuje momenty Hu, stosując skalę logarytmiczną.
- `_extract_hu_moments_image_data`: Ładuje i przetwarza dane GTSRB, obliczając momenty Hu dla każdego obrazu.
- `split_train_test_data`: Dzieli dane na zestawy treningowe i testowe oraz zapisuje je do plików .npy.
- `log_hu_moments`: Zapisuje momenty Hu dla każdej klasy do pliku tekstowego.

### setup/setup.bat

Skrypt instalacyjny dla systemu Windows.

#### Działanie:
- Sprawdza, czy środowisko wirtualne nie istnieje, a jeśli nie, tworzy nowe środowisko wirtualne.
- Aktywuje środowisko wirtualne.
- Wyświetla listę zainstalowanych pakietów przed i po instalacji nowych pakietów.
- Instaluje wymagania z pliku requirements.txt.
- Wyświetla instrukcje dotyczące uruchamiania głównego skryptu oraz dezaktywacji środowiska wirtualnego.

### setup/setup.sh

Skrypt instalacyjny dla systemów Unix.

#### Działanie:
- Tworzy środowisko wirtualne, jeśli nie istnieje.
- Aktywuje środowisko wirtualne.
- Wyświetla listę zainstalowanych pakietów przed i po instalacji nowych pakietów.
- Instaluje wymagania z pliku requirements.txt.
- Wyświetla instrukcje dotyczące uruchamiania głównego skryptu oraz dezaktywacji środowiska wirtualnego.

### method/gaussian_bayes.py

Klasyfikator Bayesa z wykorzystaniem rozkładów Gaussa do modelowania rozkładów cech.

#### Klasa GaussianBayesClassifier:
- `__init__`: Inicjalizuje klasyfikator na podstawie danych treningowych i testowych.
- `fit`: Trenuje klasyfikator na podstawie danych treningowych.
- `predict`: Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.
- `print_classification_report`: Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.
- `_calculate_posterior`: Oblicza prawdopodobieństwo a posteriori dla każdej klasy.
- `_calculate_likelihood`: Oblicza prawdopodobieństwo warunkowe dla danej klasy i przykładu.

### method/histogram_bayes.py

Klasyfikator Bayesa z wykorzystaniem histogramów do modelowania rozkładów cech.

#### Klasa HistogramBayesClassifier:
- `__init__`: Inicjalizuje klasyfikator z określoną liczbą przedziałów (binów) dla histogramów.
- `fit`: Trenuje klasyfikator na podstawie danych treningowych.
- `log_histograms`: Zapisuje histogramy do pliku tekstowego.
- `predict`: Przewiduje klasy dla danych testowych i zapisuje szczegółowe informacje o predykcji do pliku.
- `print_classification_report`: Drukuje raport klasyfikacji na podstawie danych testowych i przewidywań.
- `_calculate_class_probabilities`: Oblicza prawdopodobieństwa klas dla pojedynczego przykładu na podstawie histogramów.
- `print_histograms_for_class`: Drukuje histogramy dla wszystkich cech dla określonej klasy.
