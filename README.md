# Projekt Klasyfikacji Znaków Drogowych z Użyciem Klasyfikatora Bayesa

Ten projekt ma na celu klasyfikację znaków drogowych z wykorzystaniem klasyfikatora Bayesa. Projekt obejmuje przetwarzanie danych, trenowanie modeli klasyfikacyjnych oraz wizualizację wyników.

## Struktura Projektu

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

Poniżej znajduje się krótki opis głównych katalogów i plików:

- **control**: Zawiera główne skrypty kontrolne projektu.
  - `__init__.py`: Plik inicjalizacyjny dla modułu control.
  - `logger_utils.py`: Funkcje pomocnicze do logowania, ustawienia formatowania logów, poziomy logowania oraz mechanizmy zapisu logów do plików lub wyświetlania na konsoli.
  - `main.py`: Główny skrypt do uruchamiania klasyfikatora. Zawiera logikę inicjalizacji, ładowania danych, trenowania modelu i ewaluacji wyników.

- **debug**: Zawiera skrypty do debugowania.
  - `debug_visualize_samples.py`: Skrypt do wizualizacji próbek danych dla celów debugowania. Pomaga w zrozumieniu, jak wyglądają dane wejściowe oraz weryfikacji poprawności przetwarzania danych.

- **method**: Zawiera implementację metod bayesowskich.
  - `__init__.py`: Plik inicjalizacyjny dla modułu method.
  - `gaussian_bayes.py`: Zawiera implementację parametrycznego klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego). Klasyfikator ten wykorzystuje założenie, że cechy mają rozkład normalny, aby obliczyć prawdopodobieństwa przynależności do klas.
  - `histogram_bayes.py`: Zawiera implementację klasyfikatora nieparametrycznego klasyfikatora Bayesa (histogram wielowymiarowy). Klasyfikator oparty na histogramach cech, który wykorzystuje dystrybucje cech do obliczenia prawdopodobieństw przynależności do klas.

- **problem**: Zawiera pliki i dane specyficzne dla problemu.
  - `__init__.py`: Plik inicjalizacyjny dla modułu problem.
  - `gtsrb.py`: Metody do obsługi danych GTSRB.
  - `hu_image_data.py`: Metody do obsługi danych obrazowych z momentami Hu, ekstrakcji momentów Hu z obrazów znaków drogowych oraz podziału danych na zestawy treningowe i testowe.
  - **data/GTSRB**: Katalog zawierający dane GTSRB (German Traffic Sign Recognition Benchmark). Dane te są używane do trenowania i testowania modeli klasyfikacyjnych.
    - `gtsrb.zip`: Skompresowany plik z zestawem danych GTSRB.

- **setup**: Zawiera skrypty instalacyjne i plik z wymaganiami.
  - `requirements.txt`: Lista zależności wymaganych do projektu.
  - `setup.bat`: Skrypt wsadowy dla systemu Windows do instalacji zależności i konfiguracji środowiska.
  - `setup.sh`: Skrypt powłoki dla systemów Unix/Linux do instalacji zależności i konfiguracji środowiska.


## Instalacja i Konfiguracja Środowiska

### Wymagane moduły
Plik `requirements.txt` zawiera wszystkie wymagane pakiety i ich wersje, które są niezbędne do uruchomienia projektu. 

Do obsługi setup'u środowiska wirtualnego wraz z instalacją odpowiednich modułów służdą skrypty `setup.bat` or `setup.sh`.

### Windows

1. Uruchom `setup/setup.bat`.
2. Skrypt sprawdzi, czy środowisko wirtualne istnieje, a jeśli nie, utworzy nowe.
3. Środowisko wirtualne zostanie aktywowane.
4. Lista zainstalowanych pakietów zostanie wyświetlona przed i po instalacji nowych pakietów.
5. Pakiety z `requirements.txt` zostaną zainstalowane.
6. Wyświetlone zostaną instrukcje dotyczące uruchamiania głównego skryptu i dezaktywacji środowiska wirtualnego.

### Unix

1. Uruchom `setup/setup.sh`.
2. Skrypt sprawdzi, czy środowisko wirtualne istnieje, a jeśli nie, utworzy nowe przy użyciu `python3 -m venv`.
3. Środowisko wirtualne zostanie aktywowane poprzez `source venv/bin/activate`.
4. Lista zainstalowanych pakietów zostanie wyświetlona przed i po instalacji nowych pakietów.
5. Pakiety z `requirements.txt` zostaną zainstalowane.
6. Wyświetlone zostaną instrukcje dotyczące uruchamiania głównego skryptu i dezaktywacji środowiska wirtualnego.

Alternatywnie, możesz ręcznie zainstalować wymagane biblioteki używając pip:
```
pip install -r setup/requirements.txt
```

## Uruchomienie Projektu

Uruchom główny skrypt, który przeprowadzi wszystkie kroki projektu:
```
python control/main.py
```

## Argumenty Skryptu main.py

Skrypt main.py może przyjmować różne argumenty konfiguracyjne. 




Dzięki tym instrukcjom, powinieneś być w stanie uruchomić projekt klasyfikacji znaków drogowych przy użyciu klasyfikatora Bayesa oraz zrozumieć strukturę i funkcjonowanie poszczególnych modułów.

### control/main.py

```
python control/main.py
```

Główny plik uruchamiający proces przetwarzania danych i trenowania modeli klasyfikacyjnych.

#### Parametry:
- `bin_count` (int): Liczba koszyków dla modelu histogramowego.
- `data_dir` (str): Ścieżka do katalogu z danymi.
- `zip_path` (str): Ścieżka do pliku zip ze spakowanymi danymi w formacie train/class_dir.
- `debug` (bool): Flaga włączająca tryb debugowania.
- `no_classes` (int): Liczba klas znaków drogowych. ( default: 5 - ze względu na ograniczenie rozmiaru projektu do 20 MB )
- `no_features` (int): Liczba cech do użycia z momentów Hu. ( default: 7 - liczba momentów Hu )
- `test_size` (float): Ułamek danych przeznaczonych na zestaw testowy. ( default: 0.2 )
- `clean` (bool): Flaga włączająca tryb czyszczenia.

#### Argumenty:
- `--data_dir`: Katalog zawierający dane.
- `--zip_path`: Ścieżka do pliku zip ze spakowanymi danymi w formacie train/class_dir.
- `--debug`: Włącza tryb debugowania do wizualizacji danych.
- `--test_size`: Ułamek danych do zestawu testowego (między 0.01 a 0.99).
- `--no_classes`: Liczba klas. ( zależne od ilości posiadanych danych/klas )
- `--no_features`: Liczba cech (momentów Hu) do użycia (między 1 a 7).
- `--bin_count`: Liczba koszyków dla modelu histogramowego.
- `--clean`: Włącza tryb czyszczenia plików generowanych podczas przetwarzania danych.    

Oto przykład uruchomienia z argumentami:
```
python control/main.py --clean --debug --test_size 0.2 --no_classes 5 --no_features 7 --bin_count 10
```

## Szczegółowy Opis Działania Programu

Program przetwarza dane i trenuje dwa modele klasyfikacyjne, monitorując cały proces poprzez logowanie. Poniżej znajduje się szczegółowy opis każdego kroku.

### Krok 1: Rozpakowanie Danych

```
logger.log("Step 1: Extracting GTSRB data started.")
gtsrb = GTSRB(data_dir, zip_path)
gtsrb.extract()
```

**Opis:** Rozpakowanie danych z pliku zip z zasobami German Traffic Sign Recognition Benchmark (GTSRB).

**Czynności:**
- Zainicjowanie obiektu GTSRB z podaniem ścieżki do katalogu z danymi oraz ścieżki do pliku zip.
- Rozpakowanie pliku zip do określonego katalogu.

**Logowanie:** Rejestrowanie rozpoczęcia operacji rozpakowywania danych.

### Krok 2: Przetwarzanie Danych

```
logger.log("Step 2: Preprocessing data started.")
hu_image_data = HuImageData(data_dir, no_classes, no_features, test_size)
X_train, X_test, hu_train, hu_test, y_train, y_test = hu_image_data.split_train_test_data()
hu_image_data.log_hu_moments(hu_train, y_train, os.path.join(log_dir, 'hu_moments_log.txt'))
print(f'Train Hu moments size: {hu_train.shape[0]}, Test Hu moments size: {hu_test.shape[0]}')
print("Data preprocessing complete. Hu moments logged to", log_file)
```

**Opis:** Przetwarzanie danych, w tym obliczanie momentów Hu, podział na zbiory treningowy i testowy oraz logowanie wyników.

**Czynności:**
- Inicjalizacja obiektu HuImageData z danymi z katalogu, liczbą klas, liczbą cech i rozmiarem zbioru testowego.
- Podział danych na zbiory treningowy i testowy oraz obliczenie momentów Hu.
- Logowanie momentów Hu dla zbioru treningowego.
- Wyświetlenie informacji o rozmiarach zbiorów treningowego i testowego.

**Logowanie:** Rejestrowanie rozpoczęcia przetwarzania danych oraz logowanie momentów Hu do pliku hu_moments_log.txt.

### Krok (opcjonalny): Wizualizacja Przykładowych Danych

```
if debug:
    logger.log("Optional Step: Visualizing sample data started.")
    logger.run_script('debug/debug_visualize_samples.py', args=[data_dir])
```

**Opis:** Wizualizacja przykładowych danych (opcjonalne, w zależności od trybu debugowania).

**Czynności:**
- Rejestrowanie rozpoczęcia wizualizacji danych.
- Uruchomienie skryptu debug/debug_visualize_samples.py, który wizualizuje próbki danych.

**Logowanie:** Rejestrowanie rozpoczęcia opcjonalnego kroku wizualizacji.

### Krok 3: Uczenie Parametrycznego Klasyfikatora Bayesa ML (przy założeniu rozkładu normalnego)

```
logger.log("Step 3: Training Gaussian Bayes model started.")
g_classifier = GaussianBayesClassifier(X_train=hu_train, y_train=y_train, X_test=hu_test, y_test=y_test)
g_classifier.fit()
```

**Opis:** Trening klasyfikatora Bayesa z założeniem rozkładu normalnego (Gaussian Bayes).

**Czynności:**
- Inicjalizacja obiektu GaussianBayesClassifier z danymi treningowymi i testowymi.
- Trening modelu na zbiorze treningowym.

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

<!-- Do raportu:
Wsparcie (Support):

Wsparcie dla danej klasy to liczba wystąpień danej klasy w zbiorze danych testowych.
Wsparcie informuje o tym, jak dobrze zbalansowany jest zbiór danych testowych względem różnych klas.
Dla idealnie zrównoważonych zbiorów danych, wsparcie dla każdej klasy byłoby równe.

Średnie wartości dla wszystkich klas:

Raport klasyfikacji zwykle zawiera również średnie wartości precyzji, czułości, F1-score i wsparcia dla wszystkich klas.
Te średnie wartości są obliczane na podstawie miar dla poszczególnych klas i mogą być przydatne do oceny ogólnej jakości klasyfikatora -->