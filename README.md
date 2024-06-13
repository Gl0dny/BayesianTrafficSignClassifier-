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


## Wymagane moduły
Plik `requirements.txt` zawiera wszystkie wymagane pakiety i ich wersje, które są niezbędne do uruchomienia projektu. 

Do obsługi setup'u środowiska wirtualnego wraz z instalacją odpowiednich modułów służdą skrypty `setup.bat` or `setup.sh`.

### Krok 1: Instalacja Wymaganych modułów

Aby zainstalować wymagane moduły, użyj jednego z poniższych skryptów w zależności od systemu operacyjnego.

Dla Windows:
```
.\setup\setup.bat
```
Dla Unix/Linux:
```
./setup/setup.sh
```
Alternatywnie, możesz ręcznie zainstalować wymagane biblioteki używając pip:
```
pip install -r setup/requirements.txt
```
### Krok 2: Uruchomienie Projektu

Uruchom główny skrypt, który przeprowadzi wszystkie kroki projektu:
```
python control/main.py
```
## Szczegółowy Opis Kroków

<!-- ### Rozpakowywanie danych: 
problem/gtsrb.py rozpakowuje zestaw danych GTSRB do katalogu problem/data/GTSRB/.

### Przetwarzanie danych: 
problem/hu_image_data.py przetwarza obrazy, oblicza Hu momenty i dzieli dane na zestawy treningowe i testowe.

### Wizualizacja danych: 
debug/debug_visualize_samples.py wizualizuje przykładowe obrazy oraz ich Hu momenty.

### Trenowanie modeli: 
method/gaussian_bayes.py oraz method/histogram_bayes.py trenują odpowiednio parametryczny oraz nieparametryczny klasyfikator Bayesa i generują raporty z wyników klasyfikacji. -->

## Argumenty Skryptu main.py

Skrypt main.py może przyjmować różne argumenty konfiguracyjne. 

Oto przykład uruchomienia z argumentami:


Dzięki tym instrukcjom, powinieneś być w stanie uruchomić projekt klasyfikacji znaków drogowych przy użyciu klasyfikatora Bayesa oraz zrozumieć strukturę i funkcjonowanie poszczególnych modułów.

<!-- Do raportu:
Wsparcie (Support):

Wsparcie dla danej klasy to liczba wystąpień danej klasy w zbiorze danych testowych.
Wsparcie informuje o tym, jak dobrze zbalansowany jest zbiór danych testowych względem różnych klas.
Dla idealnie zrównoważonych zbiorów danych, wsparcie dla każdej klasy byłoby równe.

Średnie wartości dla wszystkich klas:

Raport klasyfikacji zwykle zawiera również średnie wartości precyzji, czułości, F1-score i wsparcia dla wszystkich klas.
Te średnie wartości są obliczane na podstawie miar dla poszczególnych klas i mogą być przydatne do oceny ogólnej jakości klasyfikatora -->

# README

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


# Plik Logów Monitorujący Postęp: progress_log.txt

## Opis

`progress_log.txt` jest głównym plikiem logów, który monitoruje postęp całego procesu przetwarzania danych, trenowania modeli oraz inne istotne operacje wykonywane przez program. Jest to kluczowy plik do śledzenia, ponieważ zawiera chronologiczny zapis wszystkich ważnych kroków i ewentualnych błędów, które wystąpiły podczas działania programu.

## Zawartość

- **Czas rozpoczęcia i zakończenia operacji:** Każdy wpis zawiera znacznik czasu, który informuje o momencie rozpoczęcia i zakończenia danej operacji.
- **Opisy wykonywanych kroków:** Każdy wpis zawiera opis wykonywanej operacji, co pozwala na łatwe śledzenie, które etapy zostały już zakończone.
- **Komunikaty o błędach:** W przypadku wystąpienia błędów, są one zapisywane w pliku wraz z odpowiednim komunikatem i znacznikiem czasu.
- **Informacje o statusie:** Każdy wpis może zawierać dodatkowe informacje o statusie, takie jak liczba przetworzonych próbek, dokładność modelu, itp.

## Przykład Zawartości `progress_log.txt`

```
[2024-06-13 10:00:00] INFO: Rozpoczęcie procesu przetwarzania danych
[2024-06-13 10:01:00] INFO: Wypakowywanie danych GTSRB z pliku zip
[2024-06-13 10:02:00] INFO: Dane GTSRB zostały pomyślnie wypakowane
[2024-06-13 10:05:00] INFO: Obliczanie momentów Hu dla obrazów
[2024-06-13 10:10:00] INFO: Momentu Hu zostały obliczone dla 1000 obrazów
[2024-06-13 10:15:00] INFO: Podział danych na zestawy treningowe i testowe
[2024-06-13 10:20:00] INFO: Dane zostały podzielone: 800 treningowych, 200 testowych
[2024-06-13 10:30:00] INFO: Trening modelu Gaussian Bayes
[2024-06-13 10:45:00] INFO: Model Gaussian Bayes został wytrenowany
[2024-06-13 10:50:00] INFO: Przewidywanie klas za pomocą modelu Gaussian Bayes
[2024-06-13 11:00:00] INFO: Przewidywanie zakończone, wyniki zapisane do gaussian_bayes_predictions.log
[2024-06-13 11:05:00] INFO: Trening modelu Histogram Bayes
[2024-06-13 11:20:00] INFO: Model Histogram Bayes został wytrenowany
[2024-06-13 11:25:00] INFO: Przewidywanie klas za pomocą modelu Histogram Bayes
[2024-06-13 11:35:00] INFO: Przewidywanie zakończone, wyniki zapisane do histogram_bayes_predictions.log
[2024-06-13 11:40:00] INFO: Cały proces został zakończony pomyślnie
```

### Sposób Działania

1. **Inicjalizacja logowania:** Na początku działania programu, `progress_log.txt` jest tworzony lub otwierany (jeśli już istnieje), a wszystkie wpisy są dodawane na końcu pliku.
2. **Zapis operacji:** Każdy kluczowy krok procesu przetwarzania danych, treningu modeli oraz predykcji jest zapisywany w pliku z odpowiednim komunikatem i znacznikiem czasu.
3. **Monitorowanie błędów:** W przypadku wystąpienia błędu, odpowiedni komunikat o błędzie jest zapisywany do pliku, co pozwala na łatwe zdiagnozowanie problemów.
4. **Zakończenie procesu:** Po zakończeniu całego procesu, zapisywana jest informacja o pomyślnym zakończeniu wraz z czasem zakończenia.

## Korzyści z Użycia `progress_log.txt`

- **Śledzenie postępu:** Umożliwia śledzenie postępu całego procesu, co jest szczególnie przydatne przy długotrwałych operacjach.
- **Diagnostyka błędów:** Pomaga w diagnozowaniu problemów poprzez zapisywanie komunikatów o błędach.
- **Dokumentacja:** Służy jako dokumentacja przebiegu działania programu, co może być przydatne przy analizie wyników i optymalizacji procesów.

## Uwagi

- **Regularne monitorowanie:** Zaleca się regularne monitorowanie `progress_log.txt` podczas działania programu, aby szybko wychwycić ewentualne problemy.
- **Rozmiar pliku:** W przypadku długotrwałych operacji lub dużej ilości danych, rozmiar pliku logów może znacznie wzrosnąć. W takim przypadku warto zaimplementować mechanizmy archiwizacji lub rotacji logów.

Plik `progress_log.txt` jest nieocenionym narzędziem dla programistów i administratorów systemu, zapewniając pełny wgląd w przebieg działania programu i pomagając w szybkim rozwiązywaniu problemów.

# Pliki Logów

## control/logger_utils.py

Plik `logger_utils.py` zawiera klasy Tee i Logger, które zarządzają zapisywaniem komunikatów logów do plików oraz ich wyświetlaniem w terminalu.

## Generowane Pliki Logów

### gaussian_bayes_predictions.log

**Opis:**
Plik zawiera szczegółowe informacje dotyczące predykcji dokonanych przez klasyfikator Gaussian Bayes.

**Zawartość:**
- Szczegóły predykcji dla każdej próbki w zestawie testowym.
- Prawdopodobieństwa a posteriori dla każdej klasy.
- Prawdopodobieństwa warunkowe dla każdej cechy i klasy.

**Format:**
Każda linia zawiera informacje dla jednej próbki:
- Identyfikator próbki, prawdziwa klasa, przewidywana klasa, prawdopodobieństwa a posteriori, prawdopodobieństwa warunkowe.

### histogram_bayes_predictions.log

**Opis:**
Plik zawiera szczegółowe informacje dotyczące predykcji dokonanych przez klasyfikator Histogram Bayes.

**Zawartość:**
- Szczegóły predykcji dla każdej próbki w zestawie testowym.
- Prawdopodobieństwa klas dla każdej próbki na podstawie histogramów.

**Format:**
Każda linia zawiera informacje dla jednej próbki:
- Identyfikator próbki, prawdziwa klasa, przewidywana klasa, prawdopodobieństwa klas.

### hu_moments_class_*.txt

**Opis:**
Pliki zawierają obliczone momenty Hu dla próbek należących do określonej klasy.

**Zawartość:**
- Obliczone momenty Hu dla próbek.
- Zapisane momenty Hu dla analizy i debugowania.

**Format:**
Każda linia zawiera momenty Hu dla jednej próbki:
- Identyfikator próbki, wartości momentów Hu.

### histograms_class_*.txt

**Opis:**
Pliki zawierają histogramy dla każdej cechy i klasy.

**Zawartość:**
- Histogramy dla każdej cechy i klasy.
- Zapisane histogramy do analizy i debugowania.

**Format:**
Każda linia zawiera wartości histogramu dla jednej cechy:
- Indeks cechy, wartości histogramu dla tej cechy.

## Przykłady Plików Logów

### Przykład wpisu w gaussian_bayes_predictions.log:
```
Sample ID: 1234, True Class: 5, Predicted Class: 5, Posterior Probabilities: [0.1, 0.2, 0.3, 0.4], Likelihoods: [0.5, 0.6, 0.7, 0.8]
```
### Przykład wpisu w histogram_bayes_predictions.log:
```
```
### Przykład wpisu w hu_moments_class_5.txt:
```
```
### Przykład wpisu w histograms_class_2.txt:
```
```

## Uwagi

- **Pliki logów są generowane w odpowiednich momentach pracy programu, w zależności od wykonywanych operacji.**
- **Pliki `gaussian_bayes_predictions.log` i `histogram_bayes_predictions.log` są generowane po wykonaniu predykcji przez odpowiednie klasyfikatory.**
- **Pliki `hu_moments_class_*.txt` są generowane podczas obliczania momentów Hu.**
- **Pliki `histograms_class_*.txt` są generowane podczas tworzenia histogramów dla cech i klas.**
