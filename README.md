# Projekt Klasyfikacji Znaków Drogowych z Użyciem Klasyfikatora Bayesa

## Opis Projektu

Ten projekt ma na celu klasyfikację znaków drogowych z wykorzystaniem klasyfikatora Bayesa. Projekt obejmuje przetwarzanie danych, trenowanie modeli klasyfikacyjnych oraz wizualizację wyników.

## Wymagane moduły
Plik requirements.txt zawiera wszystkie wymagane pakiety i ich wersje, które są niezbędne do uruchomienia projektu. 

Wymagane moduły można zainstalować za pomocą polecenia pip install -r requirements.txt, co zapewni, że wszystkie niezbędne pakiety zostaną zainstalowane w odpowiednich wersjach, aby projekt działał prawidłowo.

Do obsługi setup'u środowiska wirtualnego wraz z instalacją odpowiednich modułów służdą skrpty setup.bat or setup.sh.

Dokumentacja Modułów
control
__init__.py

Plik inicjalizujący moduł control. Jest to zazwyczaj pusty plik używany do oznaczenia katalogu jako modułu Pythona.
logger_utils.py

Zawiera funkcje i klasy wspomagające logowanie w aplikacji. Może zawierać ustawienia formatowania logów, poziomy logowania oraz mechanizmy zapisu logów do plików lub wyświetlania na konsoli.
main.py

Główny skrypt do uruchomienia projektu. Może zawierać logikę inicjalizacji, ładowania danych, trenowania modelu i ewaluacji wyników.
debug
debug_visualize_samples.py

Skrypt do wizualizacji próbek danych dla celów debugowania. Pomaga w zrozumieniu, jak wyglądają dane wejściowe oraz weryfikacji poprawności przetwarzania danych.
method
__init__.py

Plik inicjalizujący moduł method. Jest to zazwyczaj pusty plik używany do oznaczenia katalogu jako modułu Pythona.
gaussian_bayes.py

Zawiera implementację klasyfikatora Gaussian Naive Bayes. Klasyfikator ten wykorzystuje założenie, że cechy mają rozkład normalny, aby obliczyć prawdopodobieństwa przynależności do klas.
histogram_bayes.py

Zawiera implementację klasyfikatora Histogram Bayes. Może to być klasyfikator oparty na histogramach cech, który wykorzystuje dystrybucje cech do obliczenia prawdopodobieństw przynależności do klas.
problem
data/GTSRB

Katalog zawierający dane GTSRB (German Traffic Sign Recognition Benchmark). Dane te są używane do trenowania i testowania modeli klasyfikacyjnych.

    gtsrb.zip: Skompresowany plik z danymi GTSRB.
    __init__.py: Plik inicjalizujący katalog GTSRB jako moduł Pythona.
    gtsrb.py: Skrypt zawierający funkcje do ładowania i przetwarzania danych GTSRB.
    hu_image_data.py: Skrypt do ekstrakcji momentów Hu z obrazów znaków drogowych oraz podziału danych na zestawy treningowe i testowe.

__init__.py

Plik inicjalizujący moduł problem. Jest to zazwyczaj pusty plik używany do oznaczenia katalogu jako modułu Pythona.
setup
requirements.txt

Plik zawierający listę zależności Pythona wymaganych do uruchomienia projektu. Może zawierać nazwy pakietów i ich wersje.
setup.bat

Skrypt wsadowy dla systemu Windows do instalacji zależności i konfiguracji środowiska.
setup.sh

Skrypt powłoki dla systemów Unix/Linux do instalacji zależności i konfiguracji środowiska.
Pliki konfiguracyjne
.gitattributes

Plik konfiguracyjny Gita określający atrybuty plików. Może zawierać informacje o tym, jak Git powinien traktować określone pliki (np. normalizacja końców linii).
.gitignore

Plik konfiguracyjny Gita określający pliki i katalogi, które powinny być ignorowane przez system kontroli wersji Git. Może zawierać tymczasowe pliki, katalogi build, dane wyjściowe itp.
Instrukcje do Uruchomienia
Krok 1: Instalacja Wymaganych Bibliotek

Aby zainstalować wymagane biblioteki, użyj jednego z poniższych skryptów w zależności od systemu operacyjnego.

Dla Windows:

setup.bat

Dla Unix/Linux:

setup.sh

Alternatywnie, możesz ręcznie zainstalować wymagane biblioteki używając pip:

pip install -r setup/requirements.txt

Krok 2: Uruchomienie Projektu

Uruchom główny skrypt, który przeprowadzi wszystkie kroki projektu:

python control/main.py

Szczegółowy Opis Kroków

    Rozpakowywanie danych: Skrypt problem/gtsrb.py rozpakowuje zestaw danych GTSRB do katalogu problem/data/GTSRB/.

    Przetwarzanie danych: Skrypt problem/hu_image_data.py przetwarza obrazy, oblicza Hu momenty i dzieli dane na zestawy treningowe i testowe.

    Wizualizacja danych: Skrypt debug/debug_visualize_samples.py wizualizuje przykładowe obrazy oraz ich Hu momenty.

    Trenowanie modeli: Skrypty method/gaussian_bayes.py oraz method/histogram_bayes.py trenują odpowiednio parametryczny (Gaussian Naive Bayes) oraz nieparametryczny (Histogram Bayes) klasyfikator Bayesa i generują raporty z wyników klasyfikacji.

Argumenty Skryptu main.py

Skrypt main.py może przyjmować różne argumenty konfiguracyjne. Oto przykład uruchomienia z argumentami:

bash

python control/main.py --data_dir problem/data/GTSRB/ --output_dir results/

    --data_dir: Ścieżka do katalogu z danymi.
    --output_dir: Ścieżka do katalogu, gdzie mają być zapisane wyniki.

Dzięki tym instrukcjom, powinieneś być w stanie uruchomić projekt klasyfikacji znaków drogowych przy użyciu klasyfikatora Bayesa oraz zrozumieć strukturę i funkcjonowanie poszczególnych modułów.

## Struktura Projektu

BAYESIANTRAFFICSIGNCLASSIFIER
│
├── control
│ ├── init.py
│ ├── logger_utils.py
│ └── main.py
│
├── debug
│ └── debug_visualize_samples.py
│
├── method
│ ├── init.py
│ ├── gaussian_bayes.py
│ └── histogram_bayes.py
│
├── problem
│ ├── init.py
│ ├── gtsrb.py
│ ├── hu_image_data.py
│ └── data
│ └── GTSRB
│ └── gtsrb.zip
│
├── setup
│ ├── requirements.txt
│ ├── setup.bat
│ └── setup.sh
│
├── .gitattributes
├── .gitignore
└── README.md


Poniżej znajduje się krótki opis głównych katalogów i plików:

- **control**: Zawiera główne skrypty kontrolne projektu.
  - `__init__.py`: Plik inicjalizacyjny dla modułu control.
  - `logger_utils.py`: Funkcje pomocnicze do logowania.
  - `main.py`: Główny skrypt do uruchamiania klasyfikatora.

- **debug**: Zawiera skrypty do debugowania.
  - `debug_visualize_samples.py`: Skrypt do wizualizacji próbek w celach debugowania.

- **method**: Zawiera implementację metod bayesowskich.
  - `__init__.py`: Plik inicjalizacyjny dla modułu method.
  - `gaussian_bayes.py`: Implementacja klasyfikacji bayesowskiej z użyciem rozkładu Gaussa.
  - `histogram_bayes.py`: Implementacja klasyfikacji bayesowskiej z użyciem histogramów.

- **problem**: Zawiera pliki i dane specyficzne dla problemu.
  - `__init__.py`: Plik inicjalizacyjny dla modułu problem.
  - `gtsrb.py`: Metody do obsługi danych GTSRB.
  - `hu_image_data.py`: Metody do obsługi danych obrazowych z momentami Hu.
  - **data/GTSRB**: Katalog zawierający zestaw danych GTSRB.
    - `gtsrb.zip`: Skompresowany plik z zestawem danych GTSRB.

- **setup**: Zawiera skrypty instalacyjne i plik z wymaganiami.
  - `requirements.txt`: Lista zależności wymaganych do projektu.
  - `setup.bat`: Skrypt wsadowy do instalacji projektu w systemie Windows.
  - `setup.sh`: Skrypt powłoki do instalacji projektu w systemach Unix.



    Do raportu:
    Wsparcie (Support):

    Wsparcie dla danej klasy to liczba wystąpień danej klasy w zbiorze danych testowych.
    Wsparcie informuje o tym, jak dobrze zbalansowany jest zbiór danych testowych względem różnych klas.
    Dla idealnie zrównoważonych zbiorów danych, wsparcie dla każdej klasy byłoby równe.

    Średnie wartości dla wszystkich klas:

    Raport klasyfikacji zwykle zawiera również średnie wartości precyzji, czułości, F1-score i wsparcia dla wszystkich klas.
    Te średnie wartości są obliczane na podstawie miar dla poszczególnych klas i mogą być przydatne do oceny ogólnej jakości klasyfikatora