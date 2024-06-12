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
project_root/
│
├── data/ # Katalog zawierający dane
│ ├── GTSRB/ # Katalog z danymi GTSRB
│ │ └── Traffic_Signs/ # Przetworzone dane i cechy
│ │ ├── X_train.npy # Obrazy do trenowania
│ │ ├── X_test.npy # Obrazy do testowania
│ │ ├── hu_train.npy # Hu momenty do trenowania
│ │ ├── hu_test.npy # Hu momenty do testowania
│ │ ├── y_train.npy # Etykiety do trenowania
│ │ └── y_test.npy # Etykiety do testowania
│ └── extract_gtsrb.py # Skrypt do rozpakowywania danych
│
├── scripts/ # Katalog zawierający skrypty
│ ├── main.py # Główny skrypt do uruchamiania projektu
│ ├── extract_gtsrb.py # Skrypt do rozpakowywania danych GTSRB
│ ├── preprocess_data.py # Skrypt do przetwarzania danych i obliczania Hu momentów
│ ├── visualize_samples.py # Skrypt do wizualizacji danych
│ ├── train_gaussian_nb.py # Skrypt do trenowania parametrycznego klasyfikatora Bayesa
│ └── train_histogram_nb.py # Skrypt do trenowania nieparametrycznego klasyfikatora Bayesa
│
└── models/ # Katalog do przechowywania wytrenowanych modeli

    Do raportu:
    Wsparcie (Support):

    Wsparcie dla danej klasy to liczba wystąpień danej klasy w zbiorze danych testowych.
    Wsparcie informuje o tym, jak dobrze zbalansowany jest zbiór danych testowych względem różnych klas.
    Dla idealnie zrównoważonych zbiorów danych, wsparcie dla każdej klasy byłoby równe.

    Średnie wartości dla wszystkich klas:

    Raport klasyfikacji zwykle zawiera również średnie wartości precyzji, czułości, F1-score i wsparcia dla wszystkich klas.
    Te średnie wartości są obliczane na podstawie miar dla poszczególnych klas i mogą być przydatne do oceny ogólnej jakości klasyfikatora