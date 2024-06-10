# BayesianTrafficSignClassifier

project_root/
│
├── data/
│   ├── GTSRB/  # katalog z danymi GTSRB
│   │   └── Final_Training/
│   │       └── Images/  # tutaj będą przechowywane obrazy po rozpakowaniu
│   └── download_and_extract_gtsrb.py  # skrypt do pobierania i rozpakowywania danych
│
├── scripts/
│   ├── preprocess_data.py  # skrypt do wczytywania, przetwarzania i podziału danych
│   ├── visualize_samples.py  # skrypt do wizualizacji przykładowych obrazów
│   ├── train_model.py  # skrypt do trenowania modelu (zostanie dodany później)
│   └── evaluate_model.py  # skrypt do oceny modelu (zostanie dodany później)
│
├── models/  # katalog do przechowywania wytrenowanych modeli
│
└── main.py  # główny plik do uruchomienia całego procesu