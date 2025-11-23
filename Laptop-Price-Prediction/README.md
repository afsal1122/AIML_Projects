
# Laptop Price Prediction System

A full-stack ML application to predict laptop prices, explore data, and recommend laptops.

Laptop-Price-Prediction/
├── app/
│   ├── pages/
│   │   ├── 1_Price_Predictor.py
│   │   └── 2_Data_Explorer.py
│   ├── app_utils.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── training_dataset.csv     # raw user dataset (schema unknown)
│   └── processed/
│       └── laptops_cleaned.csv       # generated automatically
├── models/
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── features.py
│   ├── models/
│   │   ├── persistence.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── recommend/
│   │   └── recommender.py
│   └── utils.py
├── requirements.txt
└── README.md

## Structure
- **app/**: Streamlit frontend application.
- **data/**: Raw and processed datasets.
- **models/**: Saved ML models and evaluation artifacts.
- **src/**: Source code for preprocessing, training, and utilities.

## Installation

```bash
pip install -r requirements.txt
Execution Pipeline
Preprocess Data (Cleans raw CSV, generates features):
code
Bash
python -m src.data.preprocess
Train Model (Trains XGBoost, tunes params, saves model):
code
Bash
python -m src.models.train
Run App:
code
Bash
streamlit run app/streamlit_app.py