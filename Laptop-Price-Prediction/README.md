📂 Project Structure

Laptop-Price-Prediction/
├── app/
│   ├── pages/
│   │   ├── 1_Price_Predictor.py      # Page: Custom Laptop Pricing
│   │   └── 2_Data_Explorer.py        # Page: Graphs & Charts
│   ├── app_utils.py                  # Helper to load models efficiently
│   └── streamlit_app.py              # Main Page: Recommender
├── data/
│   ├── processed/
│   │   └── training_dataset.csv      # <--- PUT YOUR CSV FILE HERE (Rename it)
├── models/
│   └── (Empty initially; scripts create files here)
├── src/
│   ├── data/
│   │   ├── preprocess.py             # Cleans data & creates pipeline
│   │   └── features.py               # Feature engineering logic
│   ├── models/
│   │   ├── train.py                  # Trains the AI "Brain"
│   │   ├── evaluate.py               # Metrics (RMSE, R2)
│   │   └── persistence.py            # Save/Load logic
│   ├── recommend/
│   │   └── recommender.py            # Logic for finding best laptops
│   └── utils.py                      # Logging & Plotting tools
├── requirements.txt
└── README.md

1. Setup Files
requirements.txt

2. Source Code (src/)
src/utils.py

src/models/persistence.py

src/models/evaluate.py

src/data/features.py

src/data/preprocess.py This file handles the mapping from your CSV headers to the app's format.

src/models/train.py Uses RandomizedSearchCV to ensure high accuracy.

src/recommend/recommender.py

3. Streamlit Application (app/)
app/app_utils.py

app/streamlit_app.py (Home Page)

app/pages/1_Price_Predictor.py

app/pages/2_Data_Explorer.py

🚀 Final Steps to Run


Process: python -m src.data.preprocess

Train: python -m src.models.train

Run: python -m streamlit run app/streamlit_app.py