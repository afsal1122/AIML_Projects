Laptop-Price-Prediction/
├── data/
│   ├── processed/
│   │   └── laptops_cleaned.csv
│   ├── interim/
│   └── raw/
├── models/
│   ├── evaluation/
│   ├── best_model.pkl
│   └── preprocessing_pipeline.pkl
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── scraping/
│   │   ├── flipkart_scraper.py
│   │   ├── amazon_scraper.py
│   │   └── scraper_utils.py
│   ├── data/
│   │   ├── preprocess.py
│   │   └── features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── persistence.py
│   ├── recommend/
│   │   └── recommender.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocess.py
│   └── test_recommender.py
├── README.md
├── requirements.txt
├── run_local.bat
└── run_local.sh