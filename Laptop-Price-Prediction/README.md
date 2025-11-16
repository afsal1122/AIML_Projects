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

## 🛠️ Setup & How to Run

You can run this project in two ways:
1.  **(Recommended)** Use the bundled sample data to run the Streamlit app instantly.
2.  (Advanced) Run the full pipeline from scratch (scrape, preprocess, train).

### 1. Run the Streamlit App (Instant Demo)

This uses the pre-processed sample data in `data/processed/laptops_cleaned.csv`. No scraping or training is required.

**Prerequisites:**
* Python 3.9+

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Laptop-Price-Prediction.git](https://github.com/your-username/Laptop-Price-Prediction.git)
    cd Laptop-Price-Prediction
    ```

2.  **Run the setup script:**
    This will create a virtual environment, install requirements, and launch the app.

    * **On macOS/Linux:**
        ```bash
        chmod +x run_local.sh
        ./run_local.sh
        ```
    * **On Windows:**
        ```bat
        .\run_local.bat
        ```

3.  **Done!** The Streamlit app will open in your browser.

---

### 2. Run the Full Pipeline (From Scratch)

This will run the entire process: scrape new data, preprocess it, and train the models.

**⚠️ Scraping Ethics:**
* **Check `robots.txt`:** Before scraping any website (e.g., `https://www.amazon.in/robots.txt`), always check its `robots.txt` file to see which paths are disallowed.
* **Be Polite:** The scrapers in `src/scraping/` are designed to be polite (headers, delays, retries), but websites change frequently. **Selectors will break and need to be updated.**
* **Fallback:** If scraping fails, the project is designed to fall back to the sample data.

**Instructions:**

1.  **Setup the environment:**
    Follow Step 1 & 2 from the "Instant Demo" section, but **stop** the Streamlit app (Ctrl+C) after dependencies are installed. Your virtual environment (`venv`) will be active.

2.  **(Optional) Scrape Data:**
    *Note: This step is likely to fail without updating selectors.*
    ```bash
    python -m src.scraping.flipkart_scraper --pages 5
    python -m src.scraping.amazon_scraper --pages 3
    ```

3.  **Preprocess Data:**
    This reads from `data/raw` (or the sample CSV if raw is empty) and creates the final `data/processed/laptops_cleaned.csv` and the preprocessing pipeline.
    ```bash
    python -m src.data.preprocess
    ```

4.  **Train Models:**
    This reads the processed data, trains the models, and saves the best one to `models/best_model.pkl`.
    ```bash
    python -m src.models.train
    ```

5.  **Run the App:**
    Now that you have your own trained models, run the app.
    ```bash
    streamlit run app/streamlit_app.py
    ```

## 🧪 Running Tests

Ensure you have installed the dev dependencies (like `pytest`):
```bash
pip install -r requirements.txt
Then run:

Bash

pytest

-------------------------------------------------------------------------------------------------------
Here are the step-by-step commands to run the individual parts of the project from the VS Code terminal.

📍 Before You Start: Activate Your Environment
You must do this first. All other commands will fail if you don't.

Open the VS Code Terminal (View > Terminal).

Check your prompt. If you see (venv) at the beginning, you are ready.

If you don't see (venv), activate it by running the command for your system:

Windows: .\venv\Scripts\activate

macOS/Linux: source venv/bin/activate

1. 🕸️ To Run the Scrapers (Scrape)
This is a manual step and is optional. The app will work without it by using the sample data.

Warning: Scrapers break often as websites change. These will likely fail unless you update the code.

In the terminal, run:

Bash

# To run the Flipkart scraper
python -m src.scraping.flipkart_scraper

# To run the Amazon scraper
python -m src.scraping.amazon_scraper

Tip: You can add the --pages argument to control how many pages to scrape, like python -m src.scraping.flipkart_scraper --pages 3.

2. 🧹 To Run Preprocessing
This step is required if you've scraped new data. It takes the raw data and turns it into the clean file the model needs.

In the terminal, run:

Bash

python -m src.data.preprocess

This will read from data/raw/ and create the final data/processed/laptops_cleaned.csv and the models/preprocessing_pipeline.pkl files.

3. 🤖 To Run Model Training (Train)
This step trains the machine learning model. It must be run after Step 2.

In the terminal, run:

Bash

python -m src.models.train

This will load the processed data, train the models, and save the best one as models/best_model.pkl.

4. 🧪 To Run the Tests (Test)
This runs all the project's automated tests using pytest to make sure the code is working as expected.

In the terminal, run:

Bash

pytest

This will automatically find and run all files in the tests/ folder.

5. 🚀 To Run the Final App
This is what the run_local scripts do for you. This command starts the web application.

In the terminal, run:

Bash

streamlit run app/streamlit_app.py