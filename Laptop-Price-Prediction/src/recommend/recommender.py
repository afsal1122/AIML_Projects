import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Any, Optional

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.models.persistence import load_model
from src.utils import get_logger

# Import scraper and feature functions for the live pipeline
from src.scraping.scraper_utils import create_polite_session, polite_get
from src.scraping.flipkart_scraper import (
    parse_product_page_flipkart, FLIPKART_LISTING_SELECTOR, 
    FLIPKART_BASE_URL, FLIPKART_SEARCH_URL
)
# --- NEW: Import Amazon Scraper ---
from src.scraping.amazon_scraper import (
    parse_product_listing_amazon, AMAZON_LISTING_SELECTOR,
    AMAZON_BASE_URL, AMAZON_SEARCH_URL
)
# ---
from src.data.features import (
    parse_price, parse_ram, parse_storage, parse_cpu, parse_gpu,
    parse_display, parse_weight, parse_os, calculate_age,
    create_heuristic_features, calculate_cpu_score
)
from src.data.preprocess import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)


logger = get_logger(__name__)

# Paths
PROCESSED_DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")

# Feature weights for use case scoring
USE_CASE_WEIGHTS = {
    'gaming': {
        'is_gaming': 2.0,
        'cpu_score': 1.5,
        'ram_gb': 1.0,
    },
    'programming': {
        'cpu_score': 2.0,
        'ram_gb': 1.5,
        'cat__storage_type_SSD': 1.0, # Mapped to one-hot name
        'ppi': 0.5,
    },
    'content-creation': {
        'cpu_score': 1.5,
        'ram_gb': 1.5,
        'ppi': 1.0,
        'cat__gpu_type_Discrete': 1.0, # Mapped
    },
    'lightweight': {
        'is_ultrabook': 2.0,
        'weight_kg': -1.5, # Negative weight (lower is better)
    }
}

class Recommender:
    """
    A class to handle laptop recommendations and deal finding.
    Loads data and models once on initialization.
    """
    
    def __init__(self):
        self.df = None
        self.pipeline = None
        self.model = None
        self.df_features_scaled = None
        self.feature_names = None
        self.session = create_polite_session() # Use one session for live scrapes
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads data, pipeline, and model."""
        try:
            if PROCESSED_DATA_PATH.exists():
                self.df = pd.read_csv(PROCESSED_DATA_PATH)
            else:
                logger.error(f"{PROCESSED_DATA_PATH} not found.")
                return

            if PIPELINE_PATH.exists():
                self.pipeline = load_model(PIPELINE_PATH)
            else:
                logger.error(f"{PIPELINE_PATH} not found.")
                return
                
            if MODEL_PATH.exists():
                self.model = load_model(MODEL_PATH)
            else:
                logger.warning(f"{MODEL_PATH} not found. Deal finder will be disabled.")
                return
            
            if self.df is not None and self.pipeline is not None:
                self._prepare_features()
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}", exc_info=True)
            
    def _prepare_features(self):
        """
        Applies pipeline and scales features for similarity calculation.
        """
        try:
            # Get processed features
            self.feature_names = self.pipeline.get_feature_names_out()
            
            # Ensure all feature columns exist in df
            model_feature_cols = [
                col for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES)
                if col in self.df.columns
            ]
            processed_data = self.pipeline.transform(self.df[model_feature_cols])
            
            self.df_features = pd.DataFrame(
                processed_data, 
                columns=self.feature_names,
                index=self.df.index
            )
            
            # Scale features to [0, 1] for similarity
            scaler = MinMaxScaler()
            self.df_features_scaled = pd.DataFrame(
                scaler.fit_transform(self.df_features),
                columns=self.feature_names,
                index=self.df.index
            )
            
            # Add back original values needed for scoring heuristics
            for col in ['weight_kg', 'is_gaming', 'is_ultrabook', 'cpu_score', 'ram_gb', 'ppi']:
                if col in self.df.columns:
                    self.df_features_scaled[col] = self.df[col]

            logger.info("Feature DataFrame for recommender is ready.")
        except Exception as e:
            logger.error(f"Error preparing features: {e}", exc_info=True)
            
    def is_ready(self) -> bool:
        """Check if all components are loaded."""
        return (
            self.df is not None and 
            self.pipeline is not None and 
            self.model is not None and
            self.df_features_scaled is not None
        )

    def recommend_laptops(
        self,
        budget_min: int,
        budget_max: int,
        use_cases: List[str],
        must_haves: List[str],
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Recommends laptops from the static dataset based on filters and a scoring function.
        """
        if not self.is_ready():
            logger.error("Recommender is not ready. Artifacts missing.")
            return pd.DataFrame()

        try:
            # 1. Filter by budget
            df_filtered = self.df[
                (self.df['price'] >= budget_min) & (self.df['price'] <= budget_max)
            ].copy()

            if df_filtered.empty:
                logger.info("No laptops found in the specified budget.")
                return pd.DataFrame()

            # 2. Filter by must-haves (simple matching)
            if 'Dedicated GPU' in must_haves:
                df_filtered = df_filtered[df_filtered['gpu_type'] == 'Discrete']
            if 'SSD' in must_haves:
                df_filtered = df_filtered[df_filtered['storage_type'] == 'SSD']

            if df_filtered.empty:
                logger.info("No laptops found after applying 'must-haves'.")
                return pd.DataFrame()

            # 3. Score based on use cases
            features_filtered = self.df_features_scaled.loc[df_filtered.index]
            df_filtered['recommend_score'] = 0.0

            for use_case in use_cases:
                if use_case in USE_CASE_WEIGHTS:
                    for feature_name, weight in USE_CASE_WEIGHTS[use_case].items():
                        if feature_name in features_filtered.columns:
                            df_filtered['recommend_score'] += features_filtered[feature_name] * weight
            
            # 4. Add user rating to score
            df_filtered['recommend_score'] += (df_filtered['user_rating'].fillna(3.0) / 5.0)
            
            # 5. Sort and return top K
            df_recs = df_filtered.sort_values('recommend_score', ascending=False).head(top_k)
            
            # 6. Add Rank
            df_recs['Rank'] = range(1, len(df_recs) + 1)
            
            return df_recs[[
                'Rank', 'brand', 'model', 'price', 'ram_gb', 
                'cpu_series', 'gpu_model', 'user_rating', 'url', 'recommend_score'
            ]]

        except Exception as e:
            logger.error(f"Error during recommendation: {e}", exc_info=True)
            return pd.DataFrame()

    def _run_live_feature_engineering(self, df_live: pd.DataFrame) -> pd.DataFrame:
        """Applies the full feature engineering pipeline to a raw DataFrame."""
        df_live['price'] = df_live['price_raw'].apply(parse_price)
        df_live = df_live.dropna(subset=['price']) # Drop if no price
        
        df_live['ram_gb'] = df_live['ram'].apply(parse_ram)
        storage = df_live['storage'].apply(parse_storage)
        df_live['storage_gb'] = storage.apply(lambda x: x[0])
        df_live['storage_type'] = storage.apply(lambda x: x[1])
        cpu = df_live['cpu'].apply(parse_cpu)
        df_live['cpu_brand'] = cpu.apply(lambda x: x[0])
        df_live['cpu_series'] = cpu.apply(lambda x: x[1])
        df_live['cpu_generation'] = cpu.apply(lambda x: x[2])
        gpu = df_live['gpu'].apply(parse_gpu)
        df_live['gpu_brand'] = gpu.apply(lambda x: x[0])
        df_live['gpu_model'] = gpu.apply(lambda x: x[1])
        df_live['gpu_type'] = gpu.apply(lambda x: x[2])
        display = df_live.apply(lambda r: parse_display(r['display_size'], r['display_size']), axis=1)
        df_live['display_size_in'] = display.apply(lambda x: x[0])
        df_live['resolution'] = display.apply(lambda x: x[1])
        df_live['ppi'] = display.apply(lambda x: x[2])
        df_live['weight_kg'] = df_live['weight'].apply(parse_weight)
        df_live['os_category'] = df_live['os'].apply(parse_os)
        df_live['age_years'] = 1.0 # Assume new
        df_live['user_rating'] = pd.to_numeric(df_live['user_ratings'], errors='coerce').fillna(4.0)
        
        # Apply heuristics
        df_live = df_live.apply(create_heuristic_features, axis=1)
        df_live['cpu_score'] = df_live.apply(calculate_cpu_score, axis=1)
        
        # Fill NaNs with safe defaults from the trained model's dataset
        num_cols = NUMERIC_FEATURES
        for col in num_cols:
             df_live[col] = df_live[col].fillna(self.df[col].median())
        
        cat_cols = CATEGORICAL_FEATURES
        for col in cat_cols:
            df_live[col] = df_live[col].fillna('Unknown')
            
        bin_cols = BINARY_FEATURES
        for col in bin_cols:
            df_live[col] = df_live[col].fillna(0)

        return df_live

    def _predict_live_prices(self, df_live_engineered: pd.DataFrame) -> Optional[np.ndarray]:
        """Uses the loaded model to predict prices for engineered live data."""
        try:
            # Ensure columns are in the same order as when pipeline was trained
            model_feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
            df_live_for_pipeline = df_live_engineered[model_feature_cols]

            transformed_data = self.pipeline.transform(df_live_for_pipeline)
            log_preds = self.model.predict(transformed_data)
            predicted_prices = np.expm1(log_preds)
            return predicted_prices
        except Exception as e:
            logger.error(f"Error during live prediction: {e}")
            logger.error("This can happen if a new feature (e.g., 'Intel Core Ultra') "
                         "appears, which the pipeline wasn't trained on.")
            return None

    def scrape_live_deals_flipkart(self, brand: str, pages: int = 1) -> pd.DataFrame:
        """
        Scrapes live data from Flipkart, predicts price, and finds deals.
        """
        if self.model is None or not self.is_ready():
            logger.error("Model/pipeline not loaded. Cannot get live deals.")
            return pd.DataFrame()
            
        all_data = []
        query = f"{brand} laptop"
        for page in range(1, pages + 1):
            url = FLIPKART_SEARCH_URL.format(query=query, page=page)
            soup = polite_get(self.session, url, delay_seconds=1.0)
            if not soup: break
            listings = soup.select(FLIPKART_LISTING_SELECTOR)
            if not listings: break
            
            for item in listings:
                data = parse_product_page_flipkart(item)
                if data: all_data.append(data)
            time.sleep(1) 

        if not all_data:
            logger.warning(f"No live data found for query: {query}")
            return pd.DataFrame()

        df_live = pd.DataFrame(all_data)
        df_live_engineered = self._run_live_feature_engineering(df_live)
        
        if df_live_engineered.empty:
            logger.warning("Live data was empty after engineering.")
            return pd.DataFrame()

        predicted_prices = self._predict_live_prices(df_live_engineered)
        if predicted_prices is None:
            return pd.DataFrame()

        df_live_engineered['predicted_price'] = predicted_prices
        df_live_engineered['discount_pct'] = (predicted_prices - df_live_engineered['price']) / predicted_prices
        
        deals = df_live_engineered[
            df_live_engineered['discount_pct'] >= 0.10
        ].sort_values('discount_pct', ascending=False)
        
        deals['Rank'] = range(1, len(deals) + 1)
        
        return deals[[
            'Rank', 'model', 'price', 'predicted_price', 'discount_pct', 'url'
        ]]

    def scrape_live_deals_amazon(self, brand: str, pages: int = 1) -> pd.DataFrame:
        """
        Scrapes live data from Amazon, predicts price, and finds deals.
        *** WARNING: LIKELY TO FAIL DUE TO CAPTCHA ***
        """
        if self.model is None or not self.is_ready():
            logger.error("Model/pipeline not loaded. Cannot get live deals.")
            return pd.DataFrame()
            
        all_data = []
        query = f"{brand} laptop"
        for page in range(1, pages + 1):
            url = AMAZON_SEARCH_URL.format(query=query, page=page)
            soup = polite_get(self.session, url, delay_seconds=2.5) # Longer delay
            if not soup:
                logger.error("Failed to fetch Amazon page. Likely CAPTCHA.")
                break
            
            listings = soup.select(AMAZON_LISTING_SELECTOR)
            if not listings:
                logger.warning("No listings found on Amazon. Selectors may be outdated.")
                break
                
            for item in listings:
                data = parse_product_listing_amazon(item)
                if data: all_data.append(data)
            time.sleep(3) # Long delay between pages

        if not all_data:
            logger.warning(f"No live data found for query on Amazon: {query}")
            return pd.DataFrame()

        df_live = pd.DataFrame(all_data)
        df_live_engineered = self._run_live_feature_engineering(df_live)
        
        if df_live_engineered.empty:
            logger.warning("Live Amazon data was empty after engineering.")
            return pd.DataFrame()

        predicted_prices = self._predict_live_prices(df_live_engineered)
        if predicted_prices is None:
            return pd.DataFrame()

        df_live_engineered['predicted_price'] = predicted_prices
        df_live_engineered['discount_pct'] = (predicted_prices - df_live_engineered['price']) / predicted_prices
        
        deals = df_live_engineered[
            df_live_engineered['discount_pct'] >= 0.10
        ].sort_values('discount_pct', ascending=False)
        
        deals['Rank'] = range(1, len(deals) + 1)
        
        return deals[[
            'Rank', 'model', 'price', 'predicted_price', 'discount_pct', 'url'
        ]]