# File: src/recommend/recommender.py
"""
Simple Laptop Recommendation Logic.

This module provides functions to:
1. Find "Good Deals" (price < predicted price).
2. Recommend laptops based on budget, use case, and features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.models.persistence import load_model
from src.utils import get_logger

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
        'storage_type_SSD': 1.0, # Will need one-hot name
        'ppi': 0.5,
    },
    'content-creation': {
        'cpu_score': 1.5,
        'ram_gb': 1.5,
        'ppi': 1.0,
        'gpu_type_Discrete': 1.0,
    },
    'lightweight': {
        'is_ultrabook': 2.0,
        'weight_kg': -1.5, # Negative weight
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
        self.df_features = None
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
                
            if MODEL_PATH.exists():
                self.model = load_model(MODEL_PATH)
            else:
                logger.warning(f"{MODEL_PATH} not found. Deal finder will be disabled.")
            
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
            processed_data = self.pipeline.transform(self.df)
            feature_names = self.pipeline.get_feature_names_out()
            
            self.df_features = pd.DataFrame(
                processed_data, 
                columns=feature_names,
                index=self.df.index
            )
            
            # Scale features to [0, 1] for similarity
            scaler = MinMaxScaler()
            self.df_features_scaled = pd.DataFrame(
                scaler.fit_transform(self.df_features),
                columns=feature_names,
                index=self.df.index
            )
            
            # Add back original values needed for scoring
            self.df_features_scaled['weight_kg'] = self.df['weight_kg']
            self.df_features_scaled['is_gaming'] = self.df['is_gaming']
            self.df_features_scaled['is_ultrabook'] = self.df['is_ultrabook']
            self.df_features_scaled['cpu_score'] = self.df['cpu_score']
            self.df_features_scaled['ram_gb'] = self.df['ram_gb']
            self.df_features_scaled['ppi'] = self.df['ppi']

            logger.info("Feature DataFrame for recommender is ready.")
        except Exception as e:
            logger.error(f"Error preparing features: {e}", exc_info=True)
            
    def is_ready(self) -> bool:
        """Check if all components are loaded."""
        return (
            self.df is not None and 
            self.pipeline is not None and 
            self.df_features is not None
        )

    def find_good_deals(self, discount_threshold: float = 0.15) -> pd.DataFrame:
        """
        Finds laptops priced significantly below their predicted price.
        """
        if self.model is None or not self.is_ready():
            logger.warning("Model not loaded. Cannot find good deals.")
            return pd.DataFrame()
            
        try:
            # Predict prices (model expects log-transformed target)
            log_preds = self.model.predict(self.df_features)
            predicted_prices = np.expm1(log_preds)
            
            self.df['predicted_price'] = predicted_prices
            self.df['discount_pct'] = (predicted_prices - self.df['price']) / predicted_prices
            
            deals = self.df[
                self.df['discount_pct'] >= discount_threshold
            ].sort_values('discount_pct', ascending=False)
            
            return deals[['brand', 'model', 'price', 'predicted_price', 'discount_pct']]
        
        except Exception as e:
            logger.error(f"Error finding deals: {e}", exc_info=True)
            return pd.DataFrame()

    def recommend_laptops(
        self,
        budget_min: int,
        budget_max: int,
        use_cases: List[str],
        must_haves: List[str],
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Recommends laptops based on filters and a scoring function.
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
            # Get the scaled features for the filtered laptops
            features_filtered = self.df_features_scaled.loc[df_filtered.index]
            df_filtered['recommend_score'] = 0.0

            for use_case in use_cases:
                if use_case in USE_CASE_WEIGHTS:
                    for feature, weight in USE_CASE_WEIGHTS[use_case].items():
                        # Find the one-hot-encoded feature name if needed
                        matching_cols = [c for c in features_filtered.columns if c.startswith(f"cat__{feature}")]
                        
                        if matching_cols:
                            # Apply weight to all matching OHE columns
                            for col in matching_cols:
                                df_filtered['recommend_score'] += features_filtered[col] * weight
                        elif feature in features_filtered.columns:
                            # Direct feature (e.g., numeric or binary)
                            df_filtered['recommend_score'] += features_filtered[feature] * weight
            
            # 4. Add user rating to score
            df_filtered['recommend_score'] += (df_filtered['user_rating'].fillna(3.0) / 5.0)
            
            # 5. Sort and return top K
            df_recs = df_filtered.sort_values('recommend_score', ascending=False)
            
            return df_recs.head(top_k)[[
                'brand', 'model', 'price', 'ram_gb', 'storage_gb', 
                'storage_type', 'cpu_series', 'gpu_model', 'display_size_in', 
                'weight_kg', 'user_rating', 'recommend_score'
            ]]

        except Exception as e:
            logger.error(f"Error during recommendation: {e}", exc_info=True)
            return pd.DataFrame()