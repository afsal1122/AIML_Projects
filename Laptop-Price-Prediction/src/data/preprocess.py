# File: src/data/preprocess.py
"""
Data Preprocessing Pipeline.

This script loads raw data (either from scrapers or a fallback CSV),
applies the feature engineering functions, builds a scikit-learn
preprocessing pipeline (ColumnTransformer), and saves the final
processed DataFrame and the pipeline object.

Run as module:
python -m src.data.preprocess
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import get_logger
from src.models.persistence import save_model
from src.data.features import (
    parse_price, parse_ram, parse_storage, parse_cpu, parse_gpu,
    parse_display, parse_weight, parse_os, calculate_age,
    create_heuristic_features, calculate_cpu_score
)

logger = get_logger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/laptops_cleaned.csv")
# Fallback in case no raw data is present
FALLBACK_DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

# Define feature groups
NUMERIC_FEATURES = [
    'ram_gb', 'storage_gb', 'display_size_in', 'weight_kg', 
    'ppi', 'age_years', 'user_rating', 'cpu_score'
]
CATEGORICAL_FEATURES = [
    'brand', 'storage_type', 'cpu_brand', 'cpu_series', 
    'gpu_brand', 'gpu_type', 'os_category'
]
BINARY_FEATURES = ['is_gaming', 'is_ultrabook']
TARGET_FEATURE = 'price'

def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Loads all JSON files from the raw data directory."""
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No raw JSON files found in {data_dir}.")
        return pd.DataFrame()

    all_data = []
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.extend(data)
        except Exception as e:
            logger.error(f"Failed to load or parse {f}: {e}")
            
    logger.info(f"Loaded {len(all_data)} records from {len(json_files)} files.")
    return pd.DataFrame(all_data)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all parsing and feature creation functions."""
    logger.info("Starting feature engineering...")
    
    # Handle potential column name differences from scrapers
    df['price'] = df.get('price_raw', df.get('price')).apply(parse_price)
    df['ram_gb'] = df.get('ram', df.get('ram_gb')).apply(parse_ram)
    
    storage_cols = df.get('storage', df.get('storage_gb', pd.Series(dtype='str')))
    storage = storage_cols.apply(parse_storage)
    df['storage_gb'] = storage.apply(lambda x: x[0])
    df['storage_type'] = storage.apply(lambda x: x[1])

    cpu_cols = df.get('cpu', df.get('cpu_brand', pd.Series(dtype='str')))
    cpu = cpu_cols.apply(parse_cpu)
    df['cpu_brand'] = cpu.apply(lambda x: x[0])
    df['cpu_series'] = cpu.apply(lambda x: x[1])
    df['cpu_generation'] = cpu.apply(lambda x: x[2])

    gpu_cols = df.get('gpu', df.get('gpu_brand', pd.Series(dtype='str')))
    gpu = gpu_cols.apply(parse_gpu)
    df['gpu_brand'] = gpu.apply(lambda x: x[0])
    df['gpu_model'] = gpu.apply(lambda x: x[1])
    df['gpu_type'] = gpu.apply(lambda x: x[2])
    df['gpu_vram_gb'] = gpu.apply(lambda x: x[3]) # Not used in model, but good to have
    
    display = df.apply(lambda row: parse_display(
        row.get('display_size', row.get('display_size_in')), 
        row.get('resolution')
    ), axis=1)
    df['display_size_in'] = display.apply(lambda x: x[0])
    df['resolution'] = display.apply(lambda x: x[1])
    df['ppi'] = display.apply(lambda x: x[2])

    df['weight_kg'] = df.get('weight', df.get('weight_kg')).apply(parse_weight)
    df['os_category'] = df.get('os', df.get('os_category')).apply(parse_os)
    df['age_years'] = df.get('release_year', df.get('age_years')).apply(calculate_age)
    
    df['user_rating'] = pd.to_numeric(
        df.get('user_ratings', df.get('user_rating')), 
        errors='coerce'
    ).apply(lambda x: x if 0 <= x <= 5 else None)

    # Heuristics
    df = df.apply(create_heuristic_features, axis=1)
    df['cpu_score'] = df.apply(calculate_cpu_score, axis=1)

    # Select and clean final columns
    final_cols = (
        [TARGET_FEATURE] + NUMERIC_FEATURES + 
        CATEGORICAL_FEATURES + BINARY_FEATURES +
        ['model', 'resolution'] # Keep for recommender/display
    )
    # Filter for columns that actually exist in the dataframe
    existing_cols = [col for col in final_cols if col in df.columns]
    df_cleaned = df[existing_cols].copy()

    # Drop rows where target or critical features are missing
    df_cleaned = df_cleaned.dropna(subset=[
        'price', 'ram_gb', 'storage_gb', 'display_size_in', 'cpu_brand'
    ])
    
    # Fill missing age/weight/rating with reasonable defaults (median)
    # This will also be handled by the pipeline, but good for the CSV
    for col in ['age_years', 'weight_kg', 'user_rating', 'ppi']:
        if col in df_cleaned.columns:
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
    logger.info(f"Feature engineering complete. Final shape: {df_cleaned.shape}")
    return df_cleaned

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds the sklearn ColumnTransformer pipeline."""
    
    # Pipeline for numeric features: Impute with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: Impute with 'Unknown', then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Binary features just need imputation
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Create the full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='drop' # Drop columns not specified (e.g., 'model')
    )
    
    return preprocessor

def main():
    logger.info("Starting preprocessing script...")
    
    df_raw = load_raw_data()
    
    if df_raw.empty:
        logger.warning("No raw data found. Attempting to load fallback CSV...")
        if FALLBACK_DATA_PATH.exists():
            df = pd.read_csv(FALLBACK_DATA_PATH)
            logger.info(f"Loaded fallback data from {FALLBACK_DATA_PATH}")
            
            # --- THIS IS THE FIX ---
            # Manually create cpu_score as it's missing from the sample CSV
            if 'cpu_score' not in df.columns:
                # These lines are now correctly indented
                logger.info("Manually calculating 'cpu_score' for fallback data.")
                df['cpu_score'] = df.apply(calculate_cpu_score, axis=1)
            # --- END OF FIX ---
            
        else:
            logger.error("No raw data and no fallback CSV. Exiting.")
            return
    else:
        # If we have raw data, process it
        df = apply_feature_engineering(df_raw)

    # Ensure processed data directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the cleaned, human-readable CSV
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8')
    logger.info(f"Cleaned data saved to {PROCESSED_DATA_PATH}")

    # Now, build and save the pipeline
    # We only fit the pipeline on the training data
    if TARGET_FEATURE not in df.columns:
        logger.error(f"Target feature '{TARGET_FEATURE}' not in DataFrame. Aborting.")
        return

    X = df.drop(columns=TARGET_FEATURE)
    y = df[TARGET_FEATURE]
    
    # Split to get a training set for fitting the pipeline
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_preprocessing_pipeline()
    
    # Fit the pipeline on the training data
    logger.info("Fitting preprocessing pipeline...")
    pipeline.fit(X_train)
    
    # Save the fitted pipeline
    save_model(pipeline, PIPELINE_PATH)
    logger.info(f"Preprocessing pipeline saved to {PIPELINE_PATH}")
    
    logger.info("Preprocessing script finished successfully.")

if __name__ == "__main__":
    main()