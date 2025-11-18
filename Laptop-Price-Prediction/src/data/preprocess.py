"""
Data Preprocessing Pipeline.
Tailored for 'laptop_cleaned2.csv' structure.
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

# Define Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/laptops_cleaned.csv")

# --- FIX: Ensure this matches your actual file name ---
FALLBACK_DATA_PATH = Path("data/processed/laptop_cleaned2.csv") 
# -----------------------------------------------------

PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

# Define feature groups (Internal names used by the model)
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

FINAL_DISPLAY_COLS = ['model', 'resolution', 'url']

def load_raw_data() -> pd.DataFrame:
    """Loads data from the training CSV or raw JSONs."""
    # Priority 1: Check for the manual CSV first
    if FALLBACK_DATA_PATH.exists():
        logger.info(f"Loading training dataset from {FALLBACK_DATA_PATH}")
        return pd.read_csv(FALLBACK_DATA_PATH)
    
    # Priority 2: Check for scraped JSONs
    json_files = list(RAW_DATA_DIR.glob("*.json"))
    if json_files:
        logger.info(f"Loading {len(json_files)} scraped JSON files.")
        all_data = []
        for f in json_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    all_data.extend(json.load(file))
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        return pd.DataFrame(all_data)

    return pd.DataFrame()

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies mapping from CSV headers to internal feature names."""
    logger.info("Starting feature engineering...")
    
    # 1. CLEAN COLUMN NAMES (Remove spaces)
    df.columns = df.columns.str.strip()
    
    # --- MAP YOUR CSV COLUMNS TO INTERNAL NAMES ---
    
    # PRICE
    if 'Price' in df.columns:
        # Remove commas if present (e.g., "50,399")
        if df['Price'].dtype == 'object':
             df['price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        else:
             df['price'] = df['Price']

    # BRAND
    if 'Brand' in df.columns:
        df['brand'] = df['Brand']

    # MODEL
    if 'Name' in df.columns:
        df['model'] = df['Name']

    # RAM
    if 'RAM_GB' in df.columns:
        df['ram_gb'] = pd.to_numeric(df['RAM_GB'], errors='coerce')

    # STORAGE
    if 'Storage_capacity_GB' in df.columns:
        df['storage_gb'] = pd.to_numeric(df['Storage_capacity_GB'], errors='coerce')
    
    if 'Storage_type' in df.columns:
        # Your CSV has values like " SSD", strip whitespace
        df['storage_type'] = df['Storage_type'].astype(str).str.strip()

    # DISPLAY & RESOLUTION
    if 'Display_size_inches' in df.columns:
        df['display_size_in'] = pd.to_numeric(df['Display_size_inches'], errors='coerce')
    
    if 'Horizontal_pixel' in df.columns and 'Vertical_pixel' in df.columns:
        df['resolution'] = df['Horizontal_pixel'].astype(str) + "x" + df['Vertical_pixel'].astype(str)
    
    if 'ppi' in df.columns:
        df['ppi'] = pd.to_numeric(df['ppi'], errors='coerce')

    # CPU
    if 'Processor_brand' in df.columns:
        df['cpu_brand'] = df['Processor_brand']
    
    if 'Processor_name' in df.columns:
        df['cpu_series'] = df['Processor_name'] # e.g. "Core i5"
    
    if 'Processor_gen' in df.columns:
        # Handle values like "5.0" or missing
        df['cpu_generation'] = pd.to_numeric(df['Processor_gen'], errors='coerce')

    # GPU
    if 'Graphics_brand' in df.columns:
        df['gpu_brand'] = df['Graphics_brand']
    
    if 'Graphics_name' in df.columns:
        df['gpu_model'] = df['Graphics_name']
        
    if 'Graphics_integreted' in df.columns:
         # Map False -> 'Discrete', True -> 'Integrated'
         # Note: Your CSV header is spelled 'Graphics_integreted' (typo in CSV header is handled here)
         df['gpu_type'] = df['Graphics_integreted'].apply(
             lambda x: 'Integrated' if str(x).lower() in ['true', '1'] else 'Discrete'
         )
    else:
        df['gpu_type'] = 'Unknown'

    # OS
    if 'Operating_system' in df.columns:
        # Basic cleaning for OS
        def clean_os(x):
            x = str(x).lower()
            if 'windows' in x: return 'Windows'
            if 'mac' in x: return 'macOS'
            if 'linux' in x or 'ubuntu' in x: return 'Linux'
            return 'Other'
        df['os_category'] = df['Operating_system'].apply(clean_os)

    # WEIGHT (Missing in your CSV headers, so we impute it)
    if 'Weight' in df.columns:
        df['weight_kg'] = df['Weight'].apply(parse_weight)
    else:
        # Create default weights based on display size heuristic
        # 13-14 inch -> 1.4kg, 15-16 inch -> 1.8kg, 17+ inch -> 2.2kg
        conditions = [
            (df['display_size_in'] <= 14.1),
            (df['display_size_in'] > 14.1) & (df['display_size_in'] <= 16.1),
            (df['display_size_in'] > 16.1)
        ]
        values = [1.4, 1.8, 2.2]
        df['weight_kg'] = np.select(conditions, values, default=1.8)

    # RATING
    if 'Rating' in df.columns:
        df['user_rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # URL (Fake it so the app works)
    if 'url' not in df.columns:
        df['url'] = "https://google.com/search?q=" + df['brand'].astype(str) + "+" + df['model'].astype(str)

    # --- END MAPPING ---

    # Calculate Heuristics (Cpu Score, etc.)
    df = df.apply(create_heuristic_features, axis=1)
    df['cpu_score'] = df.apply(calculate_cpu_score, axis=1)
    
    # Handle missing 'age_years' (assume new)
    df['age_years'] = 1.0

    # Select and clean final columns
    final_cols = (
        [TARGET_FEATURE] + NUMERIC_FEATURES + 
        CATEGORICAL_FEATURES + BINARY_FEATURES +
        FINAL_DISPLAY_COLS
    )
    
    # Filter for columns that actually exist
    existing_cols = [col for col in final_cols if col in df.columns]
    df_cleaned = df[existing_cols].copy()

    # Critical check
    if 'price' not in df_cleaned.columns:
        logger.error("CRITICAL ERROR: 'price' column is missing. Check CSV headers.")
        return pd.DataFrame()

    # Drop bad rows
    df_cleaned = df_cleaned.dropna(subset=['price', 'ram_gb', 'storage_gb'])
    
    # Fill missing values
    for col in ['weight_kg', 'user_rating', 'ppi', 'cpu_score']:
        if col in df_cleaned.columns:
            median_val = df_cleaned[col].median()
            if pd.isna(median_val): median_val = 0
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
    logger.info(f"Feature engineering complete. Final shape: {df_cleaned.shape}")
    return df_cleaned

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds the sklearn ColumnTransformer pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor

def main():
    logger.info("Starting preprocessing script...")
    
    df = load_raw_data()
    
    if df.empty:
        logger.error(f"No data found at {FALLBACK_DATA_PATH}. Please check the filename and path.")
        return

    try:
        df_clean = apply_feature_engineering(df)
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        return

    if df_clean.empty:
        logger.error("DataFrame is empty after engineering. Check logs.")
        return

    # Ensure processed data directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the cleaned, human-readable CSV
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8')
    logger.info(f"Cleaned data saved to {PROCESSED_DATA_PATH}")

    # Create X and y
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    
    # Check for missing columns and fill them (e.g. if dataset didn't have weight)
    for col in feature_cols:
        if col not in df_clean.columns:
             df_clean[col] = 0

    X = df_clean[feature_cols].copy()
    y = df_clean[TARGET_FEATURE]
    
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_preprocessing_pipeline()
    
    logger.info("Fitting preprocessing pipeline...")
    pipeline.fit(X_train)
    
    save_model(pipeline, PIPELINE_PATH)
    logger.info(f"Preprocessing pipeline saved to {PIPELINE_PATH}")
    logger.info("Preprocessing script finished successfully.")

if __name__ == "__main__":
    main()