# src/data/preprocess.py
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import re
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Now import from utils
try:
    from src.utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_logger
except ImportError:
    # Fallback if import fails
    NUMERIC_FEATURES = [
        "Processor_gen_num", "Core_per_processor", "Threads", "RAM_GB",
        "Storage_capacity_GB", "Graphics_GB", "Display_size_inches",
        "Horizontal_pixel", "Vertical_pixel", "ppi",
    ]
    
    CATEGORICAL_FEATURES = [
        "Brand", "Processor_brand", "Processor_name", "Processor_variant",
        "Storage_type", "Graphics_name", "Graphics_brand", "Operating_system",
    ]
    
    def get_logger(name):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

logger = get_logger("PREPROCESS")

RAW_PATHS = [Path("data/raw/training_dataset.csv")]
PROCESSED_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
TARGET = "Price"

def parse_proc_gen_to_num(val):
    """Convert Processor_gen strings to numeric encoding"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s_lower = s.lower()

    # Apple M-series
    m = re.match(r"m\s*([1-9][0-9]?)", s_lower)
    if m:
        try:
            return 100 + int(m.group(1))
        except:
            pass

    # Intel Special mappings
    if 'meteor lake' in s_lower or 'ultra' in s_lower:
        return 14

    # Direct digits
    digits = re.findall(r"(\d+)", s)
    if digits:
        try:
            return int(digits[0])
        except:
            pass

    return np.nan

def extract_vram(gpu_name):
    """Extract VRAM from GPU name"""
    if pd.isna(gpu_name):
        return np.nan
    try:
        match = re.search(r'(\d+)\s*GB', str(gpu_name), re.IGNORECASE)
        if match:
            return int(match.group(1))
    except:
        pass
    return np.nan

def load_raw():
    for p in RAW_PATHS:
        if p.exists():
            logger.info(f"Loading data from {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"Raw dataset not found. Searched: {RAW_PATHS}")

def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normalize column names
    alt_map = {
        "price": "Price",
        "processor_gen": "Processor_gen", 
        "ram_gb": "RAM_GB",
        "storage_capacity_gb": "Storage_capacity_GB",
        "horizontal_pixel": "Horizontal_pixel",
        "vertical_pixel": "Vertical_pixel",
    }
    
    for k, v in alt_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Ensure categorical columns are strings
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = "Unknown"

    # Numeric conversions
    numeric_cols = ["Core_per_processor", "Threads", "RAM_GB", "Storage_capacity_GB",
                   "Graphics_GB", "Display_size_inches", "Horizontal_pixel", "Vertical_pixel", "ppi"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Clean target
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    else:
        raise ValueError("Price column missing from dataset")

    # Create Processor_gen_num
    df["Processor_gen_num"] = df["Processor_gen"].apply(parse_proc_gen_to_num)

    # Extract VRAM from Graphics_name if Graphics_GB is missing
    df["Graphics_GB"] = df["Graphics_GB"].fillna(df["Graphics_name"].apply(extract_vram))

    # Calculate PPI if missing
    mask = df["Horizontal_pixel"].notna() & df["Vertical_pixel"].notna() & df["Display_size_inches"].notna()
    df.loc[mask, "ppi"] = np.sqrt(
        df.loc[mask, "Horizontal_pixel"]**2 + 
        df.loc[mask, "Vertical_pixel"]**2
    ) / df.loc[mask, "Display_size_inches"]

    # Keep only required columns
    required = [TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in required:
        if col not in df.columns:
            df[col] = np.nan if col in NUMERIC_FEATURES else "Unknown"

    df_clean = df[required].copy()
    df_clean = df_clean.dropna(subset=[TARGET]).reset_index(drop=True)
    
    # Data validation
    df_clean = df_clean[df_clean["Price"] > 1000]
    df_clean = df_clean[df_clean["Price"] < 500000]
    
    # Save processed data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Saved processed dataset: {PROCESSED_PATH} (rows: {len(df_clean)})")
    
    return df_clean

def build_pipeline():
    # Numeric pipeline
    num_proc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline  
    cat_proc = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_proc, NUMERIC_FEATURES),
        ("cat", cat_proc, CATEGORICAL_FEATURES)
    ], remainder="drop")

    return preprocessor

def main():
    logger.info("Starting preprocessing pipeline...")
    
    raw = load_raw()
    logger.info(f"Raw data loaded: {len(raw)} rows")
    
    df_clean = clean_and_select(raw)
    logger.info(f"Data cleaned: {len(df_clean)} rows remaining")
    
    pipeline = build_pipeline()
    X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pipeline.fit(X)
    
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info(f"Preprocessing pipeline saved: {PIPELINE_PATH}")

if __name__ == "__main__":
    main()