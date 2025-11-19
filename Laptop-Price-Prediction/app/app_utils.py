import streamlit as st
import pandas as pd
import json
import sys, os
from pathlib import Path

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.persistence import load_model
from src.recommend.recommender import Recommender


# ------------------------------------------------------
# 1️⃣ RAW CSV → NORMALIZED COLUMN NAMES (UI-READY DF)
# ------------------------------------------------------
def normalize_raw_df(df):
    """Normalize the raw training_dataset.csv columns so the UI works properly."""
    df = df.copy()

    # --- PRICE ---
    # Handle ANY possible price column name
    possible_price_cols = ["price", "Price", "PRICE", "Selling_price", "MRP", "Sale_price"]
    found_price = None

    for col in possible_price_cols:
        if col in df.columns:
            found_price = col
            break

    if found_price is None:
        st.error("❌ training_dataset.csv has no price column!")
        return df  # return early (UI will handle gracefully)

    # Convert price to numeric
    df["price"] = (
        df[found_price]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    # --- Brand ---
    if "Brand" in df.columns:
        df["brand"] = df["Brand"].astype(str)

    # --- CPU ---
    if "Processor_brand" in df.columns:
        df["cpu_brand"] = df["Processor_brand"].astype(str)

    if "Processor_name" in df.columns:
        df["cpu_series"] = df["Processor_name"].astype(str)

    if "Processor_gen" in df.columns:
        df["cpu_gen"] = df["Processor_gen"].astype(str)

    if "Processor_variant" in df.columns:
        df["cpu_variant"] = df["Processor_variant"].astype(str)

    # --- GPU ---
    if "Graphics_brand" in df.columns:
        df["gpu_brand"] = df["Graphics_brand"].astype(str)

    if "Graphics_name" in df.columns:
        df["gpu_model"] = df["Graphics_name"].astype(str)

    if "Graphics_integreted" in df.columns:
        df["gpu_type"] = df["Graphics_integreted"].apply(
            lambda x: "Integrated" if str(x).lower() == "true" else "Discrete"
        )
    else:
        df["gpu_type"] = "Discrete"

    # --- Display ---
    if "Display_size_inches" in df.columns:
        df["display_size_in"] = pd.to_numeric(df["Display_size_inches"], errors="coerce")

    # --- OS ---
    if "Operating_system" in df.columns:
        df["os_category"] = df["Operating_system"].apply(
            lambda x: "Windows" if "win" in str(x).lower()
            else "macOS" if "mac" in str(x).lower()
            else "Linux" if "linux" in str(x).lower()
            else "Other"
        )

    # --- RAM / Storage ---
    if "RAM_GB" in df.columns:
        df["ram_gb"] = pd.to_numeric(df["RAM_GB"], errors="coerce")

    if "Storage_capacity_GB" in df.columns:
        df["storage_gb"] = pd.to_numeric(df["Storage_capacity_GB"], errors="coerce")

    if "Storage_type" in df.columns:
        df["storage_type"] = df["Storage_type"].astype(str)

    # --- Defaults for safety ---
    df["cpu_gen"] = df.get("cpu_gen", "Unknown")
    df["cpu_variant"] = df.get("cpu_variant", "Unknown")
    df["gpu_brand"] = df.get("gpu_brand", "Unknown")
    df["gpu_model"] = df.get("gpu_model", "Unknown")
    df["gpu_type"] = df.get("gpu_type", "Integrated")
    df["os_category"] = df.get("os_category", "Other")
    df["display_size_in"] = df.get("display_size_in", 15.6)

    return df



# ------------------------------------------------------
# 2️⃣ LOAD PIPELINE + MODEL
# ------------------------------------------------------
def load_pipeline_and_model():
    pipeline = load_model(Path("models/preprocessing_pipeline.pkl"))
    model = load_model(Path("models/best_model.pkl"))
    feats = []

    feat_file = Path("models/feature_names.json")
    if feat_file.exists():
        with open(feat_file, "r") as f:
            feats = json.load(f)

    return pipeline, model, feats


# ------------------------------------------------------
# 3️⃣ FINAL: LOAD EVERYTHING
# ------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads:
      - RAW dataset  (training_dataset.csv)
      - Preprocessing pipeline
      - Prediction model
      - Feature names
      - Recommendation engine
    """

    # ------------- Load RAW dataset for UI -------------
    raw_path = Path("data/processed/training_dataset.csv")
    if not raw_path.exists():
        st.error("❌ training_dataset.csv not found in data/processed/. Please add it.")
        return None, None, None, None, None

    df_raw = pd.read_csv(raw_path)
    df = normalize_raw_df(df_raw)

    # ------------- Load Model + Pipeline --------------
    pipeline, model, feats = load_pipeline_and_model()

    # ------------- Load Recommender -------------------
    rec = Recommender()

    return df, pipeline, model, feats, rec
