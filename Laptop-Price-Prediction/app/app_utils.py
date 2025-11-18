import streamlit as st
import pandas as pd
import json
import sys
import os
from pathlib import Path

# Add project root to path so it can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.persistence import load_model
from src.recommend.recommender import Recommender

# Define paths
DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.json")

@st.cache_resource
def load_artifacts():
    """
    Loads all critical artifacts: data, pipeline, model, and feature names.
    Uses st.cache_resource to load only once across all pages.
    """
    artifacts = {
        "data": None,
        "pipeline": None,
        "model": None,
        "feature_names": None,
        "recommender": None
    }
    
    # Load data
    if not DATA_PATH.exists():
        st.error(f"Error: Missing data file at {DATA_PATH}.")
        return artifacts
    artifacts["data"] = pd.read_csv(DATA_PATH)

    # Load pipeline
    if not PIPELINE_PATH.exists():
        st.error(f"Error: Missing pipeline file at {PIPELINE_PATH}.")
        return artifacts
    artifacts["pipeline"] = load_model(PIPELINE_PATH)

    # Load model
    if not MODEL_PATH.exists():
        st.error(f"Error: Missing model file at {MODEL_PATH}.")
        return artifacts
    artifacts["model"] = load_model(MODEL_PATH)
    
    # Load feature names
    if not FEATURE_NAMES_PATH.exists():
        st.error(f"Error: Missing feature names file at {FEATURE_NAMES_PATH}.")
        return artifacts
    with open(FEATURE_NAMES_PATH, 'r') as f:
        artifacts["feature_names"] = json.load(f)

    # Initialize recommender
    recommender = Recommender()
    if not recommender.is_ready():
        st.error("Recommender system failed to initialize.")
    else:
        artifacts["recommender"] = recommender

    return artifacts