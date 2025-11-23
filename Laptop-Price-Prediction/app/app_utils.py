# app/app_utils.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import sys
import os
from pathlib import Path

# Path Setup
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.persistence import load_model
from src.recommend.recommender import Recommender

# Paths
RAW_PATH = Path("data/raw/training_dataset.csv")
PROCESSED_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
FEATURES_PATH = Path("models/feature_names.json")

@st.cache_resource
def load_artifacts():
    """Load all required artifacts for the app without success messages"""
    artifacts = {
        'df': None,
        'pipeline': None, 
        'model': None,
        'feature_names': [],
        'recommender': None
    }
    
    # Load Data - silently without messages
    if PROCESSED_PATH.exists():
        artifacts['df'] = pd.read_csv(PROCESSED_PATH)
    elif RAW_PATH.exists():
        artifacts['df'] = pd.read_csv(RAW_PATH)
    else:
        return artifacts

    # Load Pipeline silently
    artifacts['pipeline'] = load_model(PIPELINE_PATH)

    # Load Model silently
    artifacts['model'] = load_model(MODEL_PATH)

    # Load Feature Names silently
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r") as f:
                artifacts['feature_names'] = json.load(f)
        except Exception:
            pass

    # Initialize Recommender silently
    try:
        artifacts['recommender'] = Recommender()
    except Exception:
        artifacts['recommender'] = None

    return (artifacts['df'], artifacts['pipeline'], artifacts['model'], 
            artifacts['feature_names'], artifacts['recommender'])

def validate_input_features(input_df, expected_features):
    """Validate that input features match expected features"""
    missing_features = [f for f in expected_features if f not in input_df.columns]
    if missing_features:
        # Add missing features with default values
        for feature in missing_features:
            if 'Processor' in feature:
                input_df[feature] = np.nan
            elif 'Graphics' in feature:
                input_df[feature] = 0
            else:
                input_df[feature] = 'Unknown'
    
    return input_df

def get_smart_defaults(df, brand, processor_brand):
    """Get smart default values based on brand and processor"""
    defaults = {
        'Core_per_processor': 4,
        'Threads': 8, 
        'Graphics_GB': 4,
        'Storage_type': 'SSD',
        'Operating_system': 'Windows 11'
    }
    
    # Filter by brand and processor for better defaults
    filtered = df[
        (df['Brand'] == brand) & 
        (df['Processor_brand'] == processor_brand)
    ]
    
    if not filtered.empty:
        defaults.update({
            'Core_per_processor': filtered['Core_per_processor'].median(),
            'Threads': filtered['Threads'].median(),
            'Graphics_GB': filtered['Graphics_GB'].median(),
            'Storage_type': filtered['Storage_type'].mode().iloc[0] if not filtered['Storage_type'].mode().empty else 'SSD',
            'Operating_system': filtered['Operating_system'].mode().iloc[0] if not filtered['Operating_system'].mode().empty else 'Windows 11'
        })
    
    # Ensure numeric values are not NaN
    for key in ['Core_per_processor', 'Threads', 'Graphics_GB']:
        if pd.isna(defaults[key]):
            defaults[key] = 4 if key == 'Core_per_processor' else (8 if key == 'Threads' else 4)
    
    return defaults# app/app_utils.py
import streamlit as st
import pandas as pd  # ADD THIS
import numpy as np   # ADD THIS
import joblib
import json
import sys
import os
from pathlib import Path

# Path Setup
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.persistence import load_model
from src.recommend.recommender import Recommender

# Paths
RAW_PATH = Path("data/raw/training_dataset.csv")
PROCESSED_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
FEATURES_PATH = Path("models/feature_names.json")

@st.cache_resource
def load_artifacts():
    """Load all required artifacts for the app without success messages"""
    artifacts = {
        'df': None,
        'pipeline': None, 
        'model': None,
        'feature_names': [],
        'recommender': None
    }
    
    # Load Data - silently without messages
    if PROCESSED_PATH.exists():
        artifacts['df'] = pd.read_csv(PROCESSED_PATH)
    elif RAW_PATH.exists():
        artifacts['df'] = pd.read_csv(RAW_PATH)
    else:
        return artifacts

    # Load Pipeline silently
    artifacts['pipeline'] = load_model(PIPELINE_PATH)

    # Load Model silently
    artifacts['model'] = load_model(MODEL_PATH)

    # Load Feature Names silently
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r") as f:
                artifacts['feature_names'] = json.load(f)
        except Exception:
            pass

    # Initialize Recommender silently
    try:
        artifacts['recommender'] = Recommender()
    except Exception:
        artifacts['recommender'] = None

    return (artifacts['df'], artifacts['pipeline'], artifacts['model'], 
            artifacts['feature_names'], artifacts['recommender'])

def validate_input_features(input_df, expected_features):
    """Validate that input features match expected features"""
    missing_features = [f for f in expected_features if f not in input_df.columns]
    if missing_features:
        # Add missing features with default values
        for feature in missing_features:
            if 'Processor' in feature:
                input_df[feature] = np.nan
            elif 'Graphics' in feature:
                input_df[feature] = 0
            else:
                input_df[feature] = 'Unknown'
    
    return input_df

def get_smart_defaults(df, brand, processor_brand):
    """Get smart default values based on brand and processor"""
    defaults = {
        'Core_per_processor': 4,
        'Threads': 8, 
        'Graphics_GB': 4,
        'Storage_type': 'SSD',
        'Operating_system': 'Windows 11'
    }
    
    # Filter by brand and processor for better defaults
    filtered = df[
        (df['Brand'] == brand) & 
        (df['Processor_brand'] == processor_brand)
    ]
    
    if not filtered.empty:
        defaults.update({
            'Core_per_processor': filtered['Core_per_processor'].median(),
            'Threads': filtered['Threads'].median(),
            'Graphics_GB': filtered['Graphics_GB'].median(),
            'Storage_type': filtered['Storage_type'].mode().iloc[0] if not filtered['Storage_type'].mode().empty else 'SSD',
            'Operating_system': filtered['Operating_system'].mode().iloc[0] if not filtered['Operating_system'].mode().empty else 'Windows 11'
        })
    
    # Ensure numeric values are not NaN
    for key in ['Core_per_processor', 'Threads', 'Graphics_GB']:
        if pd.isna(defaults[key]):
            defaults[key] = 4 if key == 'Core_per_processor' else (8 if key == 'Threads' else 4)
    
    return defaults