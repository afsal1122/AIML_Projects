# src/utils.py
import os
import logging
from pathlib import Path

# Define Project Root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "training_dataset.csv")
CLEAN_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "laptops_cleaned.csv")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Feature Definitions (Updated)
NUMERIC_FEATURES = [
    "Processor_gen_num",
    "Core_per_processor", 
    "Threads",
    "RAM_GB",
    "Storage_capacity_GB",
    "Graphics_GB",
    "Display_size_inches",
    "Horizontal_pixel",
    "Vertical_pixel",
    "ppi",
]

CATEGORICAL_FEATURES = [
    "Brand",
    "Processor_brand",
    "Processor_name", 
    "Processor_variant",
    "Storage_type",
    "Graphics_name",
    "Graphics_brand",
    "Operating_system",
]

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def ensure_directories():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)