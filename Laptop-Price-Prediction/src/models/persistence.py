# File: src/models/persistence.py
"""
Functions for saving and loading machine learning models
and other artifacts using joblib.
"""

import joblib
from pathlib import Path
from typing import Any
from src.utils import get_logger

logger = get_logger(__name__)

def save_model(model: Any, filepath: Path):
    """Saves a model or pipeline to a file using joblib."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}", exc_info=True)

def load_model(filepath: Path) -> Any:
    """Loads a model or pipeline from a file using joblib."""
    if not filepath.exists():
        logger.error(f"Model file not found at {filepath}")
        return None
    
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}", exc_info=True)
        return None