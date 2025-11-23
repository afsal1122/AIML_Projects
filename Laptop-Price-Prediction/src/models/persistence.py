# src/models/persistence.py
import joblib
import os
from src.utils import get_logger

logger = get_logger("PERSISTENCE")

def save_model(model, path):
    """Save model with proper error handling"""
    try:
        joblib.dump(model, path)
        logger.info(f"✅ Model saved at {path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False

def load_model(path):
    """Load model with proper error handling"""
    if not os.path.exists(path):
        logger.error(f"❌ Model not found at {path}")
        return None
    try:
        model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None