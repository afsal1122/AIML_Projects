import joblib
from pathlib import Path
from typing import Any
from src.utils import get_logger

logger = get_logger(__name__)

def save_model(model: Any, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved to {filepath}")

def load_model(filepath: Path) -> Any:
    if not filepath.exists():
        return None
    return joblib.load(filepath)