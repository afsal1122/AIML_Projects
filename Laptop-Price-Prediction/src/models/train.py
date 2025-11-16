# File: src/models/train.py
"""
Model Training Pipeline.

This script loads the preprocessed data, splits it,
trains a RandomForest and an XGBoost model, performs
hyperparameter tuning, evaluates the best model, and saves
the model and evaluation artifacts.

Run as module:
python -m src.models.train
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Any, Optional

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from src.utils import get_logger
from src.models.persistence import load_model, save_model
from src.models.evaluate import (
    get_metrics, plot_pred_vs_actual, 
    plot_residuals, plot_feature_importance
)
from src.data.preprocess import (
    PROCESSED_DATA_PATH, PIPELINE_PATH, TARGET_FEATURE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)

# Handle optional XGBoost import
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

logger = get_logger(__name__)

# Define paths
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
EVAL_DIR = MODEL_DIR / "evaluation"

# --- Model Configuration ---

# RandomForest
RF_PARAM_DIST = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 1.0]
}

# XGBoost
XGB_PARAM_DIST = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# -----------------------------

def load_data(
    data_path: Path = PROCESSED_DATA_PATH,
    pipeline_path: Path = PIPELINE_PATH
) -> tuple:
    """Loads the processed data and the preprocessing pipeline."""
    if not data_path.exists():
        logger.error(f"Processed data not found at {data_path}.")
        logger.error("Please run the preprocessing script first:")
        logger.error("python -m src.data.preprocess")
        raise FileNotFoundError(f"Missing {data_path}")
        
    if not pipeline_path.exists():
        logger.error(f"Preprocessing pipeline not found at {pipeline_path}.")
        logger.error("Please run the preprocessing script first.")
        raise FileNotFoundError(f"Missing {pipeline_path}")

    df = pd.read_csv(data_path)
    pipeline = load_model(pipeline_path)
    
    if df.empty or pipeline is None:
        raise ValueError("Data or pipeline failed to load.")
        
    logger.info(f"Loaded {len(df)} records and preprocessing pipeline.")
    return df, pipeline

def get_feature_names(pipeline: ColumnTransformer) -> List[str]:
    """Extracts feature names from a fitted ColumnTransformer."""
    feature_names = []
    
    # Numeric features
    feature_names.extend(NUMERIC_FEATURES)
    
    # Categorical features
    try:
        ohe_categories = pipeline.named_transformers_['cat']\
                                 .named_steps['onehot']\
                                 .categories_
        for i, col in enumerate(CATEGORICAL_FEATURES):
            feature_names.extend([f"{col}_{cat}" for cat in ohe_categories[i]])
    except Exception as e:
        logger.warning(f"Could not extract OHE names, using defaults: {e}")
        feature_names.extend(CATEGORICAL_FEATURES)

    # Binary features
    feature_names.extend(BINARY_FEATURES)
    
    return feature_names

def tune_model(
    model: Any, 
    param_dist: dict, 
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> Any:
    """Performs RandomizedSearchCV for a given model."""
    logger.info(f"Starting tuning for {model.__class__.__name__}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Keep low for speed
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    random_search.fit(X_train, y_train)
    logger.info(f"Best params: {random_search.best_params_}")
    return random_search.best_estimator_

def main():
    logger.info("Starting model training pipeline...")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    try:
        df, pipeline = load_data()
    except FileNotFoundError:
        return

    # 2. Define X and y
    # Log-transform the target variable to handle skewness
    df[TARGET_FEATURE] = np.log1p(df[TARGET_FEATURE])
    
    X = df.drop(columns=[TARGET_FEATURE])
    y = df[TARGET_FEATURE]

    # 3. Split Data (Train/Validation/Test)
    # 70% Train, 15% Validation, 15% Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=(0.15/0.85), random_state=42
    )
    
    logger.info(f"Data split: Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)})")

    # 4. Apply Preprocessing
    logger.info("Applying preprocessing pipeline...")
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)
    
    # Get feature names after transformation
    feature_names = get_feature_names(pipeline)
    
    # 5. Train Models
    models_to_train = {
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            RF_PARAM_DIST
        )
    }
    
    if XGBOOST_AVAILABLE:
        models_to_train["XGBoost"] = (
            XGBRegressor(random_state=42, n_jobs=-1),
            XGB_PARAM_DIST
        )
    else:
        logger.warning("XGBoost not found. Skipping XGBoost training.")
        logger.warning("To install: pip install xgboost")

    best_model = None
    best_score = np.inf
    trained_models = {}

    for name, (model, params) in models_to_train.items():
        logger.info(f"--- Training {name} ---")
        # Tuned model
        # best_estimator = tune_model(model, params, X_train_processed, y_train)
        
        # For speed, we'll just fit the default model
        # Uncomment tune_model above for hyperparameter search
        logger.warning(f"Skipping hyperparameter tuning for {name} for speed.")
        best_estimator = model.fit(X_train_processed, y_train)

        # Evaluate on validation set
        y_val_pred = best_estimator.predict(X_val_processed)
        metrics = get_metrics(np.expm1(y_val), np.expm1(y_val_pred))
        trained_models[name] = best_estimator
        
        if metrics['rmse'] < best_score:
            best_score = metrics['rmse']
            best_model = best_estimator
            logger.info(f"New best model: {name} (RMSE: {best_score:.2f})")

    # 6. Evaluate Best Model on Test Set
    if best_model is None:
        logger.error("No models were trained successfully.")
        return
        
    logger.info(f"--- Final Evaluation of {best_model.__class__.__name__} on Test Set ---")
    y_test_pred = best_model.predict(X_test_processed)
    
    # Inverse transform predictions (from log scale)
    y_test_orig = np.expm1(y_test)
    y_test_pred_orig = np.expm1(y_test_pred)
    
    test_metrics = get_metrics(y_test_orig, y_test_pred_orig)
    logger.info(f"Test Metrics (Original Scale): {test_metrics}")

    # 7. Save Artifacts
    logger.info("Saving best model and evaluation plots...")
    save_model(best_model, BEST_MODEL_PATH)
    
    # Save plots
    fig_pred = plot_pred_vs_actual(y_test_orig, y_test_pred_orig, title="Test Set: Predicted vs. Actual")
    fig_pred.savefig(EVAL_DIR / "test_predicted_vs_actual.png")
    
    fig_res = plot_residuals(y_test_orig, y_test_pred_orig, title="Test Set: Residuals Plot")
    fig_res.savefig(EVAL_DIR / "test_residuals_plot.png")
    
    fig_imp = plot_feature_importance(best_model, feature_names, best_model.__class__.__name__)
    fig_imp.savefig(EVAL_DIR / "feature_importance.png")
    
    logger.info("Training pipeline finished successfully.")

if __name__ == "__main__":
    main()