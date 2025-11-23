# src/models/train.py
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import with fallback
try:
    from src.utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_logger
except ImportError:
    # Fallback definitions
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

logger = get_logger("TRAIN")

try:
    from xgboost import XGBRegressor
    XGB_OK = True
    logger.info("XGBoost available")
except ImportError:
    XGB_OK = False
    logger.info("XGBoost not available, using RandomForest")

DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
FEATURES_PATH = Path("models/feature_names.json")

def get_available_features(df, expected_features):
    """Get only the features that actually exist in the dataframe"""
    available_features = [f for f in expected_features if f in df.columns]
    missing_features = [f for f in expected_features if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    return available_features

def get_feature_names_from_pipeline(pipeline):
    """Extract feature names after preprocessing"""
    try:
        # Get numeric features that actually exist
        num_feats = [f for f in NUMERIC_FEATURES if hasattr(pipeline, 'feature_names_in_') and f in pipeline.feature_names_in_]
        
        # Get categorical feature names
        try:
            cat_transformer = pipeline.named_transformers_["cat"]
            ohe = cat_transformer.named_steps["onehot"]
            cat_feats = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        except:
            cat_feats = CATEGORICAL_FEATURES
            
        return num_feats + cat_feats
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

def evaluate(y_true, y_pred):
    """Comprehensive model evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2), 
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }

def main():
    logger.info("Starting model training...")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
    
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {PIPELINE_PATH}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    if df.empty:
        raise ValueError("Dataset is empty after loading")
    
    # Get only available features
    available_numeric = get_available_features(df, NUMERIC_FEATURES)
    available_categorical = get_available_features(df, CATEGORICAL_FEATURES)
    
    logger.info(f"Available numeric features: {available_numeric}")
    logger.info(f"Available categorical features: {available_categorical}")
    
    if not available_numeric and not available_categorical:
        raise ValueError("No features available for training!")

    X = df[available_numeric + available_categorical]
    y_raw = df["Price"].astype(float).values

    # Log-transform target for better performance
    y = np.log1p(y_raw)

    # Load preprocessing pipeline
    pipeline = joblib.load(PIPELINE_PATH)
    
    # Transform features
    X_proc = pipeline.transform(X)
    
    # Get feature names
    feature_names = get_feature_names_from_pipeline(pipeline)
    logger.info(f"Transformed features: {X_proc.shape[1]} dimensions")

    # Train-test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_proc, y, df.index, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Model selection and hyperparameter tuning
    if XGB_OK:
        logger.info("Using XGBoost Regressor")
        model = XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)
        param_dist = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.8, 0.9]
        }
    else:
        logger.info("Using RandomForest Regressor") 
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        param_dist = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

    # Randomized search for hyperparameter tuning
    search = RandomizedSearchCV(
        model, param_distributions=param_dist, 
        n_iter=4, cv=3, n_jobs=-1, verbose=1, 
        random_state=42, scoring='neg_mean_squared_error'
    )
    
    logger.info("Starting hyperparameter tuning...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")

    # Evaluate model
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)

    metrics = evaluate(y_test_orig, y_pred)
    logger.info(f"Test metrics: {metrics}")

    # Save model and artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Feature names saved to {FEATURES_PATH}")
    
    # Print some predictions vs actual
    sample_size = min(5, len(y_test_orig))
    logger.info("Sample predictions vs actual:")
    for i in range(sample_size):
        logger.info(f"  Actual: ₹{y_test_orig[i]:,.0f}, Predicted: ₹{y_pred[i]:,.0f}")

    logger.info("🎯 Training completed successfully!")

if __name__ == "__main__":
    main()