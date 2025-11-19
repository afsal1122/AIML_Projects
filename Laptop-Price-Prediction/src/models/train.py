import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.utils import get_logger, save_plot
from src.models.persistence import load_model, save_model
from src.models.evaluate import get_metrics, plot_pred_vs_actual, plot_feature_importance
from src.data.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

logger = get_logger(__name__)
DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
FEAT_PATH = Path("models/feature_names.json")

def main():
    logger.info("Starting training...")
    if not DATA_PATH.exists(): return
    
    df = pd.read_csv(DATA_PATH)
    pipeline = load_model(PIPELINE_PATH)
    
    # Extract feature names for SHAP
    ohe = pipeline.named_transformers_['cat'].named_steps['onehot']
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names + BINARY_FEATURES
    with open(FEAT_PATH, 'w') as f: json.dump(feature_names, f)
    
    # Prepare Training Data
    X_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    X = df[X_cols]
    y = np.log1p(df['price']) # Log transform for better accuracy on prices
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    
    # Model Selection & Tuning
    if XGB_AVAILABLE:
        logger.info("Training XGBoost (High Accuracy Mode)...")
        model = XGBRegressor(n_jobs=-1, random_state=42)
        param_dist = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    else:
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    
    # Hyperparameter Tuning
    search = RandomizedSearchCV(
        model, param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1
    )
    search.fit(X_train_proc, y_train)
    best_model = search.best_estimator_
    logger.info(f"Best params: {search.best_params_}")
    
    # Evaluate
    preds = best_model.predict(X_test_proc)
    # Convert back from log scale for real metrics
    metrics = get_metrics(np.expm1(y_test), np.expm1(preds))
    logger.info(f"Test Metrics (Real Price): {metrics}")
    
    # Save
    save_model(best_model, MODEL_PATH)
    
    # Plots
    p1 = plot_pred_vs_actual(np.expm1(y_test), np.expm1(preds))
    save_plot(p1, "models/pred_vs_actual.png")
    p2 = plot_feature_importance(best_model, feature_names)
    save_plot(p2, "models/feature_importance.png")

if __name__ == "__main__":
    main()