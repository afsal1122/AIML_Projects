# File: src/models/evaluate.py
"""
Model Evaluation Functions.

This module provides functions to calculate regression metrics
and generate evaluation plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional

from src.utils import get_logger, save_plot

logger = get_logger(__name__)

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression metrics."""
    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

def plot_pred_vs_actual(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = "Predicted vs. Actual Prices"
) -> plt.Figure:
    """Plots a scatter plot of predicted vs. actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return fig

def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    title: str = "Residuals Plot"
) -> plt.Figure:
    """Plots the residuals (y_true - y_pred) vs. predicted values."""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title(title)
    return fig

def plot_feature_importance(
    model: Any, 
    feature_names: list,
    model_name: str = "Model"
) -> plt.Figure:
    """
    Plots feature importances from a tree-based model.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} does not have 'feature_importances_' attribute.")
        return plt.Figure() # Return empty figure
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(indices)), 
        importances[indices], 
        align='center'
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance for {model_name}")
    return fig