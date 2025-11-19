import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

def plot_pred_vs_actual(y_true, y_pred, title="Predicted vs Actual") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    return fig

def plot_residuals(y_true, y_pred, title="Residuals") -> plt.Figure:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance") -> plt.Figure:
    if not hasattr(model, 'feature_importances_'): return plt.Figure()
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(title)
    return fig