# src/models/evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def get_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Robust MAPE calculation
    try:
        # Avoid division by zero
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
    except:
        mape = 0
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": f"{round(mape, 2)}%", 
        "R2 Score": round(r2, 4),
        "Mean Actual Price": round(np.mean(y_true), 2),
        "Mean Predicted Price": round(np.mean(y_pred), 2)
    }

def plot_pred_vs_actual(y_true, y_pred, model_name="Model"):
    """
    Scatter plot of predicted vs actual values with ideal line.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate metrics for plot
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Create scatter plot
    scatter = sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None, color="#0099ff")
    
    # Ideal line (Perfect prediction)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', 
             linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Price (₹)", fontsize=12)
    plt.ylabel("Predicted Price (₹)", fontsize=12)
    plt.title(f"Actual vs Predicted Price\n{model_name} (R² = {r2:.3f}, MAE = {mae:.0f})", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_residuals(y_true, y_pred):
    """
    Plot residuals for model diagnostics.
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, color="#ff6b6b")
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel("Predicted Price")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    ax1.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax2.hist(residuals, bins=30, alpha=0.7, color="#4ecdc4", edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return None
        
    importances = model.feature_importances_
    
    # Create dataframe and sort
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(df_imp)), df_imp["importance"], color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, df_imp["importance"])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=10)
    
    plt.yticks(range(len(df_imp)), df_imp["feature"])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title(f"Top {top_n} Features Driving Laptop Price", fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_price_distribution(y_true, y_pred):
    """
    Compare distribution of actual vs predicted prices.
    """
    plt.figure(figsize=(12, 6))
    
    plt.hist(y_true, bins=30, alpha=0.7, label='Actual Prices', color='blue', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted Prices', color='red', edgecolor='black')
    
    plt.xlabel("Price (₹)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution: Actual vs Predicted Prices", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()