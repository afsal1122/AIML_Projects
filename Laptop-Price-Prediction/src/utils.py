# File: src/utils.py
"""
Utility functions for logging, file I/O, and plotting.
"""

import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.propagate = False
    return logger

logger = get_logger(__name__)

def save_plot(fig: plt.Figure, path: str):
    """Saves a matplotlib figure to a file."""
    try:
        fig.savefig(path, bbox_inches='tight', dpi=150)
        logger.info(f"Plot saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {path}: {e}")

def plot_brand_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of laptop brands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='brand', data=df, order=df['brand'].value_counts().index, ax=ax)
    ax.set_title('Laptop Count by Brand')
    ax.set_xlabel('Count')
    ax.set_ylabel('Brand')
    return fig

def plot_avg_price_by_brand(df: pd.DataFrame) -> plt.Figure:
    """Plots the average price by laptop brand."""
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price = df.groupby('brand')['price'].mean().sort_values(ascending=False)
    sns.barplot(y=avg_price.index, x=avg_price.values, ax=ax)
    ax.set_title('Average Price by Brand')
    ax.set_xlabel('Average Price (INR)')
    ax.set_ylabel('Brand')
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> plt.Figure:
    """Plots the correlation heatmap for numeric features."""
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features')
    return fig

def plot_ram_vs_price_scatter(df: pd.DataFrame) -> plt.Figure:
    """Plots a scatter plot of RAM vs. Price."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='ram_gb', y='price', data=df, alpha=0.6, ax=ax)
    ax.set_title('Price vs. RAM')
    ax.set_xlabel('RAM (GB)')
    ax.set_ylabel('Price (INR)')
    return fig