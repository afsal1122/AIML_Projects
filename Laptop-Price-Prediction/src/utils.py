import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

def save_plot(fig: plt.Figure, path: str):
    try:
        fig.savefig(path, bbox_inches='tight', dpi=150)
    except Exception:
        pass

def plot_brand_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='brand', data=df, order=df['brand'].value_counts().index, ax=ax, palette='viridis')
    ax.set_title('Laptop Count by Brand')
    return fig

def plot_avg_price_by_brand(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price = df.groupby('brand')['price'].mean().sort_values(ascending=False)
    sns.barplot(y=avg_price.index, x=avg_price.values, ax=ax, palette='coolwarm')
    ax.set_title('Average Price by Brand')
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    return fig

def plot_ram_vs_price_scatter(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='ram_gb', y='price', hue='brand', data=df, alpha=0.6, ax=ax)
    ax.set_title('Price vs. RAM')
    return fig