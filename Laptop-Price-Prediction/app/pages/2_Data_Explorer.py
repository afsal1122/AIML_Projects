import streamlit as st
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.app_utils import load_artifacts
from src.utils import (
    plot_brand_distribution, plot_avg_price_by_brand, 
    plot_correlation_heatmap, plot_ram_vs_price_scatter
)

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer (EDA)")

# --- Load Artifacts ---
artifacts = load_artifacts()
df = artifacts["data"]

if df is None:
    st.error("Data file not loaded. Cannot display EDA.")
    st.stop()

st.markdown(f"Visualizations from the dataset ({len(df)} laptops).")
st.markdown("These plots are based on the `laptops_cleaned.csv` file. They will update automatically if you re-run the full pipeline (scrape, preprocess, train).")

# --- Get Plots ---
try:
    plots = {
        "brand_dist": plot_brand_distribution(df),
        "price_by_brand": plot_avg_price_by_brand(df),
        "ram_vs_price": plot_ram_vs_price_scatter(df),
    }
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Filter out outlier columns for a cleaner heatmap
    cols_to_plot = [c for c in numeric_cols if c not in ['price', 'log_price'] and df[c].nunique() > 1 and df[c].nunique() < 50]
    plots["corr_heatmap"] = plot_correlation_heatmap(df, cols_to_plot)

    # --- Display Plots ---
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plots['brand_dist'])
        st.pyplot(plots['ram_vs_price'])
    with col2:
        st.pyplot(plots['price_by_brand'])

    st.pyplot(plots['corr_heatmap'])
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))

except Exception as e:
    st.error("An error occurred while generating plots.")
    st.exception(e)