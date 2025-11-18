import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.app_utils import load_artifacts
from src.data.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

# Handle optional SHAP import
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
st.set_page_config(page_title="Price Predictor", page_icon="🏷️", layout="wide")

st.title("🏷️ Laptop Price Predictor")
st.markdown("Build a virtual laptop configuration to get an estimated market price.")

# --- Load Artifacts ---
artifacts = load_artifacts()
df = artifacts["data"]
pipeline = artifacts["pipeline"]
model = artifacts["model"]
feature_names = artifacts["feature_names"]

if df is None or pipeline is None or model is None or feature_names is None:
    st.error("Critical artifacts are missing. Please wait or check the main page.")
    st.stop()

# --- Dependent Dropdowns (Suggestion #1) ---
# Use session state to manage selections
if 'brand' not in st.session_state:
    st.session_state.brand = df['brand'].unique()[0]

def update_brand():
    # Callback to update brand state
    st.session_state.brand = st.session_state.brand_key

# --- Input Form ---
with st.container(border=True):
    st.header("Build Your Laptop Configuration")

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        p_brand = st.selectbox(
            "Brand", 
            options=sorted(df['brand'].unique()), 
            key="brand_key",
            on_change=update_brand
        )
        
        # Filter DataFrame based on selected brand
        brand_df = df[df['brand'] == st.session_state.brand]
    
    with col2:
        p_os = st.selectbox(
            "Operating System", 
            options=sorted(brand_df['os_category'].unique()) # Dependent
        )
    with col3:
        p_storage_type = st.selectbox(
            "Storage Type", 
            options=sorted(brand_df['storage_type'].unique()) # Dependent
        )

    # Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        p_cpu_brand = st.selectbox(
            "CPU Brand", 
            options=sorted(brand_df['cpu_brand'].unique()) # Dependent
        )
    with col2:
        p_cpu_series = st.selectbox(
            "CPU Series", 
            options=sorted(brand_df['cpu_series'].unique()) # Dependent
        )
    with col3:
        p_gpu_brand = st.selectbox(
            "GPU Brand", 
            options=sorted(brand_df['gpu_brand'].unique()) # Dependent
        )

    # Row 3 (Sliders & Dropdowns)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ram_options = sorted(df['ram_gb'].unique())
        p_ram_gb = st.select_slider(
            "RAM (GB)", 
            options=ram_options, 
            value=16 if 16 in ram_options else ram_options[0]
        )
    with col2:
        storage_options = sorted([g for g in [128, 256, 512, 1024, 2048, 4096] if g >= df['storage_gb'].min() and g <= df['storage_gb'].max()])
        p_storage_gb = st.select_slider(
            "Storage (GB)", 
            options=storage_options,
            value=512 if 512 in storage_options else storage_options[0]
        )
    with col3:
        # Suggestion #4: Screen size as dropdown
        screen_options = sorted(brand_df['display_size_in'].unique())
        p_display_size = st.selectbox(
            "Screen Size (inches)",
            options=screen_options,
            index=0
        )
    with col4:
        # Suggestion #5: Weight slider dependent on brand
        min_w, max_w = float(brand_df['weight_kg'].min()), float(brand_df['weight_kg'].max())
        default_w = float(brand_df['weight_kg'].mean())
        if not min_w <= default_w <= max_w:
            default_w = min_w # Adjust default if mean is out of bounds
            
        p_weight_kg = st.slider(
            f"Weight ({p_brand} avg: {min_w:.1f}-{max_w:.1f} kg)", 
            min_value=min_w, 
            max_value=max_w, 
            value=default_w, 
            step=0.1
        )

    if st.button("Predict Price", type="primary", use_container_width=True):
        # --- Prepare Input Data for Model ---
        with st.spinner("Preparing features..."):
            # Get related features from brand/series selections
            brand_df = df[df['brand'] == st.session_state.brand]
            
            p_gpu_type = brand_df[brand_df['gpu_brand'] == p_gpu_brand]['gpu_type'].mode().iloc[0]
            
            p_ppi = brand_df[brand_df['display_size_in'] == p_display_size]['ppi'].mean()
            if pd.isna(p_ppi): p_ppi = df['ppi'].mean()
                
            p_age_years = 1.0 # Assume 1 year old
            p_user_rating = float(brand_df['user_rating'].mean())
            
            p_is_gaming = 1 if (p_gpu_type == 'Discrete' and p_ram_gb >= 16) else 0
            p_is_ultrabook = 1 if (p_weight_kg <= 1.4 and p_display_size <= 14.0) else 0
            
            p_cpu_score = float(brand_df[brand_df['cpu_series'] == p_cpu_series]['cpu_score'].mean())
            if pd.isna(p_cpu_score): p_cpu_score = df['cpu_score'].mean()

            # Create a DataFrame from the inputs
            input_data = {
                'ram_gb': [p_ram_gb], 'storage_gb': [p_storage_gb], 'display_size_in': [p_display_size],
                'weight_kg': [p_weight_kg], 'ppi': [p_ppi], 'age_years': [p_age_years],
                'user_rating': [p_user_rating], 'cpu_score': [p_cpu_score], 'brand': [st.session_state.brand],
                'storage_type': [p_storage_type], 'cpu_brand': [p_cpu_brand], 'cpu_series': [p_cpu_series],
                'gpu_brand': [p_gpu_brand], 'gpu_type': [p_gpu_type], 'os_category': [p_os],
                'is_gaming': [p_is_gaming], 'is_ultrabook': [p_is_ultrabook],
            }
            input_df = pd.DataFrame(input_data)
            
            # Ensure all columns are in the correct order for the pipeline
            model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
            input_df = input_df[model_features]


        with st.spinner("Running the magic model... ✨"):
            # 1. Transform data
            transformed_data = pipeline.transform(input_df)
            
            # 2. Predict (model expects log-price)
            log_price_pred = model.predict(transformed_data)
            
            # 3. Inverse transform to get actual price
            price_pred = np.expm1(log_price_pred)[0]

        # Display prediction
        st.metric(
            label="Predicted Laptop Price",
            value=f"₹ {price_pred:,.0f}"
        )
        st.caption("Note: This prediction is based on a model trained on historical data.")

        # --- SHAP Explanation (Suggestion #3) ---
        if SHAP_AVAILABLE:
            st.subheader("How the model made this prediction:")
            with st.spinner("Analyzing prediction factors..."):
                try:
                    # Create explainer
                    # Check for model type
                    if "XGB" in model.__class__.__name__:
                        explainer = shap.TreeExplainer(model, feature_names=feature_names)
                        shap_values = explainer.shap_values(transformed_data)
                        expected_value = explainer.expected_value
                    else: # RandomForest
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(transformed_data)
                        expected_value = explainer.expected_value[0] # RF has one per output
                    
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    shap.force_plot(
                        expected_value,
                        shap_values[0],
                        transformed_data[0],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                        text_rotation=15
                    )
                    st.pyplot(fig, bbox_inches='tight')
                    st.caption("Features pushing the price **higher** (red) or **lower** (blue).")
                    
                except Exception as e:
                    st.error(f"Could not generate SHAP plot: {e}")
                    st.exception(e) # Show full error for debugging
        else:
            st.info("Install `shap` (`pip install shap`) to see visual explanations.")