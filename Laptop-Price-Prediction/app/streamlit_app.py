# File: app/streamlit_app.py
"""
Streamlit Web Application

This file creates the main web interface for the Laptop Price Predictor,
including:
- Price Prediction tab
- EDA tab
- Recommendation tab
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time

from src.models.persistence import load_model
from src.recommend.recommender import Recommender
from src.utils import (
    plot_brand_distribution, plot_avg_price_by_brand, 
    plot_correlation_heatmap, plot_ram_vs_price_scatter
)

# Handle optional SHAP import
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Paths ---
DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")
MODEL_PATH = Path("models/best_model.pkl")
EDA_PLOTS_DIR = Path("models/evaluation")

# --- Caching ---
@st.cache_resource
def load_artifacts():
    """
    Loads the model, pipeline, and data.
    Uses st.cache_resource to load only once.
    """
    if not DATA_PATH.exists() or not PIPELINE_PATH.exists() or not MODEL_PATH.exists():
        return None, None, None, None

    data = pd.read_csv(DATA_PATH)
    pipeline = load_model(PIPELINE_PATH)
    model = load_model(MODEL_PATH)
    
    # Initialize recommender
    recommender = Recommender()
    
    return data, pipeline, model, recommender

@st.cache_data
def get_eda_plots(_df):
    """
    Generates and caches EDA plots.
    The _df argument is used to bust the cache if data changes.
    """
    plots = {
        "brand_dist": plot_brand_distribution(_df),
        "price_by_brand": plot_avg_price_by_brand(_df),
        "ram_vs_price": plot_ram_vs_price_scatter(_df),
    }
    numeric_cols = _df.select_dtypes(include=np.number).columns.tolist()
    plots["corr_heatmap"] = plot_correlation_heatmap(_df, numeric_cols)
    return plots

# --- Load Data ---
df, pipeline, model, recommender = load_artifacts()

if df is None or pipeline is None or model is None or not recommender.is_ready():
    st.error(
        "**Error: Missing critical model artifacts.**\n\n"
        "The app cannot run without the following files:\n"
        f"1. `{DATA_PATH}`\n"
        f"2. `{PIPELINE_PATH}`\n"
        f"3. `{MODEL_PATH}`\n\n"
        "Please run the training pipeline first:\n"
        "`python -m src.data.preprocess`\n"
        "`python -m src.models.train`"
    )
    st.stop()

# --- App Layout ---
st.title("💻 Laptop Price Predictor & Recommender")
st.markdown("Predict prices, explore data, and get laptop recommendations.")

tab1, tab2, tab3 = st.tabs([
    "**1. Price Predictor**", 
    "**2. Data Explorer (EDA)**", 
    "**3. Laptop Recommender**"
])


# =======================================================================
# TAB 1: PRICE PREDICTOR
# =======================================================================
with tab1:
    st.header("Predict a Laptop's Price")
    
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("Build Your Laptop 🛠️")
        
        # Get unique values from the dataset for dropdowns
        brands = sorted(df['brand'].unique())
        cpu_brands = sorted(df['cpu_brand'].unique())
        cpu_series_list = sorted(df['cpu_series'].unique())
        gpu_brands = sorted(df['gpu_brand'].unique())
        gpu_types = sorted(df['gpu_type'].unique())
        os_list = sorted(df['os_category'].unique())
        storage_types = sorted(df['storage_type'].unique())

        # --- Create Input Fields ---
        p_brand = st.selectbox("Brand", brands)
        p_os = st.selectbox("Operating System", os_list)
        
        col1, col2 = st.columns(2)
        with col1:
            p_cpu_brand = st.selectbox("CPU Brand", cpu_brands)
            p_gpu_brand = st.selectbox("GPU Brand", gpu_brands)
            p_storage_type = st.selectbox("Storage Type", storage_types)
            
        with col2:
            p_cpu_series = st.selectbox("CPU Series", cpu_series_list)
            p_gpu_type = st.selectbox("GPU Type", gpu_types)
            p_storage_gb = st.select_slider(
                "Storage (GB)", 
                options=[128, 256, 512, 1024, 2048, 4096], 
                value=512
            )
        
        p_ram_gb = st.select_slider(
            "RAM (GB)", 
            options=[4, 8, 12, 16, 24, 32, 64], 
            value=16
        )
        
        p_display_size = st.slider(
            "Screen Size (inches)", 
            min_value=11.0, max_value=18.0, value=15.6, step=0.1
        )
        p_weight_kg = st.slider(
            "Weight (kg)", 
            min_value=0.8, max_value=4.0, value=1.5, step=0.1
        )
        
        # Dummy inputs for features the pipeline needs but user might not set
        p_ppi = 141.2 # Default PPI for 15.6" 1080p
        p_age_years = 1 # Assume 1 year old
        p_user_rating = 4.0 # Assume 4.0 rating
        p_is_gaming = 1 if (p_gpu_type == 'Discrete' and p_ram_gb >= 16) else 0
        p_is_ultrabook = 1 if (p_weight_kg <= 1.4 and p_display_size <= 14.0) else 0
        p_cpu_score = 10 # Placeholder
        

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Price", type="primary", use_container_width=True):
        # Create a DataFrame from the inputs
        input_data = {
            'ram_gb': [p_ram_gb],
            'storage_gb': [p_storage_gb],
            'display_size_in': [p_display_size],
            'weight_kg': [p_weight_kg],
            'ppi': [p_ppi],
            'age_years': [p_age_years],
            'user_rating': [p_user_rating],
            'cpu_score': [p_cpu_score],
            'brand': [p_brand],
            'storage_type': [p_storage_type],
            'cpu_brand': [p_cpu_brand],
            'cpu_series': [p_cpu_series],
            'gpu_brand': [p_gpu_brand],
            'gpu_type': [p_gpu_type],
            'os_category': [p_os],
            'is_gaming': [p_is_gaming],
            'is_ultrabook': [p_is_ultrabook],
        }
        input_df = pd.DataFrame(input_data)

        with st.spinner("Running the magic model... ✨"):
            time.sleep(1) # Simulate work
            
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

            # --- SHAP Explanation ---
            if SHAP_AVAILABLE:
                st.subheader("How the model made this prediction:")
                try:
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    
                    # Get SHAP values
                    shap_values = explainer.shap_values(transformed_data)
                    
                    # Get feature names from pipeline
                    feature_names = pipeline.get_feature_names_out()

                    # Create a force plot
                    fig, ax = plt.subplots(figsize=(10, 3))
                    shap.force_plot(
                        explainer.expected_value,
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
            else:
                st.info(
                    "Install `shap` (`pip install shap`) to see "
                    "visual explanations for each prediction."
                )

# =======================================================================
# TAB 2: DATA EXPLORER (EDA)
# =======================================================================
with tab2:
    st.header("Explore the Laptop Market")
    st.markdown(f"Visualizations from the dataset ({len(df)} laptops).")
    
    # Get cached plots
    plots = get_eda_plots(df)
    
    st.pyplot(plots['brand_dist'])
    st.pyplot(plots['price_by_brand'])
    st.pyplot(plots['ram_vs_price'])
    st.pyplot(plots['corr_heatmap'])
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))


# =======================================================================
# TAB 3: LAPTOP RECOMMENDER
# =======================================================================
with tab3:
    st.header("Find Your Perfect Laptop")
    
    # --- Recommendation Inputs ---
    st.subheader("1. What's your budget?")
    budget_range = st.slider(
        "Price Range (INR)", 
        min_value=int(df['price'].min()), 
        max_value=int(df['price'].max()),
        value=(30000, 80000),
        step=1000
    )
    
    st.subheader("2. What's your primary use case?")
    use_cases = st.multiselect(
        "Select up to two use cases:",
        options=['gaming', 'programming', 'content-creation', 'lightweight'],
        max_selections=2
    )

    st.subheader("3. Any must-haves?")
    must_haves = st.multiselect(
        "Select non-negotiable features:",
        options=['Dedicated GPU', 'SSD']
    )
    
    # --- Recommendation Logic ---
    if st.button("Find Laptops", type="primary"):
        with st.spinner("Searching for recommendations..."):
            recs = recommender.recommend_laptops(
                budget_min=budget_range[0],
                budget_max=budget_range[1],
                use_cases=use_cases,
                must_haves=must_haves,
                top_k=5
            )
        
        st.subheader("Top 5 Recommendations")
        if not recs.empty:
            st.dataframe(recs.style.format({
                'price': '₹{:.0f}',
                'weight_kg': '{:.2f} kg',
                'recommend_score': '{:.2f}'
            }))
            
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(recs)
            st.download_button(
                label="Download Recommendations as CSV",
                data=csv_data,
                file_name="laptop_recommendations.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "No laptops found matching your exact criteria. "
                "Try broadening your budget or removing a 'must-have' filter."
            )
            
    # --- Good Deals Section ---
    st.divider()
    st.header("✨ Today's Good Deals")
    st.markdown(
        "These are laptops from our dataset priced at least 15% "
        "**below** their model-predicted value."
    )
    
    with st.spinner("Finding deals..."):
        deals = recommender.find_good_deals(discount_threshold=0.15)

    if not deals.empty:
        st.dataframe(
            deals.head(5).style.format({
                'price': '₹{:.0f}',
                'predicted_price': '₹{:.0f}',
                'discount_pct': '{:.1%}'
            }),
           column_config={
            "discount_pct": st.column_config.ProgressColumn(
                "Discount",
                help="Percentage below predicted price",
                min_value=0,  # <-- FIX
                max_value=float(deals['discount_pct'].max()), # <-- FIX
                format="%.1f%%"
            )
        }
        )
    else:
        st.info("No significant deals found in the current dataset.")