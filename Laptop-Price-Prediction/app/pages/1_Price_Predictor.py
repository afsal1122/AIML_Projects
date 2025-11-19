import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from app.app_utils import load_artifacts
from src.data.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

st.set_page_config(page_title="Pro Price Estimator", page_icon="💲", layout="wide")

# Load all saved artifacts
df, pipeline, model, feature_names, _ = load_artifacts()

st.title("💲 Pro Laptop Price Estimator")
st.markdown("Configure detailed specs to get an accurate market price estimation.")

# -----------------------------
# Helper: Get unique dropdown options
# -----------------------------
def _opts_from(df_local, col):
    if col not in df_local.columns:
        return []
    vals = df_local[col].dropna().astype(str).unique()
    vals = [v for v in vals if v.strip() != ""]
    return sorted(vals)


# -----------------------------
# UI FORM
# -----------------------------
with st.form("predict_form"):
    st.subheader("1. Core Specs")
    col1, col2, col3 = st.columns(3)

    # -------- Brand --------
    with col1:
        brand_options = _opts_from(df, "brand")
        brand = st.selectbox("Brand", brand_options)
        df_brand = df[df["brand"].astype(str) == str(brand)]

    # -------- CPU Brand (Brand Filter Only) --------
    with col2:
        cpu_brand_options = _opts_from(df_brand, "cpu_brand")
        if not cpu_brand_options:
            cpu_brand_options = _opts_from(df, "cpu_brand")
        cpu_brand = st.selectbox("Processor Brand", cpu_brand_options)

    # -------- CPU Series (Brand Filter Only) --------
    with col3:
        series_options = _opts_from(df_brand, "cpu_series")
        if not series_options:
            series_options = _opts_from(df, "cpu_series")
        series = st.selectbox("Processor Series", series_options)

    # -------- CPU Gen + Variant (Brand Filter Only) --------
    col4, col5 = st.columns(2)

    with col4:
        if "cpu_gen" in df.columns:
            gen_options = _opts_from(df_brand, "cpu_gen")
            if not gen_options:
                gen_options = ["Standard"]
            cpu_gen = st.selectbox("Processor Generation", gen_options)
        else:
            cpu_gen = "Standard"

    with col5:
        if "cpu_variant" in df.columns:
            variant_options = _opts_from(df_brand, "cpu_variant")
            if not variant_options:
                variant_options = ["Standard"]
            cpu_variant = st.selectbox("Processor Variant", variant_options)
        else:
            cpu_variant = "Standard"

    st.divider()
    st.subheader("2. Graphics & Memory")

    col6, col7, col8 = st.columns(3)

    # -------- GPU (Brand Filter Only) --------
    with col6:
        gpu_options = _opts_from(df_brand, "gpu_model")
        if not gpu_options:
            gpu_options = _opts_from(df, "gpu_model")
        if not gpu_options:
            gpu_options = ["Unknown GPU"]
        gpu_model = st.selectbox("Graphics Card (GPU Model)", gpu_options)

    # -------- RAM (Global) --------
    with col7:
        ram_options = sorted(list(set(df["ram_gb"].dropna().astype(int).tolist())))
        default_ram = 16 if 16 in ram_options else ram_options[0]
        ram = st.select_slider("RAM (GB)", options=ram_options, value=default_ram)

    # -------- Storage (Global) --------
    with col8:
        storage_options = [128, 256, 512, 1024, 2048, 4096]
        storage = st.select_slider("Storage (GB)", options=storage_options, value=512)

    st.divider()
    st.subheader("3. Other Details")

    col9, col10 = st.columns(2)

    # -------- OS (Brand Filter Only) --------
    with col9:
        os_options = _opts_from(df_brand, "os_category")
        if not os_options:
            os_options = ["Other"]
        os_type = st.selectbox("Operating System", os_options)

    # -------- Screen Size (Brand Filter Only) --------
    with col10:
        screen_options = sorted(list(set(df_brand["display_size_in"].dropna().astype(float).tolist())))
        if not screen_options:
            screen_options = [15.6]
        screen_size = st.selectbox("Screen Size (Inches)", screen_options)

    # -------- Storage Type (Brand Lookup) --------
    storage_type_opts = _opts_from(df_brand, "storage_type")
    storage_type = storage_type_opts[0] if storage_type_opts else "SSD"

    submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    # Get GPU brand/type from dataset
    gpu_row = df[df["gpu_model"] == gpu_model]
    if not gpu_row.empty:
        gpu_brand_val = gpu_row.iloc[0]["gpu_brand"]
        gpu_type_val = gpu_row.iloc[0]["gpu_type"]
    else:
        gpu_brand_val = "NVIDIA" if "RTX" in gpu_model else "Intel"
        gpu_type_val = "Discrete" if "RTX" in gpu_model else "Integrated"

    # Build model input
    data = {
        "brand": brand,
        "cpu_brand": cpu_brand,
        "cpu_series": series,
        "cpu_gen": cpu_gen,
        "cpu_variant": cpu_variant,
        "ram_gb": ram,
        "storage_gb": storage,
        "storage_type": storage_type,
        "gpu_brand": gpu_brand_val,
        "gpu_model": gpu_model,
        "gpu_type": gpu_type_val,
        "os_category": os_type,
        "display_size_in": screen_size,

        # Defaults
        "weight_kg": 1.8,
        "ppi": 141,
        "age_years": 1,
        "user_rating": 4.5,
        "cpu_score": 5,
        "is_gaming": 0,
        "is_ultrabook": 0,
    }

    input_df = pd.DataFrame([data])

    # Keep only valid columns
    valid_cols = [c for c in input_df.columns if c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    X_pred = input_df[valid_cols]

    try:
        with st.spinner("Calculating market value..."):
            X_proc = pipeline.transform(X_pred)
            log_pred = model.predict(X_proc)
            price = np.expm1(log_pred)[0]

            st.success(f"### 💎 Estimated Price: ₹ {price:,.0f}")

            if st.checkbox("Show Prediction Details"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_proc)
                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = base_val[0]

                st.pyplot(
                    shap.force_plot(
                        base_val,
                        shap_values[0],
                        X_pred,
                        feature_names=feature_names,
                        matplotlib=True,
                    )
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
