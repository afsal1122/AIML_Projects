# app/pages/1_Price_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.app_utils import load_artifacts, get_smart_defaults

st.set_page_config(
    page_title="AI Price Predictor", 
    page_icon="💲", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Laptop Price Predictor")
st.markdown("Get accurate price predictions using machine learning")

# Load artifacts
df, pipeline, model, feature_names, recommender = load_artifacts()

if df is None:
    st.error("❌ Dataset not available. Please check the setup.")
    st.stop()

# Prepare data
work_df = df.copy()
for col in work_df.columns:
    if work_df[col].dtype == 'object':
        work_df[col] = work_df[col].astype(str).str.strip()

# Processor generation parsing (consistent with preprocessing)
def parse_proc_gen(gen):
    if pd.isna(gen):
        return np.nan
    s = str(gen).strip().lower()
    
    # Apple M-series
    if s.startswith("m"):
        digits = ''.join(filter(str.isdigit, s))
        if digits:
            return 100 + int(digits)
        return np.nan
    
    # Intel/AMD generations
    digits = ''.join(filter(str.isdigit, s))
    if digits:
        return int(digits)
    
    # Special cases
    if "meteor" in s or "ultra" in s:
        return 14
        
    return np.nan

# Build intelligent options
st.markdown("### 🔧 Laptop Configuration")

# Brand Selection
col1, col2 = st.columns(2)
with col1:
    brands = sorted(work_df['Brand'].dropna().unique())
    selected_brand = st.selectbox("Laptop Brand", brands, key="brand")

# Processor Selection
with col2:
    proc_brands = sorted(work_df[work_df['Brand'] == selected_brand]['Processor_brand'].dropna().unique())
    if not proc_brands:
        proc_brands = sorted(work_df['Processor_brand'].dropna().unique())
    selected_proc_brand = st.selectbox("Processor Brand", proc_brands, key="proc_brand")

# Processor Model
proc_models = sorted(work_df[
    (work_df['Brand'] == selected_brand) & 
    (work_df['Processor_brand'] == selected_proc_brand)
]['Processor_name'].dropna().unique())

if not proc_models:
    proc_models = sorted(work_df[work_df['Processor_brand'] == selected_proc_brand]['Processor_name'].dropna().unique())

selected_proc_model = st.selectbox("Processor Model", proc_models, key="proc_model")

# Processor Generation - FIXED: Use available generation data
# Try to get generations from the dataset
try:
    # First try to get from the specific brand and model
    proc_gens = work_df[
        (work_df['Brand'] == selected_brand) & 
        (work_df['Processor_brand'] == selected_proc_brand) &
        (work_df['Processor_name'] == selected_proc_model)
    ]['Processor_gen_num'].dropna().unique()
    
    # If no specific generations found, try general ones for the model
    if len(proc_gens) == 0:
        proc_gens = work_df[
            (work_df['Processor_name'] == selected_proc_model)
        ]['Processor_gen_num'].dropna().unique()
    
    # If still no generations, use common ones
    if len(proc_gens) == 0:
        proc_gens = [8, 9, 10, 11, 12, 13, 14, 101, 102, 103]  # Common Intel/AMD + Apple M-series
    
    # Convert to display format
    gen_display_map = {}
    for gen in proc_gens:
        if gen >= 100:  # Apple M-series
            gen_display_map[f"M{gen-100}"] = gen
        else:  # Intel/AMD
            gen_display_map[f"{gen}th Gen"] = gen
    
    if gen_display_map:
        selected_gen_display = st.selectbox("Processor Generation", list(gen_display_map.keys()))
        selected_gen_num = gen_display_map[selected_gen_display]
        selected_gen = selected_gen_display
    else:
        selected_gen = st.text_input("Processor Generation", value="11th Gen", key="proc_gen_text")
        selected_gen_num = parse_proc_gen(selected_gen)
        
except Exception as e:
    st.warning(f"Could not load processor generations: {e}")
    selected_gen = st.text_input("Processor Generation", value="11th Gen", key="proc_gen_text_fallback")
    selected_gen_num = parse_proc_gen(selected_gen)

# GPU Selection
st.markdown("#### 🎮 Graphics Configuration")
col1, col2 = st.columns(2)

with col1:
    gpu_brands = sorted(work_df['Graphics_brand'].dropna().unique())
    selected_gpu_brand = st.selectbox("GPU Brand", gpu_brands, key="gpu_brand")

with col2:
    gpu_models = sorted(work_df[work_df['Graphics_brand'] == selected_gpu_brand]['Graphics_name'].dropna().unique())
    selected_gpu_model = st.selectbox("GPU Model", gpu_models, key="gpu_model")

# Memory & Storage
st.markdown("#### 💾 Memory & Storage")
col1, col2, col3 = st.columns(3)

with col1:
    ram_options = sorted(set(work_df['RAM_GB'].dropna().astype(int).unique()).union({4, 8, 16, 32, 64}))
    selected_ram = st.selectbox("RAM (GB)", ram_options, index=ram_options.index(16) if 16 in ram_options else 0)

with col2:
    storage_options = sorted(set(work_df['Storage_capacity_GB'].dropna().astype(int).unique()).union({256, 512, 1024, 2048}))
    selected_storage = st.selectbox("Storage (GB)", storage_options, 
                                  index=storage_options.index(512) if 512 in storage_options else 0)

with col3:
    storage_types = sorted(work_df['Storage_type'].dropna().unique())
    selected_storage_type = st.selectbox("Storage Type", storage_types, 
                                       index=storage_types.index('SSD') if 'SSD' in storage_types else 0)

# Display & OS
st.markdown("#### 🖥️ Display & System")
col1, col2, col3 = st.columns(3)

with col1:
    display_sizes = sorted(set(work_df['Display_size_inches'].dropna().astype(float).unique()).union({13.3, 14.0, 15.6, 16.0, 17.3}))
    selected_display = st.selectbox("Display Size (inches)", display_sizes,
                                  index=display_sizes.index(15.6) if 15.6 in display_sizes else 0)

with col2:
    resolution_options = ["1366x768", "1920x1080", "1920x1200", "2560x1440", "2560x1600", "3840x2160"]
    selected_resolution = st.selectbox("Resolution", resolution_options, index=1)

with col3:
    os_options = sorted(work_df['Operating_system'].dropna().unique())
    selected_os = st.selectbox("Operating System", os_options,
                             index=os_options.index('Windows 11') if 'Windows 11' in os_options else 0)

# Calculate PPI
h_px, v_px = map(int, selected_resolution.split('x'))
ppi = math.sqrt(h_px**2 + v_px**2) / selected_display

# Get smart defaults
defaults = get_smart_defaults(work_df, selected_brand, selected_proc_brand)

# Prediction Button
if st.button("🚀 Predict Price", type="primary", use_container_width=True):
    if model is None or pipeline is None:
        st.error("❌ AI model not available. Please train the model first.")
    else:
        with st.spinner("🔄 Analyzing configuration and predicting price..."):
            time.sleep(1)  # Simulate processing
            
            # Prepare input data -  Use correct column names
            input_data = {
                "Brand": selected_brand,
                "Processor_brand": selected_proc_brand,
                "Processor_name": selected_proc_model,
                "Processor_variant": "Standard",
                "Processor_gen": selected_gen,  # Keep original string for categorical
                "Processor_gen_num": selected_gen_num,  # Numeric version for model
                "Core_per_processor": defaults['Core_per_processor'],
                "Threads": defaults['Threads'],
                "RAM_GB": selected_ram,
                "Storage_capacity_GB": selected_storage,
                "Storage_type": selected_storage_type,
                "Graphics_name": selected_gpu_model,
                "Graphics_brand": selected_gpu_brand,
                "Graphics_GB": defaults['Graphics_GB'],
                "Display_size_inches": selected_display,
                "Horizontal_pixel": h_px,
                "Vertical_pixel": v_px,
                "ppi": ppi,
                "Operating_system": selected_os
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            try:
                # Transform and predict
                X_processed = pipeline.transform(input_df)
                pred_log = model.predict(X_processed)
                predicted_price = np.expm1(pred_log[0])
                
                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h1>₹ {predicted_price:,.0f}</h1>
                    <h3>Estimated Market Price</h3>
                    <p>Based on your configuration and current market trends</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show configuration summary
                st.markdown("### 📋 Configuration Summary")
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.write(f"**Brand:** {selected_brand}")
                    st.write(f"**Processor:** {selected_proc_brand} {selected_proc_model} {selected_gen}")
                    st.write(f"**RAM:** {selected_ram}GB")
                    st.write(f"**Storage:** {selected_storage}GB {selected_storage_type}")
                
                with config_col2:
                    st.write(f"**Graphics:** {selected_gpu_brand} {selected_gpu_model}")
                    st.write(f"**Display:** {selected_display}\" {selected_resolution}")
                    st.write(f"**OS:** {selected_os}")
                    st.write(f"**PPI:** {ppi:.1f}")
                
                # Price comparison
                similar_laptops = work_df[
                    (work_df['Brand'] == selected_brand) &
                    (work_df['RAM_GB'] == selected_ram) &
                    (work_df['Storage_capacity_GB'] == selected_storage)
                ]
                
                if not similar_laptops.empty:
                    avg_similar_price = similar_laptops['Price'].mean()
                    price_diff = ((predicted_price - avg_similar_price) / avg_similar_price) * 100
                    
                    st.markdown("### 📊 Price Analysis")
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        st.metric("AI Prediction", f"₹{predicted_price:,.0f}")
                    with comp_col2:
                        st.metric("Market Average", f"₹{avg_similar_price:,.0f}")
                    with comp_col3:
                        st.metric("Difference", f"{price_diff:+.1f}%", 
                                 delta=f"{price_diff:+.1f}%")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("💡 Try adjusting the configuration or check if the model is properly trained")

# Sidebar with tips
with st.sidebar:
    st.markdown("### 💡 Prediction Tips")
    st.info("""
    - **For accurate results**: Ensure all specifications match real product configurations
    - **Processor Generation**: Use exact format (e.g., '11th Gen', 'M1', 'Ryzen 5')
    - **GPU Memory**: Dedicated GPUs typically have 4GB+ VRAM
    - **Market Data**: Predictions are based on current market trends
    """)
    
    if model is not None:
        st.success("✅ AI Model: Ready")
    else:
        st.warning("⚠️ AI Model: Not Trained")
        
    if pipeline is not None:
        st.success("✅ Preprocessor: Ready")
    else:
        st.warning("⚠️ Preprocessor: Not Available")