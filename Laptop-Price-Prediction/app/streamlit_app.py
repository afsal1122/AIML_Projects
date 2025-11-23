import streamlit as st
import textwrap   
from app.app_utils import load_artifacts

st.set_page_config(
    page_title="Laptop Price Intelligence", 
    page_icon="💻", 
    layout="wide"
)

# -----------------------------
# CSS Styling (Updated for Dark Dashboard Theme)
# -----------------------------
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0f1116;
        color: white;
    }

    /* Main Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: left;
        margin-bottom: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    /* Metric Card Styling (To match the colored boxes) */
    .metric-card {
        background-color: #1c1e26;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #2d303e;
        position: relative;
    }
    .metric-card h4 {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin: 0;
        font-weight: 400;
    }
    .metric-card p {
        font-size: 1.8rem;
        color: #ffffff;
        margin: 5px 0 0 0;
        font-weight: 700;
    }
    /* Colored borders for metrics */
    .border-blue { border-left: 4px solid #3498db; }
    .border-orange { border-left: 4px solid #e67e22; }
    .border-purple { border-left: 4px solid #9b59b6; }
    .border-green { border-left: 4px solid #2ecc71; }

    /* Filter Section Styling */
    .filter-container {
        background-color: #1c1e26;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid #2d303e;
    }

    /* Result Card Styling (Dark Theme) */
    .recommendation-card {
        background: #232631; /* Dark Grey/Blue */
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #343846;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        height: 100%;
        color: white;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid #343846;
        padding-bottom: 0.8rem;
    }
    
    .brand-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Specs Grid */
    .specs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    
    .spec-box {
        background-color: #181a20;
        padding: 8px;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
    }
    
    .spec-label {
        font-size: 0.7rem;
        color: #8b92a5;
        text-transform: uppercase;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .spec-value {
        font-size: 0.9rem;
        color: #e1e1e1;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Footer Buttons */
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }
    
    .price-btn {
        background-color: #00c853;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .match-btn {
        background-color: #ff9100; /* Orange */
        color: #212121;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Customizing Streamlit Widgets */
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #e1e1e1 !important;
    }
    
    /* Green primary button */
    div.stButton > button[kind="primary"] {
        background-color: #27ae60 !important;
        border-color: #27ae60 !important;
        color: white !important;
        height: 3rem; 
        margin-top: 1.8rem; /* Align with inputs */
    }
    
</style>
""", unsafe_allow_html=True)

# Load artifacts
with st.spinner("Loading laptop intelligence system..."):
    df, pipeline, model, feature_names, recommender = load_artifacts()

if df is None:
    st.error("🚫 Unable to load dataset.")
    st.stop()

# -----------------------------
# Header Section
# -----------------------------
st.markdown('<div class="main-header">💻 Laptop Price Intelligence Platform</div>', unsafe_allow_html=True)

# -----------------------------
# Metrics Section (Custom HTML to match visual structure)
# -----------------------------
st.markdown("#### 📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card border-blue">
        <h4>Total Laptops</h4>
        <p>{len(df)}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card border-orange">
        <h4>Unique Brands</h4>
        <p>{df["Brand"].nunique()}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card border-purple">
        <h4>Avg Price</h4>
        <p>₹{int(df['Price'].mean()):,}</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    status = "Ready" if model else "Training"
    icon = "✅" if model else "❌"
    st.markdown(f"""
    <div class="metric-card border-green">
        <h4>AI Model</h4>
        <p>{icon} {status}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Smart Recommender Filter Section
# -----------------------------
st.markdown('### 🎯 Smart Recommender')
st.markdown('<div class="filter-container">', unsafe_allow_html=True)

# Layout based on reference: Brand | Usage | Min Slider | Max Slider
col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1])

with col1:
    all_brands = ["All Brands"] + sorted(df['Brand'].unique().tolist())
    selected_brand = st.selectbox("Select Brand", all_brands)

with col2:
    usage_type = st.selectbox(
        "Primary Usage",
        ["gaming", "programming", "content-creation", "lightweight", "general"],
        format_func=lambda x: {
            "gaming": "🎮 Gaming & Entertainment",
            "programming": "💻 Programming & Dev", 
            "content-creation": "🎨 Content Creation",
            "lightweight": "📱 Lightweight",
            "general": "⚡ General Purpose"
        }[x]
    )

with col3:
    data_min_price = int(df['Price'].min())
    data_max_price = int(df['Price'].max())
    min_price = st.slider("Minimum Budget (₹)", data_min_price, data_max_price, max(30000, data_min_price), 5000)

with col4:
    max_price = st.slider("Maximum Budget (₹)", data_min_price, data_max_price, min(120000, data_max_price), 5000)

# Secondary row for extras and the button
col_feat, col_btn = st.columns([3, 1])

with col_feat:
    must_haves = st.multiselect(
        "Must Have Features (Optional)",
        ["SSD", "Dedicated GPU", "High RAM (16GB+)", "Latest Processor"]
    )
    # Hidden top_k logic maintained but not prominent to match clean UI
    top_k = 8 

with col_btn:
    # Button aligned to right
    recommend_clicked = st.button("🎯 Find Smart Recommendations", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if min_price >= max_price:
    st.warning("⚠️ Maximum budget should be higher than minimum budget")

# -----------------------------
# Recommendation Logic & Results
# -----------------------------
if "recommendation_results" not in st.session_state:
    st.session_state["recommendation_results"] = None

results = st.session_state["recommendation_results"]

if recommend_clicked:
    if recommender is None:
        st.error("❌ Recommender system not available")
    else:
        with st.spinner("Finding best matches..."):
            results = recommender.recommend_with_brand(min_price, max_price, [usage_type], must_haves, selected_brand, top_k)
        st.session_state["recommendation_results"] = results
        results = st.session_state["recommendation_results"]

if results is not None:
    if results.empty:
        clear = st.button("🧹 Clear Output", type="secondary")
        if clear:
            st.session_state["recommendation_results"] = None
            st.rerun()
        st.markdown("""<h3 style="color: #7f8c8d; text-align:center;">🤔 No Perfect Matches Found</h3>""", unsafe_allow_html=True)
    else:
        # Clear button at top of results
        col_res_header, col_res_clear = st.columns([4,1])
        with col_res_clear:
            clear = st.button("🧹 Clear Output", type="secondary", use_container_width=True)
            if clear:
                st.session_state["recommendation_results"] = None
                st.rerun()

        # Grid Display
        cols = st.columns(2)
        for idx, (_, laptop) in enumerate(results.iterrows()):
            with cols[idx % 2]:
                score_percent = int(laptop['Score'] * 100) if 'Score' in laptop else 0
                processor = str(laptop.get('Processor_name', 'N/A')).replace('_', ' ').title()
                ram = laptop.get('RAM_GB', 'N/A')
                storage = laptop.get('Storage_capacity_GB', 'N/A')
                graphics = str(laptop.get('Graphics_name', 'Integrated')).replace('_', ' ').title()
                price = laptop.get('Price', 0)
                brand = laptop.get('Brand', 'Unknown')
                
                # HTML Card Construction
                card_html = f"""
                <div class="recommendation-card">
                    <div class="card-header">
                        <span class="brand-title">{brand}</span>
                        <div style="background: #374151; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                            Laptop
                        </div>
                    </div>
                    
                    <div class="specs-grid">
                        <div class="spec-box">
                            <span class="spec-label">🖥️ Processor</span>
                            <span class="spec-value">{processor}</span>
                        </div>
                        <div class="spec-box">
                            <span class="spec-label">💾 RAM</span>
                            <span class="spec-value">{ram} GB</span>
                        </div>
                        <div class="spec-box">
                            <span class="spec-label">💿 Storage</span>
                            <span class="spec-value">{storage} GB</span>
                        </div>
                        <div class="spec-box">
                            <span class="spec-label">🎮 Graphics</span>
                            <span class="spec-value">{graphics}</span>
                        </div>
                    </div>
                    
                    <div class="card-footer">
                        <div class="price-btn">₹ {price:,.0f}</div>
                        <div class="match-btn">Match: {score_percent}%</div>
                    </div>
                </div>
                """
                st.markdown(card_html.replace('\n', ''), unsafe_allow_html=True)