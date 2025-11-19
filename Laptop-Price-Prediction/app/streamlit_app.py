import streamlit as st
from app_utils import load_artifacts

# --- CHANGED APP NAME ---
st.set_page_config(page_title="Laptop Genius", page_icon="💻", layout="wide")

st.title("💻 Laptop Genius")
st.markdown("### Intelligent Laptop Price Discovery & Recommendations")
st.info("Navigate using the sidebar to access the **Pro Price Estimator**, **Smart Recommender**, and **Market Analytics**.")

# Custom CSS for Card Styling
st.markdown("""
<style>
    .laptop-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        height: 100%;
    }
    .card-title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
        white-space: nowrap; 
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-price {
        font-size: 22px;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 10px;
    }
    .card-specs {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    .card-score {
        font-size: 12px;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 8px;
        border-radius: 12px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

df, _, _, _, rec = load_artifacts()

if df is None:
    st.error("System not initialized. Please run: python -m src.data.preprocess && python -m src.models.train")
    st.stop()

# --- Quick Dashboard ---
with st.container(border=True):
    st.header("⚡ Quick Recommender")
    st.caption("Get instant recommendations based on your budget.")
    
    col1, col2 = st.columns(2)
    with col1:
        min_p = int(df['price'].min())
        max_p = int(df['price'].max())
        budget = st.slider("Your Budget (₹)", min_p, max_p, (40000, 100000), step=1000)
        
    with col2:
        usage = st.multiselect("What will you use it for?", ["gaming", "programming", "lightweight", "content-creation"])
        must = st.multiselect("Must Haves", ["Dedicated GPU", "SSD"])

    find_btn = st.button("Find Matches", type="primary", use_container_width=True)

    if find_btn:
        recs = rec.recommend(budget[0], budget[1], usage, must, top_k=6) # Get 6 for a nice grid
        
        if not recs.empty:
            st.subheader(f"Found {len(recs)} Perfect Matches")
            st.markdown("---")

            # Create a grid layout (3 columns wide)
            cols = st.columns(3)
            
            for idx, row in recs.iterrows():
                # Cycle through columns (0, 1, 2, 0, 1, 2...)
                with cols[idx % 3]:
                    with st.container(border=True):
                        # Rank Badge
                        st.markdown(f"**Rank #{row['Rank']}**")
                        
                        # Title (Truncated if too long)
                        model_name = f"{row['brand']} {row['model']}"
                        st.markdown(f"### {row['brand']}")
                        st.caption(row['model'])
                        
                        # Price
                        st.markdown(f"## ₹{row['price']:,.0f}")
                        
                        # Specs
                        st.markdown(f"**CPU:** {row['cpu_series']}")
                        st.markdown(f"**RAM:** {row['ram_gb']} GB")
                        
                        # Progress Bar for Match Score
                        score_val = row['Score']
                        # Normalize score visually for the progress bar (assuming max score around 15-20)
                        norm_score = min(1.0, score_val / 15.0) 
                        st.progress(norm_score, text=f"Match Score: {score_val:.1f}")
                        
                        # Link Button
                        st.link_button("🔍 Search on Google", row['url'], use_container_width=True)
        else:
            st.warning("No laptops found in this range. Try increasing your budget.")

st.divider()
st.markdown("#### 📈 Market Snapshot")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Laptops in Database", len(df))
col_a.metric("Average Market Price", f"₹{int(df['price'].mean()):,}")
col_b.metric("Most Common Brand", df['brand'].mode()[0])
col_b.metric("Most Common CPU", df['cpu_series'].mode()[0])
col_c.metric("Most Common GPU", df['gpu_model'].mode()[0])
col_c.metric("Most Common RAM", f"{int(df['ram_gb'].mode()[0])} GB")