"""
Streamlit Web Application (v2)

This file creates the main web interface for the Laptop Price Predictor,
featuring:
- Live Deal Finder (scraper + model)
- Static Recommender
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app_utils import load_artifacts

# --- Page Configuration ---
st.set_page_config(
    page_title="Laptop Recommender",
    page_icon="💻",
    layout="wide",
)

# --- Load Artifacts ---
artifacts = load_artifacts()
df = artifacts["data"]
recommender = artifacts["recommender"]

# --- Check if artifacts loaded successfully ---
if df is None or recommender is None or not recommender.is_ready():
    st.error(
        "**Error: Critical artifacts are missing.**\n\n"
        "The app cannot run. Please check the file paths and ensure you have run the training pipeline:\n\n"
        "1.  `python -m src.data.preprocess`\n\n"
        "2.  `python -m src.models.train`"
    )
    st.stop()

# --- Main App ---
st.title("💻 Laptop Price Predictor & Recommender")
st.markdown("Find your perfect laptop using our AI model and live market data.")

# --- Live Deal Finder Section ---
with st.container(border=True):
    st.header("✨ Find Today's Good Deals (Live)")
    st.markdown(
        "This tool scrapes live laptop data, uses our AI model to predict what "
        "the price *should* be, and shows you the best deals."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        deal_store = st.selectbox("Select a store:", ["Flipkart", "Amazon"])
    with col2:
        brand_list = ["All Brands"] + sorted(df['brand'].unique())
        deal_brand = st.selectbox("Select a brand to search for deals:", brand_list)
    
    if st.button(f"Search for live {deal_brand} deals on {deal_store}", type="primary", use_container_width=True):
        query_brand = "laptop" if deal_brand == "All Brands" else deal_brand
        
        deals_df = pd.DataFrame()
        with st.spinner(f"Scraping live {query_brand} deals from {deal_store}... (This may take a moment)"):
            if deal_store == "Flipkart":
                deals_df = recommender.scrape_live_deals_flipkart(brand=query_brand, pages=1)
            elif deal_store == "Amazon":
                deals_df = recommender.scrape_live_deals_amazon(brand=query_brand, pages=1)
        
        if not deals_df.empty:
            st.subheader(f"Live Deals Found for: {query_brand} on {deal_store}")
            st.dataframe(
                deals_df,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank", width="small", format="%d."
                    ),
                    "model": st.column_config.TextColumn("Model", width="large"),
                    "price": st.column_config.NumberColumn(
                        "Live Price", format="₹%d"
                    ),
                    "predicted_price": st.column_config.NumberColumn(
                        "Predicted Price", format="₹%d"
                    ),
                    "discount_pct": st.column_config.ProgressColumn(
                        "Deal Score (Discount)",
                        help="Discount vs. predicted price (higher is better)",
                        min_value=0,
                        max_value=max(0.5, deals_df['discount_pct'].max()),
                        format="%.1f%%"
                    ),
                    "url": st.column_config.LinkColumn(
                        "Link", width="small", display_text="Buy"
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.error(f"Could not find any live deals for {query_brand} on {deal_store}. The scraper may be blocked, selectors are outdated, or no deals were found.")

st.divider()

# --- Static Recommendation Section ---
with st.container(border=True):
    st.header("Static Recommendations (from Dataset)")
    st.markdown("Find the best laptops from our pre-compiled dataset based on your budget and needs.")
    
    col1, col2 = st.columns(2)
    with col1:
        budget_range = st.slider(
            "Price Range (INR)", 
            min_value=int(df['price'].min()), 
            max_value=int(df['price'].max()),
            value=(30000, 80000),
            step=1000,
            key="static_budget"
        )
    with col2:
        use_cases = st.multiselect(
            "Select your primary use cases (up to 2):",
            options=['gaming', 'programming', 'content-creation', 'lightweight'],
            max_selections=2,
            key="static_use_case"
        )
    
    if st.button("Find Recommendations", use_container_width=True):
        with st.spinner("Searching database..."):
            recs = recommender.recommend_laptops(
                budget_min=budget_range[0],
                budget_max=budget_range[1],
                use_cases=use_cases,
                must_haves=[], # Simpler search
                top_k=5
            )
        
        st.subheader("Top 5 Recommendations from Dataset")
        if not recs.empty:
            st.dataframe(
                recs,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small", format="%d."),
                    "model": st.column_config.TextColumn("Model", width="large"),
                    "price": st.column_config.NumberColumn("Price (INR)", format="₹%d"),
                    "url": st.column_config.LinkColumn("Link", width="small", display_text="Buy"),
                    "recommend_score": st.column_config.NumberColumn("Score", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No laptops found matching your exact criteria.")