# app/pages/2_Data_Explorer.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd  
from app.app_utils import load_artifacts

st.set_page_config(page_title="Market Insights", page_icon="📊", layout="wide")

df, _, _, _, rec = load_artifacts()

if df is None:
    st.error("Data not found.")
    st.stop()

st.title("📊 Market Insights")

with st.container():
    # Filters
    st.markdown("#### Filter Data")
    col_f1, col_f2 = st.columns(2)
    f_brands = col_f1.multiselect("Select Brands", df['Brand'].unique(), default=df['Brand'].unique()[:5])
    
    # Filtered DF
    dff = df[df['Brand'].isin(f_brands)]
    
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        st.subheader("Average Price by Brand")
        avg_price = dff.groupby("Brand")["Price"].mean().reset_index()
        fig = px.bar(avg_price, x="Brand", y="Price", color="Brand", text_auto='.2s', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with row1_2:
        st.subheader("Price vs Performance (RAM)")
        dff['RAM_Num'] = pd.to_numeric(dff['RAM_GB'], errors='coerce')
        fig2 = px.box(dff, x="RAM_Num", y="Price", color="Brand", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Processor Market Share (in Dataset)")
    fig3 = px.sunburst(dff, path=['Processor_brand', 'Processor_name'], values='Price', title="Market Segment by CPU")
    st.plotly_chart(fig3, use_container_width=True)