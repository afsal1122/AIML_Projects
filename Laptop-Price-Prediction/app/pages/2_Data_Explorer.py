import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from app.app_utils import load_artifacts

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
df, _, _, _, _ = load_artifacts()

st.title("📊 Market Insights")
st.info(f"Analyzing {len(df)} laptops.")

tab1, tab2 = st.tabs(["📈 Price Trends", "💻 Tech Specs"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Price by Brand")
        fig, ax = plt.subplots()
        df.groupby('brand')['price'].mean().sort_values().plot(kind='barh', ax=ax, color='#4F8BF9')
        st.pyplot(fig)
    with col2:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax, color='green')
        st.pyplot(fig)

with tab2:
    st.subheader("RAM vs Price Correlation")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='ram_gb', y='price', hue='brand', alpha=0.6, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Raw Data")
    st.dataframe(df)