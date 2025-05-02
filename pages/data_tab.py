import streamlit as st
from utils.data_processing import clean_data

def render_data_tab(df_prod):
    """Hiển thị nội dung tab Dữ liệu"""
    st.header("📋 Dữ liệu sau khi chọn")
    
    df_clean = clean_data(df_prod)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Số lượng dữ liệu", f"{len(df_prod):,}")
        st.metric("Giá trung bình", f"{df_prod['PRICE'].mean():.2f}")
    with col2:
        st.metric("Số lượng bán trung bình", f"{df_prod['QUANTITY'].mean():.2f}")
        st.metric("Doanh thu trung bình", f"{(df_prod['PRICE'] * df_prod['QUANTITY']).mean():.2f}")
    
    st.subheader("Dữ liệu gốc")
    st.dataframe(df_prod.head(10))
    
    st.subheader("Dữ liệu sau khi loại bỏ outliers")
    st.dataframe(df_clean.head(10))
    
    st.subheader("Thống kê mô tả")
    st.dataframe(df_clean.describe())