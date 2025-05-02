import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

def render_adjust_product_tab(merged):
    """Hiển thị nội dung tab Sản phẩm cần điều chỉnh"""
    st.header("📦 Đề xuất sản phẩm cần điều chỉnh giá")
    
    # Phân tích tất cả sản phẩm
    product_analysis = merged.groupby('ITEM_NAME').agg({
        'PRICE': 'mean',
        'QUANTITY': 'mean',
        'CALENDAR_DATE': 'count'
    }).reset_index()
    product_analysis['Revenue'] = product_analysis['PRICE'] * product_analysis['QUANTITY']
    product_analysis = product_analysis.rename(columns={'CALENDAR_DATE': 'Số giao dịch'})
    
    # Tính độ co giãn của cầu
    elasticity_dict = {}
    for item in product_analysis['ITEM_NAME']:
        df_item = merged[merged['ITEM_NAME'] == item].groupby('PRICE')['QUANTITY'].sum().reset_index()
        if len(df_item) > 1:
            X_item = df_item[['PRICE']].values
            y_item = df_item['QUANTITY'].values
            model_item = LinearRegression().fit(X_item, y_item)
            avg_price_item = df_item['PRICE'].mean()
            avg_qty_item = df_item['QUANTITY'].mean()
            elasticity_dict[item] = abs(model_item.coef_[0] * (avg_price_item / avg_qty_item)) if avg_qty_item > 0 else 0
        else:
            elasticity_dict[item] = None
    
    product_analysis['Elasticity'] = product_analysis['ITEM_NAME'].map(elasticity_dict)
    
    # Xác định sản phẩm cần điều chỉnh
    low_revenue_threshold = product_analysis['Revenue'].quantile(0.25)
    high_elasticity_threshold = product_analysis['Elasticity'].quantile(0.75, interpolation='nearest') if product_analysis['Elasticity'].notna().sum() > 0 else 1
    
    product_analysis['Đề xuất'] = product_analysis.apply(
        lambda row: 'Giảm giá' if (row['Revenue'] < low_revenue_threshold and row['Elasticity'] is not None and row['Elasticity'] > high_elasticity_threshold)
        else 'Tăng giá' if (row['Revenue'] < low_revenue_threshold and row['Elasticity'] is not None and row['Elasticity'] < 1)
        else 'Giữ nguyên', axis=1
    )
    
    st.subheader("Phân tích và đề xuất điều chỉnh giá sản phẩm")
    st.dataframe(product_analysis)
    
    adjustment_needed = product_analysis[product_analysis['Đề xuất'] != 'Giữ nguyên']
    if not adjustment_needed.empty:
        st.subheader("Sản phẩm cần điều chỉnh giá")
        st.dataframe(adjustment_needed)
    else:
        st.info("ℹ️ Không có sản phẩm nào cần điều chỉnh giá dựa trên dữ liệu hiện tại.")