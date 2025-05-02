import streamlit as st
import pandas as pd

def render_personalized_pricing_tab(merged):
    """Hiển thị nội dung tab Định giá cá nhân hóa"""
    st.header("👤 Định giá cá nhân hóa (Đang phát triển)")
    
    st.info("Tính năng này đang trong quá trình phát triển. Ý tưởng bao gồm:")
    st.markdown("""
    - Sử dụng dữ liệu khách hàng để phân khúc theo hành vi mua sắm.
    - Đề xuất giá khác nhau cho từng nhóm khách hàng dựa trên độ nhạy cảm giá.
    - Tích hợp mô hình học máy để dự đoán mức giá tối ưu cho từng khách hàng.
    """)
    
    possible_customer_cols = ['CUSTOMER_ID', 'CustomerID', 'customer_id', 'UserID', 'ClientID', 'user_id']
    customer_col = None
    for col in possible_customer_cols:
        if col in merged.columns:
            customer_col = col
            break
    
    if customer_col:
        customer_analysis = merged.groupby(customer_col).agg({
            'PRICE': 'mean',
            'QUANTITY': 'sum',
            'CALENDAR_DATE': 'count'
        }).reset_index()
        customer_analysis['Revenue'] = customer_analysis['PRICE'] * customer_analysis['QUANTITY']
        st.subheader(f"Phân tích sơ bộ theo khách hàng (sử dụng cột: {customer_col})")
        st.dataframe(customer_analysis.head())
    else:
        st.warning("Không tìm thấy cột dữ liệu khách hàng (kiểm tra các cột: {}).".format(', '.join(possible_customer_cols)))
        st.info("Thêm một cột như 'CUSTOMER_ID' vào file transaction.csv để kích hoạt tính năng này.")