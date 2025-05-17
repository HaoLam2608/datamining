import streamlit as st
import pandas as pd
import altair as alt
from utils.modeling import train_polynomial_model, predict_revenue
from utils.data_processing import clean_data

def render_price_change_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Thay đổi giá"""
    st.header("📊 Tác Động Thay Đổi Giá → Doanh Thu")
    
    df_clean = clean_data(df_prod)
    model, poly_features, _ = train_polynomial_model(df_clean)
    base_price = df_clean['PRICE'].mean()
    current_qty = df_clean['QUANTITY'].mean()
    
    # Thanh trượt thay đổi giá
    price_change = st.slider(
        "Thay đổi giá (%)", 
        min_value=-30, 
        max_value=30, 
        value=0, 
        step=5,
        help="Kéo thanh trượt để xem tác động của việc thay đổi giá"
    )
    
    # Tính toán giá mới
    new_price = base_price * (1 + price_change/100)
    new_revenue = predict_revenue(model, poly_features, new_price)
    new_qty = new_revenue / new_price if new_price > 0 else 0
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá mới", f"{new_price:.2f}", f"{price_change}%")
    with col2:
        qty_change = ((new_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        st.metric("Số lượng dự đoán", f"{new_qty:.2f}", f"{qty_change:.2f}%")
    with col3:
        current_revenue = base_price * current_qty
        rev_change = ((new_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        st.metric("Doanh thu dự đoán", f"{new_revenue:.2f}", f"{rev_change:.2f}%")
    
    # Tạo bảng các mức thay đổi giá
    pct = [-15, -10, -5, 0, 5, 10, 15]
    results = []
    for p in pct:
        adj_price = base_price * (1 + p/100)
        adj_revenue = predict_revenue(model, poly_features, adj_price)
        adj_qty = adj_revenue / adj_price if adj_price > 0 else 0
        qty_pct_change = ((adj_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        rev_pct_change = ((adj_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        results.append({
            'Thay đổi giá (%)': p,
            'Giá mới': round(adj_price, 2),
            'Số lượng dự đoán': round(adj_qty, 2),
            'Thay đổi SL (%)': round(qty_pct_change, 2),
            'Doanh thu dự đoán': round(adj_revenue, 2),
            'Thay đổi doanh thu (%)': round(rev_pct_change, 2)
        })
    
    result_df = pd.DataFrame(results)
    st.subheader("Bảng tác động thay đổi giá")
    st.dataframe(result_df)
    
    # Vẽ biểu đồ
    melted_df = result_df.melt(
        id_vars=['Thay đổi giá (%)'],
        value_vars=['Doanh thu dự đoán', 'Số lượng dự đoán'],
        var_name='Chỉ số',
        value_name='Giá trị'
    )
    chart = alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X('Thay đổi giá (%):Q', title='Thay đổi giá (%)'),
        y=alt.Y('Giá trị:Q', title='Giá trị'),
        color=alt.Color('Chỉ số:N', title='Chỉ số'),
        tooltip=['Thay đổi giá (%)', 'Chỉ số', 'Giá trị']
    ).properties(
        title='Tác động của thay đổi giá đến số lượng và doanh thu'
    )
    st.altair_chart(chart, use_container_width=True)