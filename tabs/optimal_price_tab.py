import streamlit as st
import numpy as np
import pandas as pd
from utils.modeling import train_polynomial_model, predict_revenue
from utils.visualization import create_price_revenue_chart

def render_optimal_price_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Giá tối ưu"""
    st.header("📈 Tìm Giá Bán Tối Ưu")
    
    # Huấn luyện mô hình
    model, poly_features, grp = train_polynomial_model(df_prod)
    grp['Revenue'] = grp['PRICE'] * grp['QUANTITY']
    
    # Tạo giá trị giá để dự đoán
    price_range = np.linspace(grp['PRICE'].min(), grp['PRICE'].max(), 100)
    revenue_pred = [predict_revenue(model, poly_features, p) for p in price_range]
    
    # Tìm giá tối ưu
    opt_idx = np.argmax(revenue_pred)
    opt_price = price_range[opt_idx]
    opt_revenue = revenue_pred[opt_idx]
    
    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Giá tối ưu từ dữ liệu thực tế", f"{grp.loc[grp['Revenue'].idxmax()]['PRICE']:.2f}")
        st.metric("💰 Doanh thu max từ dữ liệu", f"{grp['Revenue'].max():,.2f}")
    with col2:
        st.metric("📊 Giá tối ưu từ mô hình", f"{opt_price:.2f}")
        st.metric("📈 Doanh thu dự đoán", f"{opt_revenue:,.2f}")
    
    # Tạo biểu đồ
    pred_df = pd.DataFrame({'Giá': price_range, 'Doanh thu dự đoán': revenue_pred})
    chart = create_price_revenue_chart(grp, pred_df)
    st.altair_chart(chart, use_container_width=True)
    
    # Hiển thị điểm tối ưu
    st.markdown(f"**Điểm giá tối ưu:** {opt_price:.2f} (Doanh thu dự đoán: {opt_revenue:,.2f})")
    
    # Tính phần trăm tăng doanh thu
    current_price = df_prod['PRICE'].mean()
    current_qty = df_prod['QUANTITY'].mean()
    current_revenue = current_price * current_qty
    revenue_increase = (opt_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
    st.info(f"Nếu thay đổi giá từ {current_price:.2f} thành {opt_price:.2f}, doanh thu dự kiến sẽ tăng {revenue_increase:.2f}%")