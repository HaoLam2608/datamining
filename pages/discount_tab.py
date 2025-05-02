import streamlit as st
import pandas as pd
import altair as alt
from utils.modeling import train_polynomial_model, predict_revenue
from utils.data_processing import clean_data

def render_discount_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Giảm giá"""
    st.header("📉 Tác Động Của Giảm Giá Đến Lượng Hàng Bán Ra")
    
    df_clean = clean_data(df_prod)
    model, poly_features, _ = train_polynomial_model(df_clean)
    base_price = df_clean['PRICE'].mean()
    current_qty = df_clean['QUANTITY'].mean()
    
    # Thanh trượt mức giảm giá
    discount_pct = st.slider(
        "Mức giảm giá (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Kéo thanh trượt để điều chỉnh mức giảm giá"
    )
    
    # Tính giá sau giảm
    discounted_price = base_price * (1 - discount_pct/100)
    discounted_revenue = predict_revenue(model, poly_features, discounted_price)
    discounted_qty = discounted_revenue / discounted_price if discounted_price > 0 else 0
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá sau giảm", f"{discounted_price:.2f}", f"-{discount_pct}%")
    with col2:
        qty_change = ((discounted_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        st.metric("Số lượng dự đoán", f"{discounted_qty:.2f}", f"{qty_change:.2f}%")
    with col3:
        current_revenue = base_price * current_qty
        rev_change = ((discounted_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        st.metric("Doanh thu dự đoán", f"{discounted_revenue:.2f}", f"{rev_change:.2f}%")
    
    # Tạo bảng các mức giảm giá
    discount_range = range(0, 55, 5)
    results = []
    for d in discount_range:
        adj_price = base_price * (1 - d/100)
        adj_revenue = predict_revenue(model, poly_features, adj_price)
        adj_qty = adj_revenue / adj_price if adj_price > 0 else 0
        qty_pct_change = ((adj_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        rev_pct_change = ((adj_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        results.append({
            'Giảm giá (%)': d,
            'Giá sau giảm': round(adj_price, 2),
            'Số lượng dự đoán': round(adj_qty, 2),
            'Thay đổi SL (%)': round(qty_pct_change, 2),
            'Doanh thu dự đoán': round(adj_revenue, 2),
            'Thay đổi doanh thu (%)': round(rev_pct_change, 2)
        })
    
    result_df = pd.DataFrame(results)
    opt_discount = result_df.loc[result_df['Doanh thu dự đoán'].idxmax()]
    
    st.subheader("Phân tích các mức giảm giá khác nhau")
    st.dataframe(result_df)
    
    st.success(f"✅ Mức giảm giá tối ưu: **{opt_discount['Giảm giá (%)']}%** - Doanh thu dự đoán: **{opt_discount['Doanh thu dự đoán']:.2f}** (+{opt_discount['Thay đổi doanh thu (%)']:.2f}%)")
    
    # Vẽ biểu đồ
    chart = alt.Chart(result_df).mark_line(point=True).encode(
        x=alt.X('Giảm giá (%):Q', title='Giảm giá (%)'),
        y=alt.Y('Doanh thu dự đoán:Q', title='Doanh thu dự đoán'),
        tooltip=['Giảm giá (%)', 'Giá sau giảm', 'Số lượng dự đoán', 'Doanh thu dự đoán']
    ).properties(
        title='Tác động của giảm giá đến doanh thu'
    )
    
    highlight = alt.Chart(pd.DataFrame([opt_discount])).mark_circle(size=100, color='red').encode(
        x='Giảm giá (%):Q', 
        y='Doanh thu dự đoán:Q'
    )
    
    st.altair_chart(chart + highlight, use_container_width=True)
    
    # Phân tích chi tiết
    st.subheader("Phân tích chi tiết")
    if opt_discount['Giảm giá (%)'] == 0:
        st.info("ℹ️ Không cần giảm giá - Giá hiện tại đã tối ưu.")
    elif opt_discount['Giảm giá (%)'] <= 15:
        st.info(f"ℹ️ Mức giảm giá nhẹ ({opt_discount['Giảm giá (%)']}%) phù hợp cho khuyến mãi ngắn hạn.")
    elif opt_discount['Giảm giá (%)'] <= 30:
        st.warning(f"⚠️ Mức giảm giá trung bình ({opt_discount['Giận giá (%)']}%) cần cân nhắc lợi nhuận.")
    else:
        st.error(f"❗ Mức giảm giá cao ({opt_discount['Giảm giá (%)']}%) cho thấy giá hiện tại có thể quá cao.")