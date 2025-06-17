import streamlit as st
import altair as alt
import pandas as pd
from utils.modeling import train_polynomial_model
from utils.data_processing import clean_data

def render_price_analysis_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Phân tích giá"""
    st.header("🔎 Phân Tích Giá ↔ Nhu Cầu")
    
    df_clean = clean_data(df_prod)

    # Xử lý outlier theo phân vị 1% và 99%
    initial_count = len(df_clean)
    lower_q = df_clean['QUANTITY'].quantile(0.01)
    upper_q = df_clean['QUANTITY'].quantile(0.99)
    df_clean = df_clean[(df_clean['QUANTITY'] >= lower_q) & (df_clean['QUANTITY'] <= upper_q)]
    filtered_count = len(df_clean)


    grp = df_clean.groupby('PRICE')['QUANTITY'].sum().reset_index()
    grp['Revenue'] = grp['PRICE'] * grp['QUANTITY']
    
    # Tính hệ số tương quan
    corr = grp['PRICE'].corr(grp['QUANTITY'])
    st.write(f"**Hệ số tương quan giữa giá và số lượng:** {corr:.2f}")
    
    if corr < -0.5:
        st.success("👍 Có mối tương quan âm mạnh giữa giá và số lượng bán. Khi giá giảm, số lượng bán tăng rõ rệt.")
    elif corr < 0:
        st.info("ℹ️ Có mối tương quan âm yếu giữa giá và số lượng bán. Giá có ảnh hưởng nhưng không nhiều.")
    elif corr == 0:
        st.warning("⚠️ Không có mối tương quan giữa giá và số lượng bán. Có thể là sản phẩm không nhạy cảm với giá.")
    else:
        st.error("❗ Có mối tương quan dương giữa giá và số lượng bán. Đây là trường hợp đặc biệt (hàng xa xỉ).")
    
    # Vẽ biểu đồ phân tán
    scatter = alt.Chart(grp).mark_circle(size=60).encode(
        x=alt.X('PRICE', title='Giá'),
        y=alt.Y('QUANTITY', title='Số lượng'),
        size='Revenue',
        color=alt.Color('Revenue', scale=alt.Scale(scheme='viridis')),
        tooltip=['PRICE', 'QUANTITY', 'Revenue']
    ).properties(
        title='Mối quan hệ giữa Giá và Số lượng bán'
    ).interactive()
    
   # Thêm đường hồi quy bậc hai (Polynomial)
    regression_quad = alt.Chart(grp).transform_regression(
        'PRICE', 'QUANTITY', method='poly', order=2
    ).mark_line(color='red').encode(
        x='PRICE',
        y='QUANTITY'
    )

    
    st.altair_chart(scatter + regression_quad, use_container_width=True)


    # Hiển thị phân phối số lượng
    st.subheader("📊 Phân phối số lượng bán (QUANTITY)")
    hist = alt.Chart(df_clean).mark_bar().encode(
        alt.X("QUANTITY", bin=alt.Bin(maxbins=50), title="Số lượng"),
        y='count()',
        tooltip=['count()']
    ).properties(title="Histogram số lượng bán")
    st.altair_chart(hist, use_container_width=True)
    
    # Tính độ co giãn của cầu
    model, _, _ = train_polynomial_model(df_clean)
    avg_price = df_clean['PRICE'].mean()
    avg_qty = df_clean['QUANTITY'].mean()
    price_elasticity = model.coef_[0] * (avg_price / avg_qty) if avg_qty > 0 else 0
    
    st.subheader("Phân tích độ co giãn của cầu (Price Elasticity)")
    st.metric("Độ co giãn của cầu", f"{abs(price_elasticity):.2f}")
    
    if abs(price_elasticity) > 1:
        st.success("📈 Cầu có tính co giãn cao (elastic): Thay đổi giá sẽ tạo ra sự thay đổi lớn về số lượng bán.")
    elif abs(price_elasticity) < 1:
        st.info("📉 Cầu kém co giãn (inelastic): Thay đổi giá sẽ không ảnh hưởng nhiều đến số lượng bán.")
    else:
        st.warning("⚖️ Cầu co giãn đơn vị (unit elastic): Thay đổi giá và số lượng bán tỷ lệ thuận với nhau.")
