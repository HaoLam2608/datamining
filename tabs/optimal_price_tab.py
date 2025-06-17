import streamlit as st
import numpy as np
import pandas as pd
from utils.modeling import train_polynomial_model, predict_revenue, train_random_forest_model,prepare_features
from utils.visualization import create_price_revenue_chart
from sklearn.metrics import r2_score
import altair as alt

def render_optimal_price_tab(df_prod, combo_label):
    st.header("📈 Tìm Giá Bán Tối Ưu")

    # Kiểm tra dữ liệu đầu vào
    if df_prod.empty or not all(col in df_prod.columns for col in ['PRICE', 'QUANTITY', 'CALENDAR_DATE']):
        st.error("❗ Dữ liệu không hợp lệ. Vui lòng kiểm tra lại.")
        return

    # Ép kiểu ngày nếu chưa đúng
    try:
        df_prod['CALENDAR_DATE'] = pd.to_datetime(df_prod['CALENDAR_DATE'], errors='coerce')
    except Exception as e:
        st.error(f"Lỗi khi ép kiểu ngày: {e}")
        return

    # Bỏ các dòng thiếu dữ liệu
    df_prod = df_prod.dropna(subset=['PRICE', 'QUANTITY', 'CALENDAR_DATE'])

    # Nếu sau khi làm sạch dữ liệu còn quá ít
    if len(df_prod) < 10:
        st.warning("⚠️ Không đủ dữ liệu để phân tích cho sản phẩm này.")
        return

    # Train Random Forest trên dữ liệu không gộp
    model = train_random_forest_model(df_prod)

    # Tính R²
    X, y = prepare_features(df_prod)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    st.metric("🎯 R² của mô hình", f"{r2:.2f}")

    # Dự đoán doanh thu trên dải giá
    price_range = np.linspace(df_prod['PRICE'].min(), df_prod['PRICE'].max(), 100)
    avg_dayofweek = int(df_prod['CALENDAR_DATE'].dt.dayofweek.mode()[0])  

    revenue_pred = []
    for p in price_range:
        try:
            qty_pred = model.predict([[p, avg_dayofweek]])[0]
        except Exception as e:
            st.error(f"Lỗi khi dự đoán với giá {p:.2f}: {e}")
            qty_pred = 0
        qty_pred = max(0, qty_pred)
        revenue_pred.append(p * qty_pred)

    # Nếu doanh thu toàn 0 thì cảnh báo
    if all(r == 0 for r in revenue_pred):
        st.warning("⚠️ Mô hình dự đoán doanh thu bằng 0 cho tất cả các mức giá. Vui lòng kiểm tra lại dữ liệu.")
        return

    # Tìm giá tối ưu
    opt_idx = np.argmax(revenue_pred)
    opt_price = price_range[opt_idx]
    opt_revenue = revenue_pred[opt_idx]

    # Hiển thị kết quả từ dữ liệu thực tế
    df_prod['Revenue'] = df_prod['PRICE'] * df_prod['QUANTITY']
    max_revenue_row = df_prod.loc[df_prod['Revenue'].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Giá tối ưu từ dữ liệu thực tế", f"{max_revenue_row['PRICE']:.2f}")
        st.metric("💰 Doanh thu max từ dữ liệu", f"{max_revenue_row['Revenue']:,.2f}")
    with col2:
        st.metric("📊 Giá tối ưu từ mô hình", f"{opt_price:.2f}")
        st.metric("📈 Doanh thu dự đoán", f"{opt_revenue:,.2f}")

    # Tạo DataFrame cho biểu đồ
    pred_df = pd.DataFrame({'Giá': price_range, 'Doanh thu dự đoán': revenue_pred})

    # Nếu biểu đồ không hiển thị do lỗi chart
    try:
        chart = create_price_revenue_chart(df_prod, pred_df)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ doanh thu: {e}")
        st.write(pred_df)

    # Thông tin tăng doanh thu
    current_price = df_prod['PRICE'].mean()
    current_qty = df_prod['QUANTITY'].mean()
    current_revenue = current_price * current_qty
    revenue_increase = (opt_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
    st.info(f"Nếu thay đổi giá từ {current_price:.2f} thành {opt_price:.2f}, doanh thu dự kiến sẽ tăng {revenue_increase:.2f}%")

    # Biểu đồ so sánh
    df_compare = pd.DataFrame({
        'Loại': ['Hiện tại', 'Tối ưu (dự đoán)'],
        'Doanh thu': [current_revenue, opt_revenue]
    })
    chart_compare = alt.Chart(df_compare).mark_bar().encode(
        x='Loại',
        y='Doanh thu',
        color='Loại'
    ).properties(title="So sánh doanh thu hiện tại vs tối ưu")
    st.altair_chart(chart_compare, use_container_width=True)

    # Tải dữ liệu dự đoán
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Tải về dữ liệu dự đoán", data=csv, file_name="du_doan_doanh_thu.csv", mime='text/csv')

