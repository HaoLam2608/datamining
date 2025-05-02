import streamlit as st
import pandas as pd
import altair as alt
from utils.modeling import train_polynomial_model, predict_revenue
from utils.data_processing import clean_data

def render_competitor_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Đối thủ"""
    st.header("🤝 Phân tích cạnh tranh")
    st.write(f"Sản phẩm/Combo: **{combo_label}**")
    
    df_clean = clean_data(df_prod)
    model, poly_features, _ = train_polynomial_model(df_clean)
    base_price = df_clean['PRICE'].mean()
    
    # Nhập thông tin đối thủ
    st.subheader("Thông tin đối thủ")
    col1, col2 = st.columns(2)
    with col1:
        competitor_names = st.text_area(
            "Tên đối thủ (mỗi dòng một tên)",
            "Đối thủ A\nĐối thủ B\nĐối thủ C",
            help="Nhập tên các đối thủ, mỗi đối thủ một dòng"
        ).strip().split("\n")
    
    with col2:
        competitor_prices = st.text_area(
            "Giá của đối thủ (mỗi dòng một giá)",
            f"{base_price*0.95:.2f}\n{base_price*1.05:.2f}\n{base_price*1.15:.2f}",
            help="Nhập giá của từng đối thủ, mỗi giá một dòng"
        ).strip().split("\n")
    
    # Kiểm tra đầu vào
    if len(competitor_names) != len(competitor_prices):
        st.error("Số lượng tên đối thủ và giá không khớp nhau!")
        return
    
    try:
        competitor_prices = [float(price) for price in competitor_prices]
        competitors = dict(zip(competitor_names, competitor_prices))
        
        # Phân tích thị phần
        data = []
        total_q = 0
        for name, price in competitors.items():
            revenue = predict_revenue(model, poly_features, price)
            q = revenue / price if price > 0 else 0
            total_q += q
            data.append((name, price, q, revenue))
        
        # Thêm dữ liệu shop của mình
        own_shop_name = "Shop của bạn"
        own_q = df_clean['QUANTITY'].mean()
        own_r = base_price * own_q
        total_q += own_q
        data.append((own_shop_name, base_price, own_q, own_r))
        
        df_comp = pd.DataFrame(data, columns=['Đối thủ', 'Giá', 'SL dự đoán', 'Doanh thu'])
        df_comp['Thị phần (%)'] = (df_comp['SL dự đoán'] / total_q * 100).round(2)
        
        st.subheader("Phân tích cạnh tranh dựa trên giá")
        st.dataframe(df_comp)
        
        # Vẽ biểu đồ
        chart1 = alt.Chart(df_comp).mark_arc().encode(
            theta=alt.Theta(field="Thị phần (%)", type="quantitative"),
            color=alt.Color(field="Đối thủ", type="nominal", legend=alt.Legend(title="Đối thủ")),
            tooltip=['Đối thủ', 'Giá', 'SL dự đoán', 'Thị phần (%)']
        ).properties(title="Thị phần dự đoán", width=300, height=300)
        
        chart2 = alt.Chart(df_comp).mark_bar().encode(
            x=alt.X('Đối thủ:N', title='Đối thủ'),
            y=alt.Y('Giá:Q', title='Giá'),
            color=alt.Color('Đối thủ:N', legend=None),
            tooltip=['Đối thủ', 'Giá']
        ).properties(title="So sánh giá giữa các đối thủ", width=300, height=300)
        
        st.altair_chart(alt.hconcat(chart1, chart2), use_container_width=True)
        
        # Phân tích vị thế cạnh tranh
        own_position = df_comp[df_comp['Đối thủ'] == own_shop_name].iloc[0]
        competitors_df = df_comp[df_comp['Đối thủ'] != own_shop_name]
        avg_competitor_price = competitors_df['Giá'].mean()
        price_difference = ((base_price - avg_competitor_price) / avg_competitor_price) * 100
        
        st.subheader("Phân tích vị thế cạnh tranh")
        if price_difference > 5:
            st.warning(f"⚠️ Giá của bạn cao hơn {abs(price_difference):.2f}% so với giá trung bình của đối thủ.")
        elif price_difference < -5:
            st.success(f"✅ Giá của bạn thấp hơn {abs(price_difference):.2f}% so với giá trung bình của đối thủ.")
        else:
            st.info(f"ℹ️ Giá của bạn tương đương với giá trung bình của đối thủ (chênh lệch {price_difference:.2f}%).")
        
        # Đề xuất chiến lược
        st.subheader("Đề xuất chiến lược cạnh tranh")
        if own_position['Thị phần (%)'] < 20:
            if price_difference > 5:
                st.error("❗ Thị phần thấp và giá cao hơn đối thủ. Nên xem xét giảm giá để tăng khả năng cạnh tranh.")
            else:
                st.warning("⚠️ Thị phần thấp. Nên xem xét các yếu tố khác ngoài giá như chất lượng, dịch vụ.")
        else:
            if price_difference < -5:
                st.success("✅ Thị phần tốt và giá thấp hơn đối thủ. Có thể xem xét tăng giá để tối ưu doanh thu.")
            else:
                st.success("👍 Vị thế cạnh tranh tốt. Nên duy trì chiến lược hiện tại và theo dõi đối thủ.")
                
    except ValueError as e:
        st.error(f"Lỗi khi xử lý dữ liệu đối thủ: {e}")