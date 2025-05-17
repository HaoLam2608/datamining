import streamlit as st
import pandas as pd
import altair as alt
import chardet
from io import StringIO
from utils.modeling import train_polynomial_model
from utils.data_processing import clean_data

# Hàm đọc file CSV tự động nhận diện encoding
def read_uploaded_csv(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    
    try:
        decoded_data = raw_data.decode(encoding)
        return pd.read_csv(StringIO(decoded_data))
    except Exception as e:
        raise ValueError(f"Lỗi khi giải mã file CSV bằng encoding '{encoding}': {e}")

def render_competitor_tab(df_prod, combo_label):
    st.header("🤝 Phân tích cạnh tranh")
    st.write(f"Sản phẩm/Combo: **{combo_label}**")
    
    df_clean = clean_data(df_prod)
    model, poly_features, _ = train_polynomial_model(df_clean)
    base_price = df_clean['PRICE'].mean()
    
    # Tải file đối thủ
    st.subheader("Tải lên file CSV chứa thông tin đối thủ")
    uploaded_file = st.file_uploader("📄 Chọn file CSV có cột NAME, PRICE, QUANTITY, REVENUE", type=["csv"])
    
    if uploaded_file is None:
        st.info("📥 Vui lòng tải lên file CSV có 4 cột: NAME, PRICE, QUANTITY, REVENUE.")
        return
    
    try:
        df_competitor = read_uploaded_csv(uploaded_file)
        required_cols = {'NAME', 'PRICE', 'QUANTITY', 'REVENUE'}
        if not required_cols.issubset(df_competitor.columns):
            st.error(f"❌ File CSV phải có bốn cột: {', '.join(required_cols)}.")
            return
        
        # Lấy dữ liệu đối thủ
        competitor_names = df_competitor['NAME'].astype(str).tolist()
        competitor_prices = df_competitor['PRICE'].astype(float).tolist()
        competitor_quantities = df_competitor['QUANTITY'].astype(float).tolist()
        competitor_revenues = df_competitor['REVENUE'].astype(float).tolist()
        
        data = []
        total_q = 0
        
        # Dùng dữ liệu đối thủ có sẵn, không dự đoán nữa
        for name, price, q, r in zip(competitor_names, competitor_prices, competitor_quantities, competitor_revenues):
            total_q += q
            data.append((name, price, q, r))
        
        # Thêm dữ liệu shop của bạn (dự đoán dựa trên model)
        own_shop_name = "Your Shop"
        own_q = df_clean['QUANTITY'].mean()
        own_r = base_price * own_q
        total_q += own_q
        data.append((own_shop_name, base_price, own_q, own_r))
        
        df_comp = pd.DataFrame(data, columns=['Competitor', 'Price', 'Quantity', 'Revenue'])
        df_comp['Market Share (%)'] = (df_comp['Quantity'] / total_q * 100).round(2)
        
        st.subheader("Phân tích cạnh tranh dựa trên dữ liệu thực tế")
        st.dataframe(df_comp)
        
        chart1 = alt.Chart(df_comp).mark_arc().encode(
            theta=alt.Theta(field="Market Share (%)", type="quantitative"),
            color=alt.Color(field="Competitor", type="nominal"),
            tooltip=['Competitor', 'Price', 'Quantity', 'Market Share (%)']
        ).properties(title="Thị phần dựa trên số lượng", width=300, height=300)

        chart2 = alt.Chart(df_comp).mark_bar().encode(
            x=alt.X('Competitor:N'),
            y=alt.Y('Price:Q'),
            color=alt.Color('Competitor:N', legend=None),
            tooltip=['Competitor', 'Price']
        ).properties(title="So sánh giá", width=300, height=300)

        st.altair_chart(alt.hconcat(chart1, chart2), use_container_width=True)

        # Phân tích vị thế
        own_position = df_comp[df_comp['Competitor'] == own_shop_name].iloc[0]
        competitors_df = df_comp[df_comp['Competitor'] != own_shop_name]
        avg_competitor_price = competitors_df['Price'].mean()
        price_difference = ((base_price - avg_competitor_price) / avg_competitor_price) * 100
        
        st.subheader("Phân tích vị thế cạnh tranh")
        if price_difference > 5:
            st.warning(f"⚠️ Giá của bạn cao hơn {abs(price_difference):.2f}% so với giá trung bình của đối thủ.")
        elif price_difference < -5:
            st.success(f"✅ Giá của bạn thấp hơn {abs(price_difference):.2f}% so với giá trung bình của đối thủ.")
        else:
            st.info(f"ℹ️ Giá của bạn tương đương với giá trung bình của đối thủ (chênh lệch {price_difference:.2f}%).")

        st.subheader("Đề xuất chiến lược cạnh tranh")
        if own_position['Market Share (%)'] < 20:
            if price_difference > 5:
                st.error("❗ Thị phần thấp và giá cao hơn đối thủ. Nên xem xét giảm giá để tăng cạnh tranh.")
            else:
                st.warning("⚠️ Thị phần thấp. Nên xem xét các yếu tố khác ngoài giá như chất lượng, dịch vụ.")
        else:
            if price_difference < -5:
                st.success("✅ Thị phần tốt và giá thấp hơn đối thủ. Có thể xem xét tăng giá để tối ưu doanh thu.")
            else:
                st.success("👍 Vị thế cạnh tranh tốt. Nên duy trì chiến lược hiện tại.")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file hoặc xử lý dữ liệu: {e}")
