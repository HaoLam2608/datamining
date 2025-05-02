import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tối Ưu Giá Bán Cafe & Phân Tích Đối Thủ", layout="wide")
st.title("☕ Ứng dụng Tối Ưu & Phân Tích Giá Bán Cafe Shop (Cạnh Tranh)")

# Sidebar: upload dữ liệu
st.sidebar.header("🚀 Upload dữ liệu")
u_meta  = st.sidebar.file_uploader("Sell Meta Data (CSV)", type="csv")
u_trans = st.sidebar.file_uploader("Transaction Store (CSV)", type="csv")
u_date  = st.sidebar.file_uploader("Date Info (CSV)", type="csv")

if not (u_meta and u_trans and u_date):
    st.sidebar.info("Vui lòng upload cả 3 file để bắt đầu!")
    st.info("Chào mừng bạn đến với ứng dụng phân tích giá bán cà phê! Vui lòng tải lên 3 tệp CSV để bắt đầu phân tích:")
    st.markdown("""
    - **Sell Meta Data**: Chứa thông tin về sản phẩm và danh mục
    - **Transaction Store**: Chứa dữ liệu giao dịch bán hàng
    - **Date Info**: Chứa thông tin về ngày (lễ, cuối tuần, mùa vụ)
    """)
    st.stop()

try:
    # Giả sử bạn đã tải các file từ sidebar
    sell_meta = pd.read_csv(u_meta)
    transaction = pd.read_csv(u_trans)
    date_info = pd.read_csv(u_date)

    # Chuẩn hóa tên cột
    sell_meta.columns = sell_meta.columns.str.strip()
    transaction.columns = transaction.columns.str.strip()
    date_info.columns = date_info.columns.str.strip()

    # Convert CALENDAR_DATE sang định dạng datetime (nếu chưa)
    transaction['CALENDAR_DATE'] = pd.to_datetime(transaction['CALENDAR_DATE'], errors='coerce')
    date_info['CALENDAR_DATE'] = pd.to_datetime(date_info['CALENDAR_DATE'], errors='coerce')

    # Merge dữ liệu transaction và sell_meta
    merged = pd.merge(transaction, sell_meta, on=["SELL_ID", "SELL_CATEGORY"], how="left")

    # Merge dữ liệu merged với date_info dựa trên CALENDAR_DATE
    merged = pd.merge(merged, date_info, on="CALENDAR_DATE", how="left")

    # Kiểm tra kết quả merge
    st.write("Kết quả sau khi merge:")
    st.dataframe(merged.head())
    
    # Hiển thị thống kê cơ bản
    st.write("Thống kê cơ bản của dữ liệu:")
    st.write(f"Số lượng giao dịch: {len(merged):,}")
    st.write(f"Số lượng sản phẩm: {merged['ITEM_NAME'].nunique():,}")
    st.write(f"Khoảng thời gian: {merged['CALENDAR_DATE'].min().date()} đến {merged['CALENDAR_DATE'].max().date()}")

except Exception as e:
    st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    st.stop()

# Tiếp tục phần chọn sản phẩm...
items = merged['ITEM_NAME'].dropna().unique().tolist()

# Sidebar: chọn 1 hoặc 2 sản phẩm
selected_items = st.sidebar.multiselect("🛒 Chọn 1 hoặc 2 sản phẩm:", items, max_selections=2)
if not selected_items:
    st.sidebar.info("Chọn ít nhất 1 sản phẩm.")
    st.stop()

# Lọc dữ liệu theo lựa chọn
if len(selected_items) == 1:
    df_prod = merged[merged['ITEM_NAME'] == selected_items[0]].copy()
    combo_label = selected_items[0]
else:
    df_prod = merged[merged['ITEM_NAME'].isin(selected_items)].copy()
    combo_label = ' + '.join(selected_items)
    df_prod['ITEM_NAME'] = combo_label

if df_prod.empty:
    st.warning("Không có dữ liệu cho lựa chọn này.")
    st.stop()

# Xử lý dữ liệu chung
df_prod = df_prod.dropna(subset=['PRICE', 'QUANTITY'])
# Loại bỏ outliers bằng IQR
Q1_price = df_prod['PRICE'].quantile(0.25)
Q3_price = df_prod['PRICE'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

Q1_qty = df_prod['QUANTITY'].quantile(0.25)
Q3_qty = df_prod['QUANTITY'].quantile(0.75)
IQR_qty = Q3_qty - Q1_qty
lower_bound_qty = Q1_qty - 1.5 * IQR_qty
upper_bound_qty = Q3_qty + 1.5 * IQR_qty

df_clean = df_prod[
    (df_prod['PRICE'] >= lower_bound_price) & 
    (df_prod['PRICE'] <= upper_bound_price) &
    (df_prod['QUANTITY'] >= lower_bound_qty) & 
    (df_prod['QUANTITY'] <= upper_bound_qty)
].copy()

# Tính toán giá trung bình và số lượng theo từng mức giá
grp = df_clean.groupby('PRICE')['QUANTITY'].sum().reset_index()
grp['Revenue'] = grp['PRICE'] * grp['QUANTITY']

# Tab UI
tabs = st.tabs([
    "📋 Dữ liệu", "📈 Giá tối ưu", "🔍 Phân tích giá", "💰 Thay đổi giá",
    "🏢 Đối thủ", "📊 So sánh giá & SL", "🌸 Xu hướng theo mùa", "📉 Giảm giá",
    "🎯 Tối ưu CTKM", "📦 Sản phẩm cần điều chỉnh", "👤 Định giá cá nhân hóa"
])

# Tab 1: Dữ liệu
with tabs[0]:
    st.header("📋 Dữ liệu sau khi chọn")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Số lượng dữ liệu", f"{len(df_prod):,}")
        st.metric("Giá trung bình", f"{df_prod['PRICE'].mean():.2f}")
    with col2:
        st.metric("Số lượng bán trung bình", f"{df_prod['QUANTITY'].mean():.2f}")
        st.metric("Doanh thu trung bình", f"{(df_prod['PRICE'] * df_prod['QUANTITY']).mean():.2f}")
    
    # Hiển thị trước và sau khi loại bỏ outliers
    st.subheader("Dữ liệu gốc")
    st.dataframe(df_prod.head(10))
    
    st.subheader("Dữ liệu sau khi loại bỏ outliers")
    st.dataframe(df_clean.head(10))
    
    # Thống kê mô tả
    st.subheader("Thống kê mô tả")
    st.dataframe(df_clean.describe())

# Tab 2: Giá tối ưu
with tabs[1]:
    st.header("📈 Tìm Giá Bán Tối Ưu")
    
    # Thử nhiều mô hình và chọn mô hình tốt nhất
    X = grp[['PRICE']].values
    y = grp['QUANTITY'].values
    
    # Thử bậc 2 cho mô hình đa thức
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # Huấn luyện mô hình hồi quy đa thức
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    
    # Tạo giá trị giá để dự đoán
    price_range = np.linspace(grp['PRICE'].min(), grp['PRICE'].max(), 100)
    X_range = np.array(price_range).reshape(-1, 1)
    X_poly_range = poly_features.transform(X_range)
    
    # Dự đoán số lượng
    y_pred = poly_model.predict(X_poly_range)
    
    # Tính doanh thu dự đoán
    revenue_pred = price_range * y_pred
    
    # Tìm giá tối ưu từ mô hình
    opt_idx = np.argmax(revenue_pred)
    opt_price = price_range[opt_idx]
    opt_revenue = revenue_pred[opt_idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Giá tối ưu từ dữ liệu thực tế", f"{grp.loc[grp['Revenue'].idxmax()]['PRICE']:.2f}")
        st.metric("💰 Doanh thu max từ dữ liệu", f"{grp['Revenue'].max():,.2f}")
    with col2:
        st.metric("📊 Giá tối ưu từ mô hình", f"{opt_price:.2f}")
        st.metric("📈 Doanh thu dự đoán", f"{opt_revenue:,.2f}")
    
    # Biểu đồ doanh thu theo giá
    pred_df = pd.DataFrame({
        'Giá': price_range,
        'Doanh thu dự đoán': revenue_pred
    })
    
    chart1 = alt.Chart(grp).mark_circle(size=100).encode(
        x=alt.X('PRICE', title='Giá'),
        y=alt.Y('Revenue', title='Doanh thu thực tế'),
        tooltip=['PRICE', 'QUANTITY', 'Revenue']
    ).properties(title='Doanh thu thực tế theo giá')
    
    chart2 = alt.Chart(pred_df).mark_line(color='red').encode(
        x=alt.X('Giá', title='Giá'),
        y=alt.Y('Doanh thu dự đoán', title='Doanh thu dự đoán'),
        tooltip=['Giá', 'Doanh thu dự đoán']
    )
    
    st.altair_chart(chart1 + chart2, use_container_width=True)
    
    # Hiển thị điểm tối ưu
    st.markdown(f"**Điểm giá tối ưu:** {opt_price:.2f} (Doanh thu dự đoán: {opt_revenue:,.2f})")
    
    # Tính phần trăm tăng doanh thu so với giá hiện tại
    current_price = df_clean['PRICE'].mean()
    current_qty = df_clean['QUANTITY'].mean()
    current_revenue = current_price * current_qty
    
    revenue_increase = (opt_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
    st.info(f"Nếu thay đổi giá từ {current_price:.2f} thành {opt_price:.2f}, doanh thu dự kiến sẽ tăng {revenue_increase:.2f}%")

# Tab 3: Phân tích giá
with tabs[2]:
    st.header("🔎 Phân Tích Giá ↔ Nhu Cầu")
    
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
    
    # Thêm đường hồi quy
    regression_line = scatter.transform_regression(
        'PRICE', 'QUANTITY'
    ).mark_line(color='red').encode(
        x='PRICE',
        y='QUANTITY'
    )
    
    st.altair_chart(scatter + regression_line, use_container_width=True)
    
    # Tính độ co giãn của cầu (Price Elasticity of Demand)
    avg_price = df_clean['PRICE'].mean()
    avg_qty = df_clean['QUANTITY'].mean()
    
    # Sử dụng hệ số hồi quy để tính độ co giãn
    model = LinearRegression().fit(X, y)
    price_elasticity = model.coef_[0] * (avg_price / avg_qty)
    
    st.subheader("Phân tích độ co giãn của cầu (Price Elasticity)")
    st.metric("Độ co giãn của cầu", f"{abs(price_elasticity):.2f}")
    
    if abs(price_elasticity) > 1:
        st.success("📈 Cầu có tính co giãn cao (elastic): Thay đổi giá sẽ tạo ra sự thay đổi lớn về số lượng bán.")
    elif abs(price_elasticity) < 1:
        st.info("📉 Cầu kém co giãn (inelastic): Thay đổi giá sẽ không ảnh hưởng nhiều đến số lượng bán.")
    else:
        st.warning("⚖️ Cầu co giãn đơn vị (unit elastic): Thay đổi giá và số lượng bán tỷ lệ thuận với nhau.")

# Tab 4: Thay đổi giá
with tabs[3]:
    st.header("📊 Tác Động Thay Đổi Giá → Doanh Thu")
    
    # Sử dụng mô hình đa thức bậc 2 để dự đoán tốt hơn
    base_price = df_clean['PRICE'].mean()
    
    # Tạo thanh trượt để điều chỉnh phần trăm thay đổi giá
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
    
    # Dự đoán số lượng với giá mới
    new_price_poly = poly_features.transform(np.array([[new_price]]))
    new_qty = poly_model.predict(new_price_poly)[0]
    new_revenue = new_price * new_qty
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá mới", f"{new_price:.2f}", f"{price_change}%")
    with col2:
        current_qty = df_clean['QUANTITY'].mean()
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
        adj_price_poly = poly_features.transform(np.array([[adj_price]]))
        adj_qty = max(0, poly_model.predict(adj_price_poly)[0])  # Đảm bảo số lượng không âm
        adj_revenue = adj_price * adj_qty
        
        # Tính phần trăm thay đổi so với hiện tại
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

    # Hiển thị bảng kết quả
    result_df = pd.DataFrame(results)
    st.subheader("Bảng tác động thay đổi giá")
    st.dataframe(result_df)

    # Debugging: Check columns and data
    st.write("Columns in result_df:", result_df.columns.tolist())
    if result_df[['Doanh thu dự đoán', 'Số lượng dự đoán']].isna().any().any():
        st.warning("Warning: NaN values detected in 'Doanh thu dự đoán' or 'Số lượng dự đoán'.")

    # Melt the DataFrame to long format
    melted_df = result_df.melt(
        id_vars=['Thay đổi giá (%)'],
        value_vars=['Doanh thu dự đoán', 'Số lượng dự đoán'],
        var_name='Chỉ số',
        value_name='Giá trị'
    )

    # Debugging: Inspect melted DataFrame
    st.write("Melted DataFrame:", melted_df.head())

    # Create the Altair chart using the melted DataFrame
    chart = alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X('Thay đổi giá (%):Q', title='Thay đổi giá (%)'),
        y=alt.Y('Giá trị:Q', title='Giá trị'),
        color=alt.Color('Chỉ số:N', title='Chỉ số'),
        tooltip=['Thay đổi giá (%)', 'Chỉ số', 'Giá trị']
    ).properties(
        title='Tác động của thay đổi giá đến số lượng và doanh thu'
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

# Tab 5: Đối thủ
with tabs[4]:
    st.header("🤝 Phân tích cạnh tranh")
    st.write(f"Sản phẩm/Combo: **{combo_label}**")
    
    # Cho phép người dùng nhập thông tin đối thủ
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
            help="Nhập giá của từng đối thủ, mỗi giá một dòng, theo thứ tự tương ứng với tên đối thủ"
        ).strip().split("\n")
    
    # Kiểm tra đầu vào
    if len(competitor_names) != len(competitor_prices):
        st.error("Số lượng tên đối thủ và giá không khớp nhau!")
    else:
        # Chuyển đổi giá sang số
        try:
            competitor_prices = [float(price) for price in competitor_prices]
            
            # Tạo dữ liệu đối thủ
            competitors = {}
            for name, price in zip(competitor_names, competitor_prices):
                competitors[name] = price
            
            # Phân tích thị phần dựa trên mô hình
            data = []
            total_q = 0
            
            # Tính số lượng bán dự đoán cho từng đối thủ và tổng
            for name, price in competitors.items():
                price_poly = poly_features.transform(np.array([[price]]))
                q = max(0, poly_model.predict(price_poly)[0])
                r = price * q
                total_q += q
                data.append((name, price, q, r))
            
            # Thêm dữ liệu của shop của mình
            own_shop_name = "Shop của bạn"
            own_q = df_clean['QUANTITY'].mean()
            own_r = base_price * own_q
            total_q += own_q
            data.append((own_shop_name, base_price, own_q, own_r))
            
            # Tính thị phần
            df_comp = pd.DataFrame(data, columns=['Đối thủ', 'Giá', 'SL dự đoán', 'Doanh thu'])
            df_comp['Thị phần (%)'] = (df_comp['SL dự đoán'] / total_q * 100).round(2)
            
            # Hiển thị bảng phân tích
            st.subheader("Phân tích cạnh tranh dựa trên giá")
            st.dataframe(df_comp)
            
            # Vẽ biểu đồ thị phần
            chart1 = alt.Chart(df_comp).mark_arc().encode(
                theta=alt.Theta(field="Thị phần (%)", type="quantitative"),
                color=alt.Color(field="Đối thủ", type="nominal", legend=alt.Legend(title="Đối thủ")),
                tooltip=['Đối thủ', 'Giá', 'SL dự đoán', 'Thị phần (%)']
            ).properties(title="Thị phần dự đoán", width=300, height=300)
            
            # Vẽ biểu đồ so sánh giá
            chart2 = alt.Chart(df_comp).mark_bar().encode(
                x=alt.X('Đối thủ:N', title='Đối thủ'),
                y=alt.Y('Giá:Q', title='Giá'),
                color=alt.Color('Đối thủ:N', legend=None),
                tooltip=['Đối thủ', 'Giá']
            ).properties(title="So sánh giá giữa các đối thủ", width=300, height=300)
            
            # Hiển thị biểu đồ
            st.altair_chart(alt.hconcat(chart1, chart2), use_container_width=True)
            
            # Phân tích vị thế cạnh tranh
            own_position = df_comp[df_comp['Đối thủ'] == own_shop_name].iloc[0]
            competitors_df = df_comp[df_comp['Đối thủ'] != own_shop_name]
            
            # So sánh giá
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
                    st.warning("⚠️ Thị phần thấp. Nên xem xét các yếu tố khác ngoài giá như chất lượng, dịch vụ để tăng khả năng cạnh tranh.")
            else:
                if price_difference < -5:
                    st.success("✅ Thị phần tốt và giá thấp hơn đối thủ. Có thể xem xét tăng giá để tối ưu doanh thu.")
                else:
                    st.success("👍 Vị thế cạnh tranh tốt. Nên duy trì chiến lược hiện tại và theo dõi đối thủ.")
            
        except ValueError as e:
            st.error(f"Lỗi khi xử lý dữ liệu đối thủ: {e}")

# Tab 6: So sánh giá & SL
with tabs[5]:
    st.header("📊 Nhạy Cảm Giá – Dựa trên DateInfo")

    # Chuẩn hóa biến ngày tháng
    df = df_clean.copy()
    
    # Kiểm tra các cột cần thiết có tồn tại không
    date_cols = []
    if 'HOLIDAY' in df.columns:
        df['IS_HOLIDAY'] = df['HOLIDAY'].notna().astype(int)
        date_cols.append('IS_HOLIDAY')
    
    if 'IS_WEEKEND' in df.columns:
        date_cols.append('IS_WEEKEND')
    elif 'CALENDAR_DATE' in df.columns:
        # Tạo cột IS_WEEKEND nếu chưa có
        df['IS_WEEKEND'] = df['CALENDAR_DATE'].dt.weekday >= 5
        date_cols.append('IS_WEEKEND')
    
    if 'IS_SCHOOLBREAK' in df.columns:
        date_cols.append('IS_SCHOOLBREAK')
    
    if 'IS_OUTDOOR' in df.columns:
        date_cols.append('IS_OUTDOOR')
    
    if not date_cols:
        st.warning("Không tìm thấy cột ngày tháng phù hợp trong dữ liệu.")
    else:
        # Tính elasticity theo cách an toàn
        df = df.sort_values('CALENDAR_DATE')
        df['ΔP'] = df['PRICE'].pct_change()
        df['ΔQ'] = df['QUANTITY'].pct_change()
        
        # Loại bỏ các giá trị không hợp lệ và chia 0
        df = df.dropna(subset=['ΔP', 'ΔQ'])
        df = df[df['ΔP'] != 0]  # Tránh chia cho 0
        
        df['Elasticity'] = df['ΔQ'] / df['ΔP']
        
        # Loại bỏ các elasticity quá lớn (outliers)
        Q1_el = df['Elasticity'].quantile(0.25)
        Q3_el = df['Elasticity'].quantile(0.75)
        IQR_el = Q3_el - Q1_el
        lower_bound_el = Q1_el - 3 * IQR_el
        upper_bound_el = Q3_el + 3 * IQR_el
        
        df = df[(df['Elasticity'] >= lower_bound_el) & (df['Elasticity'] <= upper_bound_el)]
        
        # Phân tích elasticity theo từng yếu tố ngày tháng
        records = []
        
        for factor in date_cols:
            try:
                grp = df.groupby(factor)['Elasticity'].mean().reset_index()
                grp.columns = [factor, 'Elasticity_TB']
                
                for _, row in grp.iterrows():
                    factor_value = "Có" if row[factor] == 1 else "Không"
                    records.append({
                        'Yếu tố': factor.replace('IS_', ''),
                        'Giá trị': factor_value,
                        'Elasticity trung bình': round(row['Elasticity_TB'], 2)
                    })
            except Exception as e:
                st.warning(f"Không thể tính elasticity cho yếu tố {factor}: {e}")
        
        if records:
            df_el = pd.DataFrame(records)
            
            # Hiển thị bảng elasticity
            st.subheader("Elasticity trung bình theo yếu tố")
            st.dataframe(df_el)
            
            # Vẽ biểu đồ
            for factor in df_el['Yếu tố'].unique():
                df_plot = df_el[df_el['Yếu tố'] == factor]
                
                chart = alt.Chart(df_plot).mark_bar().encode(
                    x=alt.X('Giá trị:O', title=factor),
                    y=alt.Y('Elasticity trung bình:Q', title='Elasticity trung bình'),
                    color=alt.Color('Giá trị:N', legend=None),
                    tooltip=['Yếu tố', 'Giá trị', 'Elasticity trung bình']
                ).properties(title=f"Elasticity theo {factor}", width=300)
                
                st.altair_chart(chart, use_container_width=True)
            
            # Phân tích kết quả
            st.subheader("Phân tích nhạy cảm giá theo điều kiện")
            
            for factor in df_el['Yếu tố'].unique():
                factor_data = df_el[df_el['Yếu tố'] == factor]
                
                if len(factor_data) >= 2:
                    values = factor_data['Elasticity trung bình'].values
                    diff = abs(values[0] - values[1])
                    
                    if diff > 0.5:
                        st.info(f"📌 **{factor}**: Có sự khác biệt lớn về nhạy cảm giá ({diff:.2f}). Nên xem xét điều chỉnh giá theo yếu tố này.")
                    else:
                        st.write(f"📌 **{factor}**: Không có nhiều sự khác biệt về nhạy cảm giá ({diff:.2f}).")

# Tab 7: Xu hướng theo mùa
with tabs[6]:
    st.header("🌸 Dự đoán xu hướng thay đổi giá theo mùa vụ")
    
    # Phân loại các ngày trong năm theo bốn mùa
    def classify_season(date):
        if date.month in [3, 4, 5]:  # Xuân
            return 'Xuân'
        elif date.month in [6, 7, 8]:  # Hạ
            return 'Hạ'
        elif date.month in [9, 10, 11]:  # Thu
            return 'Thu'
        else:  # Đông
            return 'Đông'

    # Thêm cột Mùa vào dataframe
    df_season = df_clean.copy()
    if 'CALENDAR_DATE' in df_season.columns:
        df_season['Season'] = df_season['CALENDAR_DATE'].apply(classify_season)
        
        # Nhóm dữ liệu theo mùa và tính trung bình giá và số lượng
        season_avg = df_season.groupby('Season').agg({
            'PRICE': 'mean',
            'QUANTITY': 'mean'
        }).reset_index()
        
        season_avg['Revenue'] = season_avg['PRICE'] * season_avg['QUANTITY']
        
        # Thêm các thông tin khác nếu có
        if 'IS_WEEKEND' in df_season.columns:
            season_weekend_avg = df_season[df_season['IS_WEEKEND'] == 1].groupby('Season').agg({
                'PRICE': 'mean',
                'QUANTITY': 'mean'
            }).reset_index()
            season_weekend_avg = season_weekend_avg.rename(columns={
                'PRICE': 'PRICE_WEEKEND',
                'QUANTITY': 'QUANTITY_WEEKEND'
            })
            season_avg = pd.merge(season_avg, season_weekend_avg, on='Season', how='left')
        
        # Hiển thị bảng kết quả
        st.subheader("Phân tích theo mùa")
        st.dataframe(season_avg)
        
        # Vẽ biểu đồ giá theo mùa
        chart1 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('PRICE:Q', title='Giá trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Giá trung bình theo mùa")
        
        # Vẽ biểu đồ số lượng theo mùa
        chart2 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('QUANTITY:Q', title='Số lượng trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Số lượng trung bình theo mùa")
        
        # Vẽ biểu đồ doanh thu theo mùa
        chart3 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('Revenue:Q', title='Doanh thu trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Doanh thu trung bình theo mùa")
        
        # Hiển thị biểu đồ
        st.altair_chart(alt.vconcat(chart1, chart2, chart3), use_container_width=True)
        
        # Đề xuất điều chỉnh giá theo mùa
        st.subheader("Đề xuất điều chỉnh giá theo mùa")
        
        # Tìm mùa có doanh thu cao nhất và thấp nhất
        max_revenue_season = season_avg.loc[season_avg['Revenue'].idxmax()]['Season']
        min_revenue_season = season_avg.loc[season_avg['Revenue'].idxmin()]['Season']
        
        # Tính phần trăm chênh lệch
        max_revenue = season_avg['Revenue'].max()
        min_revenue = season_avg['Revenue'].min()
        revenue_diff_pct = ((max_revenue - min_revenue) / min_revenue) * 100 if min_revenue > 0 else 0
        
        if revenue_diff_pct > 20:
            st.success(f"✅ Có sự chênh lệch đáng kể về doanh thu giữa các mùa ({revenue_diff_pct:.2f}%). Nên xem xét điều chỉnh giá theo mùa.")
            
            # Đề xuất cụ thể cho từng mùa
            for _, row in season_avg.iterrows():
                season = row['Season']
                price = row['PRICE']
                quantity = row['QUANTITY']
                revenue = row['Revenue']
                
                if season == max_revenue_season:
                    st.info(f"📈 **{season}**: Có doanh thu cao nhất. Có thể tăng giá thêm 5-10% để tối ưu lợi nhuận.")
                elif season == min_revenue_season:
                    st.warning(f"📉 **{season}**: Có doanh thu thấp nhất. Nên khuyến mãi hoặc giảm giá 5-10% để kích thích nhu cầu.")
                else:
                    if revenue > season_avg['Revenue'].mean():
                        st.write(f"📊 **{season}**: Doanh thu khá tốt. Có thể giữ nguyên giá hoặc tăng nhẹ 2-5%.")
                    else:
                        st.write(f"📊 **{season}**: Doanh thu dưới mức trung bình. Có thể giảm nhẹ giá 2-5% để kích thích nhu cầu.")
        else:
            st.info(f"ℹ️ Không có sự chênh lệch đáng kể về doanh thu giữa các mùa ({revenue_diff_pct:.2f}%). Có thể giữ nguyên giá xuyên suốt năm.")
    else:
        st.warning("Không tìm thấy dữ liệu ngày tháng để phân tích theo mùa.")

# Tab 8: Phân tích tác động giảm giá
with tabs[7]:
    st.header("📉 Tác Động Của Giảm Giá Đến Lượng Hàng Bán Ra")
    
    # Tạo thanh slider để điều chỉnh mức giảm giá
    discount_pct = st.slider(
        "Mức giảm giá (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Kéo thanh trượt để điều chỉnh mức giảm giá và xem tác động"
    )
    
    # Tính giá sau giảm giá
    discounted_price = base_price * (1 - discount_pct/100)
    
    # Dự đoán số lượng bán với giá giảm
    discounted_price_poly = poly_features.transform(np.array([[discounted_price]]))
    discounted_qty = max(0, poly_model.predict(discounted_price_poly)[0])
    discounted_revenue = discounted_price * discounted_qty
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá sau giảm", f"{discounted_price:.2f}", f"-{discount_pct}%")
    with col2:
        current_qty = df_clean['QUANTITY'].mean()
        qty_change = ((discounted_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        st.metric("Số lượng dự đoán", f"{discounted_qty:.2f}", f"+{qty_change:.2f}%" if qty_change > 0 else f"{qty_change:.2f}%")
    with col3:
        current_revenue = base_price * current_qty
        rev_change = ((discounted_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        st.metric("Doanh thu dự đoán", f"{discounted_revenue:.2f}", f"+{rev_change:.2f}%" if rev_change > 0 else f"{rev_change:.2f}%")
    
    # Tạo bảng các mức giảm giá
    discount_range = range(0, 55, 5)
    results = []
    
    for d in discount_range:
        adj_price = base_price * (1 - d/100)
        adj_price_poly = poly_features.transform(np.array([[adj_price]]))
        adj_qty = max(0, poly_model.predict(adj_price_poly)[0])
        adj_revenue = adj_price * adj_qty
        
        # Tính phần trăm thay đổi so với không giảm giá
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
    
    # Tìm mức giảm giá tối ưu (doanh thu cao nhất)
    result_df = pd.DataFrame(results)
    opt_discount = result_df.loc[result_df['Doanh thu dự đoán'].idxmax()]
    
    # Hiển thị bảng kết quả
    st.subheader("Phân tích các mức giảm giá khác nhau")
    st.dataframe(result_df)
    
    # Hiển thị mức giảm giá tối ưu
    st.success(f"✅ Mức giảm giá tối ưu: **{opt_discount['Giảm giá (%)']}%** - Doanh thu dự đoán: **{opt_discount['Doanh thu dự đoán']:.2f}** (+{opt_discount['Thay đổi doanh thu (%)']:.2f}%)")
    
    # Vẽ biểu đồ
    chart = alt.Chart(result_df).mark_line(point=True).encode(
        x=alt.X('Giảm giá (%):Q', title='Giảm giá (%)'),
        y=alt.Y('Doanh thu dự đoán:Q', title='Doanh thu dự đoán'),
        tooltip=['Giảm giá (%)', 'Giá sau giảm', 'Số lượng dự đoán', 'Doanh thu dự đoán', 'Thay đổi doanh thu (%)']
    ).properties(
        title='Tác động của giảm giá đến doanh thu'
    )
    
    # Đánh dấu điểm tối ưu
    highlight = alt.Chart(pd.DataFrame([opt_discount])).mark_circle(size=100, color='red').encode(
        x='Giảm giá (%):Q', 
        y='Doanh thu dự đoán:Q'
    )
    
    st.altair_chart(chart + highlight, use_container_width=True)
    
    # Phân tích chi tiết
    st.subheader("Phân tích chi tiết")
    
    if opt_discount['Giảm giá (%)'] == 0:
        st.info("ℹ️ Không cần giảm giá - Giá hiện tại đã tối ưu cho doanh thu.")
    elif opt_discount['Giảm giá (%)'] <= 15:
        st.info(f"ℹ️ Mức giảm giá nhẹ ({opt_discount['Giảm giá (%)']}%) có thể tăng doanh thu. Nên xem xét áp dụng cho các chương trình khuyến mãi ngắn hạn.")
    elif opt_discount['Giảm giá (%)'] <= 30:
        st.warning(f"⚠️ Mức giảm giá trung bình ({opt_discount['Giảm giá (%)']}%) có thể tối ưu doanh thu, nhưng cần cân nhắc về lợi nhuận. Phù hợp cho các sự kiện lớn.")
    else:
        st.error(f"❗ Mức giảm giá cao ({opt_discount['Giảm giá (%)']}%) cho thấy giá hiện tại có thể quá cao so với mức chấp nhận của khách hàng. Nên xem xét điều chỉnh giá cơ bản.")

# Tab 9: Tối ưu chương trình khuyến mãi
with tabs[8]:
    st.header("🎯 Tối ưu hóa chương trình khuyến mãi dựa trên giá")
    
    # Tạo các lựa chọn cho loại CTKM
    promo_type = st.radio(
        "Loại chương trình khuyến mãi",
        ["Giảm giá trực tiếp", "Mua 1 tặng 1", "Combo giảm giá", "Giảm giá theo số lượng"],
        horizontal=True
    )
    
    if promo_type == "Giảm giá trực tiếp":
        st.subheader("Giảm giá trực tiếp")
        
        # Giả định về chi phí và lợi nhuận
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_direct_discount"
        )
        cost_price = base_price * (cost_pct / 100)
        
        # Tính toán lợi nhuận cho các mức giảm giá
        profit_results = []
        
        for d in discount_range:
            adj_price = base_price * (1 - d/100)
            adj_price_poly = poly_features.transform(np.array([[adj_price]]))
            adj_qty = max(0, poly_model.predict(adj_price_poly)[0])
            adj_revenue = adj_price * adj_qty
            adj_profit = (adj_price - cost_price) * adj_qty
            
            # Tính phần trăm thay đổi
            current_profit = (base_price - cost_price) * current_qty
            profit_pct_change = ((adj_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
            
            profit_results.append({
                'Giảm giá (%)': d,
                'Giá sau giảm': round(adj_price, 2),
                'Số lượng dự đoán': round(adj_qty, 2),
                'Doanh thu dự đoán': round(adj_revenue, 2),
                'Lợi nhuận dự đoán': round(adj_profit, 2),
                'Thay đổi lợi nhuận (%)': round(profit_pct_change, 2)
            })
        
        # Tìm mức giảm giá tối ưu (lợi nhuận cao nhất)
        profit_df = pd.DataFrame(profit_results)
        opt_profit_discount = profit_df.loc[profit_df['Lợi nhuận dự đoán'].idxmax()]
        
        # Hiển thị bảng kết quả
        st.dataframe(profit_df)
        
        # Hiển thị mức giảm giá tối ưu cho lợi nhuận
        st.success(f"✅ Mức giảm giá tối ưu cho lợi nhuận: **{opt_profit_discount['Giảm giá (%)']}%** - Lợi nhuận dự đoán: **{opt_profit_discount['Lợi nhuận dự đoán']:.2f}** (+{opt_profit_discount['Thay đổi lợi nhuận (%)']:.2f}%)")
        
        # Kiểm tra dữ liệu trước khi vẽ biểu đồ
        st.write("Dữ liệu đầu vào:", profit_df.head())
        st.write("Danh sách cột:", profit_df.columns.tolist())

        melted_df = profit_df.melt(
            id_vars=['Giảm giá (%)'], 
            value_vars=['Doanh thu dự đoán', 'Lợi nhuận dự đoán'], 
            var_name='Chỉ số', 
            value_name='Giá trị'
        )
        # Vẽ biểu đồ so sánh doanh thu và lợi nhuận
        chart = alt.Chart(melted_df).mark_line(point=True).encode(
            x=alt.X('Giảm giá (%):Q', title='Giảm giá (%)'),
            y=alt.Y('Giá trị:Q', title='Giá trị'),
            color=alt.Color('Chỉ số:N', title='Chỉ số'),
            tooltip=['Giảm giá (%)', 'Chỉ số', 'Giá trị']
        ).properties(
            title='So sánh doanh thu và lợi nhuận theo mức giảm giá'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
    elif promo_type == "Mua 1 tặng 1":
        st.subheader("Phân tích chương trình Mua 1 tặng 1")
        
        # Chi phí sản xuất/nhập hàng
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_bogo"
        )
        cost_price = base_price * (cost_pct / 100)
        
        # Tính toán
        effective_discount = 50  # Mua 1 tặng 1 tương đương giảm 50%
        effective_price = base_price * 0.5
        
        # Dự đoán số lượng bán
        effective_price_poly = poly_features.transform(np.array([[effective_price]]))
        effective_qty = max(0, poly_model.predict(effective_price_poly)[0]) * 2  # Nhân 2 vì mỗi đơn hàng là 2 sản phẩm
        
        # Tính doanh thu và lợi nhuận
        effective_revenue = base_price * effective_qty / 2  # Doanh thu chỉ tính trên sản phẩm được bán
        effective_cost = cost_price * effective_qty  # Chi phí tính trên tất cả sản phẩm (cả tặng)
        effective_profit = effective_revenue - effective_cost
        
        # So sánh với không khuyến mãi
        current_revenue = base_price * current_qty
        current_cost = cost_price * current_qty
        current_profit = current_revenue - current_cost
        
        revenue_change = ((effective_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        profit_change = ((effective_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số lượng sản phẩm dự đoán", f"{effective_qty:.2f}", f"+{((effective_qty - current_qty) / current_qty * 100):.2f}%")
            st.metric("Doanh thu dự đoán", f"{effective_revenue:.2f}", f"{revenue_change:.2f}%")
        with col2:
            st.metric("Chi phí dự đoán", f"{effective_cost:.2f}", f"+{((effective_cost - current_cost) / current_cost * 100):.2f}%")
            st.metric("Lợi nhuận dự đoán", f"{effective_profit:.2f}", f"{profit_change:.2f}%")
        
        # Đề xuất
        st.subheader("Đánh giá chương trình")
        if profit_change > 0:
            st.success(f"✅ Chương trình Mua 1 tặng 1 dự kiến làm tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.error(f"❌ Chương trình Mua 1 tặng 1 dự kiến làm giảm lợi nhuận {abs(profit_change):.2f}%. Không nên áp dụng.")
    
    elif promo_type == "Combo giảm giá":
        st.subheader("Phân tích chương trình Combo giảm giá")
        
        # Cho phép người dùng chọn sản phẩm thứ 2 để combo
        second_product = st.selectbox("Chọn sản phẩm thứ 2 cho combo", items)
        combo_discount = st.slider("Giảm giá cho combo (%)", 5, 30, 15, 5)
        
        # Tính toán
        second_price = merged[merged['ITEM_NAME'] == second_product]['PRICE'].mean()
        total_price = base_price + second_price
        combo_price = total_price * (1 - combo_discount / 100)
        
        st.write(f"Giá gốc của hai sản phẩm: {total_price:.2f}")
        st.write(f"Giá combo sau giảm: {combo_price:.2f} (Tiết kiệm: {total_price - combo_price:.2f})")
        
        # Dự đoán số lượng combo bán được
        equivalent_single_price = combo_price / 2  # Giả định giá bình quân mỗi sản phẩm trong combo
        equivalent_price_poly = poly_features.transform(np.array([[equivalent_single_price]]))
        estimated_combo_qty = max(0, poly_model.predict(equivalent_price_poly)[0]) * 0.5  # Giả định 50% khách hàng sẽ mua combo
        
        # Hiển thị dự đoán
        st.metric("Số lượng combo dự đoán", f"{estimated_combo_qty:.2f}")
        st.metric("Doanh thu từ combo", f"{combo_price * estimated_combo_qty:.2f}")

        # Giả định chi phí và tính lợi nhuận
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_combo_discount"
        )
        cost_price_first = base_price * (cost_pct / 100)
        cost_price_second = second_price * (cost_pct / 100)
        total_cost_per_combo = cost_price_first + cost_price_second
        combo_profit = (combo_price - total_cost_per_combo) * estimated_combo_qty

        # So sánh với không khuyến mãi
        current_qty = df_clean['QUANTITY'].mean()
        current_revenue = base_price * current_qty
        current_cost = cost_price_first * current_qty
        current_profit = current_revenue - current_cost

        revenue_change = ((combo_price * estimated_combo_qty - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        profit_change = ((combo_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0

        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Doanh thu combo dự đoán", f"{combo_price * estimated_combo_qty:.2f}", f"{revenue_change:.2f}%")
        with col2:
            st.metric("Lợi nhuận combo dự đoán", f"{combo_profit:.2f}", f"{profit_change:.2f}%")

        # Đề xuất
        st.subheader("Đánh giá chương trình Combo giảm giá")
        if profit_change > 0:
            st.success(f"✅ Chương trình combo giảm giá {combo_discount}% dự kiến tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.warning(f"⚠️ Chương trình combo giảm giá {combo_discount}% dự kiến giảm lợi nhuận {abs(profit_change):.2f}%. Cần cân nhắc thêm.")

    elif promo_type == "Giảm giá theo số lượng":
        st.subheader("Phân tích chương trình Giảm giá theo số lượng")

        # Người dùng chọn số lượng tối thiểu và mức giảm giá
        min_qty = st.number_input("Số lượng tối thiểu để áp dụng giảm giá", min_value=2, value=3, step=1)
        qty_discount_pct = st.slider("Mức giảm giá khi mua từ số lượng tối thiểu (%)", 5, 30, 10, 5)

        # Người dùng nhập chi phí (% giá bán)
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 
            30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_quantity_discount"
        )

        # Tính giá sau giảm
        discounted_price = base_price * (1 - qty_discount_pct / 100)

        # Dự đoán số lượng bán với giá giảm
        discounted_price_poly = poly_features.transform(np.array([[discounted_price]]))
        discounted_qty = max(0, poly_model.predict(discounted_price_poly)[0])

        # Giả định một phần khách hàng sẽ mua số lượng tối thiểu
        estimated_qty = discounted_qty * (min_qty / 2)  # Giả định trung bình mua gấp đôi số lượng tối thiểu
        revenue_qty_discount = discounted_price * estimated_qty
        cost_price = base_price * (cost_pct / 100)
        profit_qty_discount = (discounted_price - cost_price) * estimated_qty

        # So sánh với không khuyến mãi
        current_qty = df_clean['QUANTITY'].mean()
        current_revenue = base_price * current_qty
        current_cost = cost_price * current_qty
        current_profit = current_revenue - current_cost

        revenue_change = ((revenue_qty_discount - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        profit_change = ((profit_qty_discount - current_profit) / current_profit * 100) if current_profit > 0 else 0

        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số lượng dự đoán", f"{estimated_qty:.2f}")
            st.metric("Doanh thu dự đoán", f"{revenue_qty_discount:.2f}", f"{revenue_change:.2f}%")
        with col2:
            st.metric("Lợi nhuận dự đoán", f"{profit_qty_discount:.2f}", f"{profit_change:.2f}%")

        # Đề xuất
        st.subheader("Đánh giá chương trình Giảm giá theo số lượng")
        if profit_change > 0:
            st.success(f"✅ Chương trình giảm giá {qty_discount_pct}% khi mua từ {min_qty} sản phẩm dự kiến tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.warning(f"⚠️ Chương trình giảm giá {qty_discount_pct}% khi mua từ {min_qty} sản phẩm dự kiến giảm lợi nhuận {abs(profit_change):.2f}%. Cần cân nhắc thêm.")

# Tab 10: Đề xuất sản phẩm cần điều chỉnh giá
with tabs[9]:
    st.header("📦 Đề xuất sản phẩm cần điều chỉnh giá")

    # Phân tích tất cả sản phẩm
    product_analysis = merged.groupby('ITEM_NAME').agg({
        'PRICE': 'mean',
        'QUANTITY': 'mean',
        'CALENDAR_DATE': 'count'
    }).reset_index()
    product_analysis['Revenue'] = product_analysis['PRICE'] * product_analysis['QUANTITY']
    product_analysis = product_analysis.rename(columns={'CALENDAR_DATE': 'Số giao dịch'})

    # Tính độ co giãn của cầu cho từng sản phẩm (nếu có đủ dữ liệu)
    elasticity_dict = {}
    for item in product_analysis['ITEM_NAME']:
        df_item = merged[merged['ITEM_NAME'] == item].groupby('PRICE')['QUANTITY'].sum().reset_index()
        if len(df_item) > 1:  # Cần ít nhất 2 mức giá để tính elasticity
            X_item = df_item[['PRICE']].values
            y_item = df_item['QUANTITY'].values
            model_item = LinearRegression().fit(X_item, y_item)
            avg_price_item = df_item['PRICE'].mean()
            avg_qty_item = df_item['QUANTITY'].mean()
            elasticity_dict[item] = abs(model_item.coef_[0] * (avg_price_item / avg_qty_item)) if avg_qty_item > 0 else 0
        else:
            elasticity_dict[item] = None

    product_analysis['Elasticity'] = product_analysis['ITEM_NAME'].map(elasticity_dict)

    # Xác định sản phẩm cần điều chỉnh giá
    low_revenue_threshold = product_analysis['Revenue'].quantile(0.25)
    high_elasticity_threshold = product_analysis['Elasticity'].quantile(0.75, interpolation='nearest') if product_analysis['Elasticity'].notna().sum() > 0 else 1

    product_analysis['Đề xuất'] = product_analysis.apply(
        lambda row: 'Giảm giá' if (row['Revenue'] < low_revenue_threshold and row['Elasticity'] is not None and row['Elasticity'] > high_elasticity_threshold)
        else 'Tăng giá' if (row['Revenue'] < low_revenue_threshold and row['Elasticity'] is not None and row['Elasticity'] < 1)
        else 'Giữ nguyên', axis=1
    )

    # Hiển thị bảng kết quả
    st.subheader("Phân tích và đề xuất điều chỉnh giá sản phẩm")
    st.dataframe(product_analysis)

    # Lọc các sản phẩm cần điều chỉnh
    adjustment_needed = product_analysis[product_analysis['Đề xuất'] != 'Giữ nguyên']
    if not adjustment_needed.empty:
        st.subheader("Sản phẩm cần điều chỉnh giá")
        st.dataframe(adjustment_needed)
    else:
        st.info("ℹ️ Không có sản phẩm nào cần điều chỉnh giá dựa trên dữ liệu hiện tại.")

# Tab 11: Định giá cá nhân hóa (đang phát triển)
with tabs[10]:
    st.header("👤 Định giá cá nhân hóa (Đang phát triển)")

    st.info("Tính năng này đang trong quá trình phát triển. Ý tưởng bao gồm:")
    st.markdown("""
    - Sử dụng dữ liệu khách hàng (nếu có) để phân khúc khách hàng theo hành vi mua sắm.
    - Đề xuất giá khác nhau cho từng nhóm khách hàng dựa trên độ nhạy cảm giá.
    - Tích hợp mô hình học máy để dự đoán mức giá tối ưu cho từng khách hàng.
    """)

    # Possible column names for customer ID
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
        st.write("Các cột hiện có trong dữ liệu merged:", merged.columns.tolist())
        st.info("Để kích hoạt tính năng này, hãy thêm một cột như 'CUSTOMER_ID' vào file transaction.csv.")

# Kết thúc ứng dụng
st.sidebar.success("Phân tích hoàn tất! Chọn tab để xem kết quả.")