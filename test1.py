# app_v6_competitor_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from itertools import combinations

st.set_page_config(page_title="Tối Ưu Giá Bán Cafe & Phân Đối Thủ", layout="wide")
st.title("☕ Ứng dụng Tối Ưu & Phân Tích Giá Bán Cafe Shop (Cạnh Tranh)")

# Sidebar: upload dữ liệu
st.sidebar.header("🚀 Upload dữ liệu")
u_meta  = st.sidebar.file_uploader("Sell Meta Data (CSV)", type="csv")
u_trans = st.sidebar.file_uploader("Transaction Store (CSV)", type="csv")
u_date  = st.sidebar.file_uploader("Date Info (CSV)", type="csv")

if not (u_meta and u_trans and u_date):
    st.sidebar.info("Vui lòng upload cả 3 file để bắt đầu!")
    st.stop()

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
st.dataframe(merged)

# Tiếp tục phần chọn sản phẩm...
items = merged['ITEM_NAME'].dropna().unique().tolist()

# Sidebar: chọn 1 hoặc 2 sản phẩm
items = merged['ITEM_NAME'].dropna().unique().tolist()
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

# Tab UI
tabs = st.tabs([
        "📋 Dữ liệu", "📈 Giá tối ưu", "🔍 Phân tích giá", "💰 Thay đổi giá",
        "🏢 Đối thủ", "📊 So sánh giá & SL", "🛒 Ước lượng doanh thu", "📉 Giảm giá",
        "🎯 Tối ưu CTKM", "📦 Sản phẩm cần điều chỉnh", "👤 Định giá cá nhân hóa"
    ])

# Tab 1
with tabs[0]:
    st.header("📋 Dữ liệu sau khi chọn")
    st.dataframe(df_prod)

# Tab 2
with tabs[1]:
    st.header("📈 Tìm Giá Bán Tối Ưu")
    grp = df_prod.groupby('PRICE')['QUANTITY'].sum().reset_index()
    grp['Revenue'] = grp['PRICE'] * grp['QUANTITY']
    opt = grp.loc[grp['Revenue'].idxmax()]
    st.metric("💵 Giá tối ưu", f"{opt['PRICE']:.2f}")
    st.metric("💰 Doanh thu max", f"{opt['Revenue']:,}")
    st.bar_chart(grp.set_index('PRICE')['Revenue'])

# Tab 3
with tabs[2]:
    st.header("🔎 Phân Tích Giá ↔ Nhu Cầu")
    grp_q = grp[['PRICE','QUANTITY']]
    corr = grp_q['PRICE'].corr(grp_q['QUANTITY'])
    st.write(f"**Hệ số tương quan:** {corr:.2f}")
    scatter = alt.Chart(grp_q).mark_circle().encode(x='PRICE', y='QUANTITY').interactive()
    st.altair_chart(scatter, use_container_width=True)

# Tab 4
with tabs[3]:
    st.header("📊 Tác Động Thay Đổi Giá → Doanh Thu")
    X = grp[['PRICE']].values; y = grp['QUANTITY'].values
    model = LinearRegression().fit(X,y)
    pct = [-0.15,-0.1,-0.05,0,0.05,0.1,0.15]
    base = df_prod['PRICE'].mean()
    res = []
    for p in pct:
        new_p = base*(1+p)
        q = model.predict([[new_p]])[0]
        r = new_p*q
        res.append((f"{int(p*100)}%",round(new_p,2),int(q),int(r)))
    df_res = pd.DataFrame(res, columns=['Δ Giá','Giá','SL','Doanh thu'])
    st.dataframe(df_res)
    st.line_chart(df_res.set_index('Δ Giá')['Doanh thu'])

# Tab 5: Competitor
with tabs[4]:
    st.header("🤝 Phân tích cạnh tranh")
    st.write(f"Sản phẩm/Combo: **{combo_label}**")
    # Tự tạo đối thủ: 3 đối thủ với giá ±5%, ±10%, ±15%
    base_price = df_prod['PRICE'].mean()
    competitors = {
        'Đối thủ A': base_price*0.95,
        'Đối thủ B': base_price*1.05,
        'Đối thủ C': base_price*1.15
    }
    # Tính market share giả định qua giá (ngược giá)
    data = []
    for name, price in competitors.items():
        q = model.predict([[price]])[0]
        r = price*q
        data.append((name,price,int(q),int(r)))
    df_comp = pd.DataFrame(data, columns=['Đối thủ','Giá','SL dự đoán','Doanh thu'])
    st.dataframe(df_comp)
    chart = alt.Chart(df_comp).mark_bar().encode(
        x='Đối thủ', y='Doanh thu')
    st.altair_chart(chart, use_container_width=True)
# Tab 6: Nhạy Cảm Giá – Dựa trên DateInfo
with tabs[5]:
    st.header("💡 Nhạy Cảm Giá – Dựa trên DateInfo")

    # 1. Chuẩn hóa biến Holiday thành nhị phân
    df = df_prod.copy()
    df['IS_HOLIDAY'] = df['HOLIDAY'].notna().astype(int)

    # Các cột nhị phân đã có: IS_WEEKEND, IS_SCHOOLBREAK, IS_OUTDOOR
    # Kiểm tra tồn tại
    factors = [c for c in ['IS_HOLIDAY', 'IS_WEEKEND', 'IS_SCHOOLBREAK', 'IS_OUTDOOR'] if c in df.columns]
    
    if not factors:
        st.warning("Không tìm thấy cột Holiday/Weekend/Schoolbreak/Outdoor trong dữ liệu.")
        st.stop()

    # 2. Tính ΔP, ΔQ, Elasticity
    df = df.sort_values('CALENDAR_DATE')
    df['ΔP'] = df['PRICE'].pct_change()
    df['ΔQ'] = df['QUANTITY'].pct_change()
    df = df.dropna(subset=['ΔP', 'ΔQ'])
    df['Elasticity'] = df['ΔQ'] / df['ΔP']

    # 3. Nhóm theo từng factor và tính Elasticity trung bình
    records = []
    for f in factors:
        grp = df.groupby(f)['Elasticity'].mean().reset_index()
        grp.columns = [f, 'Elasticity_TB']
        for _, row in grp.iterrows():
            records.append({
                'Yếu tố': f,
                'Giá trị': int(row[f]),
                'Elasticity trung bình': round(row['Elasticity_TB'], 2)
            })

    df_el = pd.DataFrame(records)
    st.subheader("Elasticity trung bình theo yếu tố")
    st.dataframe(df_el)

    # 4. Vẽ biểu đồ cho mỗi factor
    for f in factors:
        df_plot = df_el[df_el['Yếu tố'] == f]
        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X('Giá trị:O', title=f),
                y=alt.Y('Elasticity trung bình:Q'),
                tooltip=['Giá trị', 'Elasticity trung bình']
            )
            .properties(title=f"Elasticity theo {f}")
        )
        st.altair_chart(chart, use_container_width=True)
# Tab 7: Dự đoán xu hướng thay đổi giá theo mùa vụ
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
    df_prod['Season'] = df_prod['CALENDAR_DATE'].apply(classify_season)
    
    # Nhóm dữ liệu theo mùa và tính trung bình giá
    season_avg_price = df_prod.groupby('Season')['PRICE'].mean().reset_index()
    
    # Thêm các yếu tố khác để huấn luyện mô hình
    df_prod['IS_WEEKEND'] = df_prod['CALENDAR_DATE'].dt.weekday >= 5  # Cuối tuần (Thứ 7, Chủ nhật)
    df_prod['IS_HOLIDAY'] = df_prod['HOLIDAY'].notna().astype(int)  # Ngày lễ
    
    # Tạo cột mùa vụ dưới dạng biến giả (dummy variables) cho dữ liệu huấn luyện
    df_prod_dummies = pd.get_dummies(df_prod[['Season', 'IS_WEEKEND', 'IS_HOLIDAY']], drop_first=True)
    
    # Sử dụng Linear Regression để huấn luyện mô hình
    X_season = df_prod_dummies.values
    y_price = df_prod['PRICE'].values
    
    # Dự đoán giá theo mùa sử dụng Linear Regression
    model = LinearRegression().fit(X_season, y_price)
    
    # Thêm các cột cần thiết vào season_avg_price để phù hợp với dữ liệu huấn luyện
    season_avg_price['IS_WEEKEND'] = 0  # Giả sử không phải cuối tuần
    season_avg_price['IS_HOLIDAY'] = 0  # Giả sử không phải ngày lễ
    
    # Chuyển đổi cột 'Season' thành các biến giả (dummy variables)
    season_avg_price_dummies = pd.get_dummies(season_avg_price[['Season', 'IS_WEEKEND', 'IS_HOLIDAY']], drop_first=True)
    
    # Dự đoán giá cho từng mùa
    predicted_prices = model.predict(season_avg_price_dummies.values)
    season_avg_price['Predicted_Price'] = predicted_prices

    # Hiển thị kết quả dự đoán
    st.subheader("Dự đoán giá theo mùa")
    st.dataframe(season_avg_price)

    # Vẽ biểu đồ so sánh giá thực tế và giá dự đoán theo mùa
    chart_comparison = alt.Chart(season_avg_price).mark_bar().encode(
        x='Season:O', 
        y='Predicted_Price:Q',
        color='Season:O',
        tooltip=['Season', 'Predicted_Price']
    ).properties(title="So sánh giá dự đoán theo mùa vụ")
    st.altair_chart(chart_comparison, use_container_width=True)
# Tab 8: Phân tích tác động giảm giá tới lượng hàng bán
with tabs[7]:
    st.header("📊 Tác Động Của Giảm Giá Đến Lượng Hàng Bán Ra")
    
    # Dữ liệu giá và lượng bán
    grp_q = df_prod[['PRICE', 'QUANTITY']].dropna()
    
    # Tạo mô hình hồi quy tuyến tính để dự đoán sự thay đổi của số lượng bán khi thay đổi giá
    X = grp_q[['PRICE']].values  # Giá
    y = grp_q['QUANTITY'].values  # Lượng bán
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Hiển thị hệ số hồi quy (slope)
    st.write(f"**Hệ số hồi quy (slope)**: {model.coef_[0]:.2f}")
    
    # Dự đoán lượng bán khi thay đổi giá (giảm giá 5%, 10%, 15%)
    percentage_changes = [-0.05, -0.10, -0.15]
    price_changes = [base_price * (1 + p) for p in percentage_changes]
    predicted_sales = model.predict(np.array(price_changes).reshape(-1, 1))
    
    # Tạo bảng dữ liệu kết quả phân tích
    results = pd.DataFrame({
        'Giảm Giá (%)': [int(p * 100) for p in percentage_changes],
        'Giá Sau Giảm (%)': [round(base_price * (1 + p), 2) for p in percentage_changes],
        'Lượng Bán Dự Đoán': [int(s) for s in predicted_sales]
    })
    
    st.subheader("Kết quả tác động giảm giá tới lượng bán")
    st.dataframe(results)
    
    # Vẽ biểu đồ tác động giảm giá tới lượng bán
    chart = alt.Chart(results).mark_bar().encode(
        x='Giảm Giá (%):O',
        y='Lượng Bán Dự Đoán:Q',
        color='Giảm Giá (%):O',
        tooltip=['Giảm Giá (%)', 'Lượng Bán Dự Đoán']
    ).properties(title="Tác Động Của Giảm Giá Đến Lượng Hàng Bán Ra")
    
    st.altair_chart(chart, use_container_width=True)
# Tab 9: Tối ưu chương trình khuyến mãi
    with tabs[8]:
        st.header("🎯 Tối ưu hóa chương trình khuyến mãi dựa trên giá")
        discount_range = np.arange(0, 55, 5)
        promo_df = pd.DataFrame()
        for d in discount_range:
            adj_price = df['PRICE'].mean() * (1 - d/100)
            adj_quantity = df['QUANTITY'].mean() * (1 + d/100)
            rev = adj_price * adj_quantity
            promo_df = pd.concat([promo_df, pd.DataFrame({
                'Giảm giá (%)': [d],
                'Giá sau giảm': [adj_price],
                'Dự đoán SL': [adj_quantity],
                'Doanh thu': [rev]
            })])
        chart = alt.Chart(promo_df).mark_line(point=True).encode(
            x=alt.X('Giảm giá (%):Q'),
            y=alt.Y('Doanh thu:Q'),
            tooltip=['Giảm giá (%)', 'Giá sau giảm', 'Dự đoán SL', 'Doanh thu']
        ).properties(title="📈 Doanh thu dự đoán theo mức giảm giá")
        st.altair_chart(chart, use_container_width=True)

    # Tab 10: Sản phẩm cần điều chỉnh giá
    with tabs[9]:
        st.header("📦 Sản phẩm cần điều chỉnh giá")
        grouped = df.groupby('ITEM_NAME').agg({'PRICE': 'mean', 'QUANTITY': 'mean'}).reset_index()
        grouped['Đề xuất'] = grouped.apply(
            lambda row: 'Giảm giá' if row['QUANTITY'] < df['QUANTITY'].mean() and row['PRICE'] > df['PRICE'].mean() else
                        'Tăng giá' if row['QUANTITY'] > df['QUANTITY'].mean() * 1.2 else
                        'Giữ nguyên', axis=1)
        st.dataframe(grouped[['ITEM_NAME', 'PRICE', 'QUANTITY', 'Đề xuất']])

    # Tab 11: Định giá cá nhân hóa
    with tabs[10]:
        st.header("👤 Định giá cá nhân hóa cho khách hàng")
        st.markdown("Giả sử có 3 nhóm khách hàng: Nhạy cảm giá, Trung bình, Cao cấp")
        customer_segments = pd.DataFrame({
            'Nhóm': ['Nhạy cảm', 'Trung bình', 'Cao cấp'],
            'Ưu đãi (%)': [20, 10, 0],
        })
        customer_segments['Giá đề xuất'] = df['PRICE'].mean() * (1 - customer_segments['Ưu đãi (%)'] / 100)
        st.dataframe(customer_segments)

        chart = alt.Chart(customer_segments).mark_bar().encode(
            x=alt.X('Nhóm:N'),
            y=alt.Y('Giá đề xuất:Q'),
            color='Nhóm:N',
            tooltip=['Nhóm', 'Ưu đãi (%)', 'Giá đề xuất']
        ).properties(title="💡 Giá đề xuất theo nhóm khách hàng")
        st.altair_chart(chart, use_container_width=True)