import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import uuid

# Hàm giả lập CUSTOMER_ID dựa trên dữ liệu giao dịch
def generate_customer_ids(df, n_customers=100):
    """Tạo CUSTOMER_ID giả lập dựa trên mẫu giao dịch."""
    np.random.seed(42)
    # Giả định số lượng khách hàng là n_customers
    customer_ids = [str(uuid.uuid4())[:8] for _ in range(n_customers)]
    # Phân bổ ngẫu nhiên CUSTOMER_ID cho các giao dịch
    df['CUSTOMER_ID'] = np.random.choice(customer_ids, size=len(df))
    return df

# Hàm tính độ co giãn giá cho từng nhóm khách hàng
def calculate_elasticity(df, group_col, price_col='PRICE', qty_col='QUANTITY'):
    """Tính độ co giãn giá trung bình cho từng nhóm."""
    elasticity_dict = {}
    for group in df[group_col].unique():
        df_group = df[df[group_col] == group].groupby(price_col)[qty_col].sum().reset_index()
        if len(df_group) > 1:
            X = df_group[[price_col]].values
            y = df_group[qty_col].values
            model = LinearRegression().fit(X, y)
            avg_price = df_group[price_col].mean()
            avg_qty = df_group[qty_col].mean()
            elasticity = abs(model.coef_[0] * (avg_price / avg_qty)) if avg_qty > 0 else 0
            elasticity_dict[group] = elasticity
        else:
            elasticity_dict[group] = None
    return elasticity_dict

# Tab 11: Định giá cá nhân hóa
def render_personalized_pricing_tab(merged, df_clean, poly_features, poly_model):
    st.header("👤 Định giá cá nhân hóa")

    # Kiểm tra xem có cột CUSTOMER_ID chưa
    possible_customer_cols = ['CUSTOMER_ID', 'CustomerID', 'customer_id', 'UserID', 'ClientID', 'user_id']
    customer_col = None
    for col in possible_customer_cols:
        if col in merged.columns:
            customer_col = col
            break

    if not customer_col:
        st.info("Không tìm thấy cột CUSTOMER_ID. Đang giả lập dữ liệu khách hàng...")
        # Giả lập CUSTOMER_ID
        merged = generate_customer_ids(merged.copy())
        customer_col = 'CUSTOMER_ID'
    else:
        st.success(f"Đã tìm thấy cột: {customer_col}")

    # Phân tích dữ liệu khách hàng
    try:
        customer_analysis = merged.groupby(customer_col).agg({
            'PRICE': 'mean',
            'QUANTITY': 'sum',
            'CALENDAR_DATE': 'count',
            'ITEM_NAME': 'nunique'
        }).reset_index()
        customer_analysis = customer_analysis.rename(columns={
            'PRICE': 'Giá trung bình',
            'QUANTITY': 'Tổng số lượng',
            'CALENDAR_DATE': 'Số giao dịch',
            'ITEM_NAME': 'Số sản phẩm khác nhau'
        })
        customer_analysis['Doanh thu'] = customer_analysis['Giá trung bình'] * customer_analysis['Tổng số lượng']

        # Hiển thị dữ liệu khách hàng
        st.subheader("Phân tích hành vi khách hàng")
        st.dataframe(customer_analysis.head(10))

        # Phân khúc khách hàng bằng KMeans
        st.subheader("Phân khúc khách hàng")
        n_clusters = st.slider("Số nhóm khách hàng", 2, 5, 3, help="Chọn số lượng nhóm để phân khúc khách hàng")
        features = customer_analysis[['Giá trung bình', 'Tổng số lượng', 'Số giao dịch', 'Số sản phẩm khác nhau']].fillna(0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        customer_analysis['Nhóm'] = kmeans.fit_predict(features).astype(str)

        # Hiển thị đặc điểm từng nhóm
        group_summary = customer_analysis.groupby('Nhóm').agg({
            'Giá trung bình': 'mean',
            'Tổng số lượng': 'mean',
            'Số giao dịch': 'mean',
            'Số sản phẩm khác nhau': 'mean',
            'Doanh thu': 'mean',
            customer_col: 'count'
        }).reset_index()
        group_summary = group_summary.rename(columns={customer_col: 'Số khách hàng'})
        st.dataframe(group_summary)

        # Tính độ co giãn giá cho từng nhóm
        merged_with_groups = merged.merge(customer_analysis[[customer_col, 'Nhóm']], on=customer_col)
        elasticity_dict = calculate_elasticity(merged_with_groups, 'Nhóm')
        group_summary['Độ co giãn giá'] = group_summary['Nhóm'].map(elasticity_dict)

        # Đề xuất giá cá nhân hóa
        st.subheader("Đề xuất giá cá nhân hóa")
        base_price = df_clean['PRICE'].mean()
        recommendations = []
        for _, row in group_summary.iterrows():
            group = row['Nhóm']
            elasticity = row['Độ co giãn giá']
            avg_price = row['Giá trung bình']
            avg_qty = row['Tổng số lượng'] / row['Số khách hàng']
            if elasticity is None:
                recommendation = "Giữ nguyên giá"
                new_price = base_price
            elif elasticity > 1:  # Cầu co giãn
                recommendation = f"Giảm giá ~5-10% (co giãn cao: {elasticity:.2f})"
                new_price = base_price * 0.925  # Giảm 7.5%
            elif elasticity < 0.5:  # Cầu kém co giãn
                recommendation = f"Tăng giá ~5-10% (co giãn thấp: {elasticity:.2f})"
                new_price = base_price * 1.075  # Tăng 7.5%
            else:
                recommendation = f"Giữ nguyên hoặc điều chỉnh nhẹ (co giãn trung bình: {elasticity:.2f})"
                new_price = base_price

            # Dự đoán doanh thu với giá mới
            new_price_poly = poly_features.transform(np.array([[new_price]]))
            new_qty = max(0, poly_model.predict(new_price_poly)[0])
            new_revenue = new_price * new_qty
            current_revenue = avg_price * avg_qty
            revenue_change = ((new_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0

            recommendations.append({
                'Nhóm': group,
                'Số khách hàng': row['Số khách hàng'],
                'Độ co giãn giá': elasticity if elasticity is not None else 'N/A',
                'Giá hiện tại': round(avg_price, 2),
                'Giá đề xuất': round(new_price, 2),
                'Doanh thu dự đoán': round(new_revenue, 2),
                'Thay đổi doanh thu (%)': round(revenue_change, 2),
                'Đề xuất': recommendation
            })

        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df)

        # Biểu đồ so sánh giá hiện tại và giá đề xuất
        melted_df = recommendations_df.melt(
            id_vars=['Nhóm'],
            value_vars=['Giá hiện tại', 'Giá đề xuất'],
            var_name='Loại giá',
            value_name='Giá'
        )
        chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X('Nhóm:N', title='Nhóm khách hàng'),
            y=alt.Y('Giá:Q', title='Giá'),
            color=alt.Color('Loại giá:N', title='Loại giá'),
            xOffset='Loại giá:N',
            tooltip=['Nhóm', 'Loại giá', 'Giá']
        ).properties(
            title='So sánh giá hiện tại và giá đề xuất theo nhóm'
        )
        st.altair_chart(chart, use_container_width=True)

        # Đề xuất chiến lược
        st.subheader("Chiến lược định giá cá nhân hóa")
        for _, row in recommendations_df.iterrows():
            st.write(f"**Nhóm {row['Nhóm']}** ({row['Số khách hàng']} khách hàng): {row['Đề xuất']}")
            st.write(f"- Giá đề xuất: {row['Giá đề xuất']:.2f} (Doanh thu dự đoán: {row['Doanh thu dự đoán']:.2f}, thay đổi: {row['Thay đổi doanh thu (%)']:.2f}%)")

    except Exception as e:
        st.error(f"Lỗi khi phân tích dữ liệu khách hàng: {e}")