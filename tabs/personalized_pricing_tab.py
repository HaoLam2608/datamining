import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import uuid

# Hàm giả lập CUSTOMER_ID
def generate_customer_ids(df, n_customers=100):
    """Tạo CUSTOMER_ID giả lập dựa trên mẫu giao dịch."""
    np.random.seed(42)
    customer_ids = [str(uuid.uuid4())[:8] for _ in range(n_customers)]
    df['CUSTOMER_ID'] = np.random.choice(customer_ids, size=len(df))
    return df

# Hàm tính độ co giãn giá
def calculate_elasticity(df, group_col, price_col='PRICE', qty_col='QUANTITY'):
    """Tính độ co giãn giá trung bình cho từng nhóm."""
    elasticity_dict = {}
    for group in df[group_col].unique():
        df_group = df[df[group_col] == group].dropna(subset=[price_col, qty_col])
        if len(df_group[price_col].unique()) > 1:
            X = df_group[[price_col]].values
            y = df_group[qty_col].values
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            avg_price = df_group[price_col].mean()
            pred_qty = model.predict(poly.transform([[avg_price]]))[0]
            coef = model.coef_[1] + 2 * model.coef_[2] * avg_price  # Đạo hàm đa thức bậc 2
            elasticity = abs(coef * avg_price / pred_qty) if pred_qty != 0 else 0
            elasticity = min(elasticity, 5.0)  # Giới hạn tối đa là 5
            elasticity_dict[group] = elasticity
        else:
            elasticity_dict[group] = None
    return elasticity_dict

# Hàm phân tích và định giá cá nhân hóa
def analyze_personalized_pricing(df_clean):
    st.header("👤 Phân tích và định giá cá nhân hóa")
    
    # Kiểm tra cột CUSTOMER_ID
    possible_customer_cols = ['CUSTOMER_ID', 'CustomerID', 'customer_id', 'UserID', 'ClientID', 'user_id']
    customer_col = None
    for col in possible_customer_cols:
        if col in df_clean.columns:
            customer_col = col
            break
    
    if not customer_col:
        with st.spinner("Không tìm thấy cột CUSTOMER_ID. Đang giả lập dữ liệu khách hàng..."):
            df_clean = generate_customer_ids(df_clean.copy())
            customer_col = 'CUSTOMER_ID'
            st.info("Đã tạo CUSTOMER_ID giả lập cho phân tích")
    else:
        st.success(f"Đã tìm thấy cột: {customer_col}")
    
    # Xây dựng mô hình đa thức cho dự đoán số lượng
    df_for_model = df_clean[['PRICE', 'QUANTITY']].dropna()
    X = df_for_model[['PRICE']].values
    y = df_for_model['QUANTITY'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression().fit(X_poly, y)
    
    # Phân tích dữ liệu khách hàng
    try:
        # Tổng quan dữ liệu
        st.subheader("Tổng quan dữ liệu")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số khách hàng", df_clean[customer_col].nunique())
        col2.metric("Tổng số giao dịch", len(df_clean))
        col3.metric("Giá trung bình", f"{df_clean['PRICE'].mean():.2f}")
        col4.metric("Tổng doanh thu", f"{(df_clean['PRICE'] * df_clean['QUANTITY']).sum():.2f}")
        
        # Phân tích hành vi khách hàng
        customer_analysis = df_clean.groupby(customer_col).agg({
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
        
        st.subheader("Phân tích hành vi khách hàng")
        st.dataframe(customer_analysis.head(10), use_container_width=True)
        
        # Phân khúc khách hàng
        st.subheader("Phân khúc khách hàng")
        n_clusters = st.slider("Số nhóm khách hàng", 2, 5, 3, help="Chọn số lượng nhóm để phân khúc khách hàng")
        
        features = customer_analysis[['Giá trung bình', 'Tổng số lượng', 'Số giao dịch', 'Doanh thu']].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        with st.spinner("Đang phân khúc khách hàng..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Tạo tên nhóm dựa trên đặc điểm
        group_summary = customer_analysis.groupby(cluster_labels).agg({
            'Giá trung bình': 'mean',
            'Tổng số lượng': 'mean',
            'Số giao dịch': 'mean',
            'Số sản phẩm khác nhau': 'mean',
            'Doanh thu': 'mean',
            customer_col: 'count'
        }).reset_index()
        group_summary = group_summary.rename(columns={customer_col: 'Số khách hàng', 'index': 'Cluster'})
        
        # Tính trung bình toàn bộ để so sánh
        overall_means = customer_analysis[['Giá trung bình', 'Tổng số lượng', 'Số giao dịch', 'Doanh thu']].mean()
        
        # Gán tên nhóm bằng tiếng Việt
        group_names = []
        for idx, row in group_summary.iterrows():
            price = row['Giá trung bình']
            qty = row['Tổng số lượng']
            freq = row['Số giao dịch']
            revenue = row['Doanh thu']
            
            is_high_price = price > overall_means['Giá trung bình'] * 1.1
            is_low_price = price < overall_means['Giá trung bình'] * 0.9
            is_high_qty = qty > overall_means['Tổng số lượng'] * 1.1
            is_low_qty = qty < overall_means['Tổng số lượng'] * 0.9
            is_high_freq = freq > overall_means['Số giao dịch'] * 1.1
            is_low_freq = freq < overall_means['Số giao dịch'] * 0.9
            is_high_revenue = revenue > overall_means['Doanh thu'] * 1.1
            is_low_revenue = revenue < overall_means['Doanh thu'] * 0.9
            
            if is_high_price and is_high_revenue and not is_low_qty:
                name = "Khách Hàng Cao Cấp Trung Thành"
            elif is_low_price and is_high_qty:
                name = "Khách Hàng Nhạy Cảm Giá Mua Số Lượng Lớn"
            elif is_low_qty and is_low_freq:
                name = "Khách Hàng Thỉnh Thoảng Chi Tiêu Thấp"
            elif is_high_freq and not is_low_revenue:
                name = "Khách Hàng Thường Xuyên Mua Sắm Đều Đặn"
            else:
                name = "Khách Hàng Cân Bằng Cấp Trung"
            group_names.append(name)
        
        cluster_to_name = {str(i): name for i, name in enumerate(group_names)}
        customer_analysis['Nhóm'] = [cluster_to_name[str(label)] for label in cluster_labels]
        group_summary['Nhóm'] = group_names
        
        st.dataframe(group_summary.drop(columns=['Cluster']).style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Biểu đồ phân khúc
        st.subheader("Biểu đồ phân khúc khách hàng")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Trục X", ['Giá trung bình', 'Tổng số lượng', 'Số giao dịch', 'Doanh thu'], index=0)
        with col2:
            y_axis = st.selectbox("Trục Y", ['Giá trung bình', 'Tổng số lượng', 'Số giao dịch', 'Doanh thu'], index=1)
        
        scatter_chart = alt.Chart(customer_analysis).mark_circle(size=60).encode(
            x=alt.X(x_axis, title=x_axis),
            y=alt.Y(y_axis, title=y_axis),
            color=alt.Color('Nhóm:N', title='Nhóm khách hàng'),
            tooltip=[customer_col, 'Nhóm', x_axis, y_axis, 'Doanh thu']
        ).properties(
            title=f'Phân khúc khách hàng theo {x_axis} và {y_axis}',
            width=600,
            height=400
        ).interactive()
        
        st.altair_chart(scatter_chart, use_container_width=True)
        
        # Độ co giãn giá
        merged_with_groups = df_clean.merge(customer_analysis[[customer_col, 'Nhóm']], on=customer_col, how='left')
        elasticity_dict = calculate_elasticity(merged_with_groups, 'Nhóm')
        group_summary['Độ co giãn giá'] = group_summary['Nhóm'].map(elasticity_dict)
        
        # Đề xuất giá cá nhân hóa
        st.subheader("Đề xuất giá cá nhân hóa")
        
        recommendations = []
        damping_factor = 0.5  # Hệ số giảm tác động của độ co giãn
        max_qty_change = 0.5  # Giới hạn thay đổi số lượng tối đa 50%
        
        for _, row in group_summary.iterrows():
            group = row['Nhóm']
            elasticity = row['Độ co giãn giá']
            current_price = row['Giá trung bình']
            avg_qty = row['Tổng số lượng']
            num_customers = row['Số khách hàng']
            current_revenue = row['Doanh thu'] * num_customers  # Doanh thu tổng hiện tại
            
            if elasticity is None or pd.isna(elasticity):
                recommendation = "Giữ nguyên giá do thiếu dữ liệu độ co giãn"
                new_price = current_price
            else:
                if elasticity > 1:
                    recommendation = f"Giảm giá ~5-10% (co giãn cao: {elasticity:.2f})"
                    new_price = current_price * 0.925  # Giảm 7.5%
                elif elasticity < 0.5:
                    recommendation = f"Tăng giá ~5-10% (co giãn thấp: {elasticity:.2f})"
                    new_price = current_price * 1.075  # Tăng 7.5%
                else:
                    recommendation = f"Giữ nguyên hoặc điều chỉnh nhẹ (co giãn trung bình: {elasticity:.2f})"
                    new_price = current_price
            
            # Dự đoán số lượng mới dựa trên độ co giãn
            price_change = (new_price - current_price) / current_price if current_price > 0 else 0
            qty_change = elasticity * price_change * damping_factor
            qty_change = max(min(qty_change, max_qty_change), -max_qty_change)  # Giới hạn thay đổi số lượng
            predicted_qty = avg_qty * (1 + qty_change) if elasticity is not None else avg_qty
            predicted_qty = max(0, predicted_qty)  # Đảm bảo số lượng không âm
            
            # Tính doanh thu dự đoán cho toàn nhóm
            predicted_revenue = new_price * predicted_qty * num_customers
            revenue_change = ((predicted_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
            
            recommendations.append({
                'Nhóm': group,
                'Số khách hàng': num_customers,
                'Độ co giãn giá': elasticity if elasticity is not None and not pd.isna(elasticity) else 'N/A',
                'Giá hiện tại': round(current_price, 2),
                'Giá đề xuất': round(new_price, 2),
                'Doanh thu dự đoán': round(predicted_revenue, 2),
                'Thay đổi doanh thu (%)': round(revenue_change, 2),
                'Đề xuất': recommendation
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df.style.highlight_max(axis=0, subset=['Thay đổi doanh thu (%)']), use_container_width=True)
        
        # Biểu đồ giá
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
            title='So sánh giá hiện tại và giá đề xuất theo nhóm',
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Biểu đồ doanh thu
        revenue_chart = alt.Chart(recommendations_df).mark_bar().encode(
            x=alt.X('Nhóm:N', title='Nhóm khách hàng'),
            y=alt.Y('Thay đổi doanh thu (%):Q', title='Thay đổi doanh thu (%)'),
            color=alt.condition(
                alt.datum['Thay đổi doanh thu (%)'] > 0,
                alt.value('green'),
                alt.value('red')
            ),
            tooltip=['Nhóm', 'Thay đổi doanh thu (%)', 'Giá đề xuất', 'Doanh thu dự đoán']
        ).properties(
            title='Dự đoán thay đổi doanh thu theo nhóm',
            width=600,
            height=400
        )
        st.altair_chart(revenue_chart, use_container_width=True)
        
        # Đề xuất chiến lược
        st.subheader("Chiến lược định giá cá nhân hóa")
        st.write("Dựa trên phân tích độ co giãn giá của từng phân khúc khách hàng, chúng tôi đề xuất:")
        
        for _, row in recommendations_df.iterrows():
            with st.expander(f"{row['Nhóm']} ({row['Số khách hàng']} khách hàng)"):
                st.write(f"**Đề xuất:** {row['Đề xuất']}")
                st.write(f"- Giá hiện tại: {row['Giá hiện tại']:.2f}")
                st.write(f"- Giá đề xuất: {row['Giá đề xuất']:.2f}")
                st.write(f"- Doanh thu dự đoán: {row['Doanh thu dự đoán']:.2f}")
                st.write(f"- Thay đổi doanh thu: {row['Thay đổi doanh thu (%)']:.2f}%")
                
                if row['Độ co giãn giá'] == 'N/A' or pd.isna(row['Độ co giãn giá']):
                    st.write("**Chiến lược marketing:** Thu thập thêm dữ liệu để phân tích độ co giãn giá")
                elif row['Độ co giãn giá'] > 1:
                    st.write("**Chiến lược marketing:** Nhóm này nhạy cảm với giá. Nên tập trung vào các chương trình khuyến mãi, giảm giá theo số lượng, hoặc gói combo tiết kiệm.")
                elif row['Độ co giãn giá'] < 0.5:
                    st.write("**Chiến lược marketing:** Nhóm này ít nhạy cảm với giá. Nên tập trung vào chất lượng sản phẩm, dịch vụ khách hàng, và trải nghiệm cao cấp.")
                else:
                    st.write("**Chiến lược marketing:** Nhóm có độ nhạy cảm giá trung bình. Cân bằng giữa yếu tố giá và giá trị cung cấp.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")
        return

# Hàm chính
def render_personalized_pricing_tab():
    with st.sidebar:
        st.header("Hướng dẫn")
        st.markdown("""
        - Tải lên file customer_data.csv.
        - File phải chứa các cột: CUSTOMER_ID, ITEM_NAME, PRICE, QUANTITY, CALENDAR_DATE.
        - Xem kết quả phân khúc khách hàng và đề xuất giá cá nhân hóa.
        """)
    
    # Tải file customer_data
    st.subheader("Tải lên dữ liệu")
    customer_file = st.file_uploader("Tải lên customer_data.csv", type=["csv"])
    
    if not customer_file:
        st.warning("Vui lòng tải lên file customer_data.csv để tiếp tục.")
        return
    
    try:
        # Đọc file
        df_clean = pd.read_csv(customer_file)
        
        # Chuẩn hóa tên cột
        df_clean.columns = df_clean.columns.str.strip().str.upper()
        
        # Đảm bảo các cột cần thiết
        required_columns = ['CUSTOMER_ID', 'ITEM_NAME', 'PRICE', 'QUANTITY', 'CALENDAR_DATE']
        if not all(col in df_clean.columns for col in required_columns):
            st.error("File customer_data.csv phải chứa các cột: CUSTOMER_ID, ITEM_NAME, PRICE, QUANTITY, CALENDAR_DATE.")
            return
        
        # Chuyển đổi kiểu dữ liệu
        df_clean['CALENDAR_DATE'] = pd.to_datetime(df_clean['CALENDAR_DATE'], errors='coerce')
        df_clean['PRICE'] = pd.to_numeric(df_clean['PRICE'], errors='coerce')
        df_clean['QUANTITY'] = pd.to_numeric(df_clean['QUANTITY'], errors='coerce')
        
        # Làm sạch dữ liệu
        df_clean = df_clean.dropna(subset=['PRICE', 'QUANTITY', 'CALENDAR_DATE', 'ITEM_NAME'])
        df_clean = df_clean[df_clean['PRICE'] > 0]
        df_clean = df_clean[df_clean['QUANTITY'] > 0]
        
        if df_clean.empty:
            st.warning("Dữ liệu sau khi làm sạch rỗng. Vui lòng kiểm tra file đầu vào.")
            return
        
        # Gọi hàm phân tích
        analyze_personalized_pricing(df_clean)
        
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc và xử lý file: {e}")
        return