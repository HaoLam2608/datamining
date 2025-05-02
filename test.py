from analyze_price_sensitivity.combo import analyze_price_sensitivity_factors1
from analyze_price_sensitivity.one_product import analyze_price_sensitivity_factors
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from forecast_seasonal_price.combo import forecast_seasonal_price1
from forecast_seasonal_price.one_product import forecast_seasonal_price


# ------------------------
# Hàm tải và xử lý dữ liệu
# ------------------------
@st.cache_data
def load_data(sell_meta, transactions, date_info):
    """Đọc và kết hợp dữ liệu từ ba file CSV."""
    try:
        sell_meta_df = pd.read_csv(sell_meta)
        transactions_df = pd.read_csv(transactions)
        date_info_df = pd.read_csv(date_info)

        # Chuẩn hóa tên cột
        column_mapping = {
            "CALENDAR_DATE": "Date",
            "Transaction_Date": "Date",
            "ITEM_NAME": "Product",
            "SALES_QUANTITY": "QUANTITY",
            "PRICE": "Price"
        }
        
        # Áp dụng mapping cho từng DataFrame
        for df in [transactions_df, date_info_df, sell_meta_df]:
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)

        # Kiểm tra cột Date cần thiết
        required_columns = {
            'transactions': ['Date', 'SELL_ID', 'SELL_CATEGORY', 'QUANTITY', 'Price'],
            'date_info': ['Date'],
            'sell_meta': ['SELL_ID', 'SELL_CATEGORY', 'Product']
        }
        
        for df_name, cols in required_columns.items():
            df = locals()[f"{df_name}_df"]
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                st.error(f"Thiếu các cột cần thiết trong {df_name}: {', '.join(missing_cols)}")
                return pd.DataFrame()

        # Loại bỏ dữ liệu null
        sell_meta_df.dropna(subset=['SELL_ID', 'SELL_CATEGORY', 'Product'], inplace=True)
        transactions_df.dropna(subset=['Date', 'SELL_ID', 'SELL_CATEGORY', 'QUANTITY', 'Price'], inplace=True)
        date_info_df.dropna(subset=['Date'], inplace=True)

        # Merge dữ liệu
        merged_df = transactions_df.merge(sell_meta_df, on=["SELL_ID", "SELL_CATEGORY"], how="left")
        merged_df = merged_df.merge(date_info_df, on="Date", how="left")
        
        # Chuyển đổi Date sang định dạng datetime
        merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
        # Loại bỏ các dòng có Date không hợp lệ
        merged_df.dropna(subset=['Date'], inplace=True)
        
        return merged_df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()

# ------------------------
# Hàm tạo combo từ các sản phẩm
# ------------------------
def create_combo(data, combo_products):
    """Tạo dữ liệu combo từ các sản phẩm được chọn."""
    if not combo_products or len(combo_products) < 2:
        st.warning("Cần chọn ít nhất 2 sản phẩm để tạo combo.")
        return pd.DataFrame()
        
    combo_data = data[data['Product'].isin(combo_products)].copy()
    if combo_data.empty:
        st.warning("Không tìm thấy dữ liệu cho các sản phẩm được chọn.")
        return pd.DataFrame()
    
    # Tính tổng số lượng và giá trung bình theo ngày và sản phẩm
    daily_data = combo_data.groupby(['Date', 'Product']).agg({
        'QUANTITY': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    # Tạo pivot tables cho số lượng và giá
    pivot_quantity = daily_data.pivot(index='Date', columns='Product', values='QUANTITY').fillna(0)
    pivot_price = daily_data.pivot(index='Date', columns='Product', values='Price').fillna(0)
    
    # Tính toán số lượng combo (lấy giá trị nhỏ nhất giữa các sản phẩm)
    combo_quantity = pivot_quantity.min(axis=1)
    
    # Tính giá combo (tổng giá các sản phẩm)
    combo_price = pivot_price.sum(axis=1)
    
    # Tạo DataFrame kết quả
    combo_df = pd.DataFrame({
        'Date': combo_quantity.index,
        'Combo_Quantity': combo_quantity.values,
        'Combo_Price': combo_price.values
    })
    
    # Thêm cột doanh thu
    combo_df['Combo_Revenue'] = combo_df['Combo_Quantity'] * combo_df['Combo_Price']
    
    return combo_df

# ------------------------
# Hàm xây dựng mô hình định giá với hồi quy đa thức
# ------------------------
def build_pricing_model(data, price_col='Price', quantity_col='QUANTITY', degree=2):
    """Xây dựng mô hình hồi quy đa thức để dự đoán giá tối ưu."""
    if data.empty or len(data) < degree + 1:
        return None, None, None, np.nan

    # Loại bỏ các dòng có giá trị NaN hoặc vô cực
    valid_data = data[[price_col, quantity_col]].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) < degree + 1:
        return None, None, None, np.nan

    X = valid_data[[price_col]].values
    y = valid_data[quantity_col].values

    # Xây dựng mô hình
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

    # Giới hạn miền giá dựa trên dữ liệu
    min_price = valid_data[price_col].min()
    max_price = valid_data[price_col].max()
    price_range = (max_price - min_price) * 0.2  # Mở rộng khoảng giá 20%
    search_min = max(min_price - price_range, 0)  # Giá không âm
    search_max = max_price + price_range

    # Hàm tính doanh thu dự kiến (đảo dấu để sử dụng với minimize_scalar)
    def revenue_function(p):
        predicted_quantity = model.predict(np.array([[p]]))[0]
        # Đảm bảo số lượng không âm
        predicted_quantity = max(0, predicted_quantity)
        return -p * predicted_quantity  # Đảo dấu để tìm cực đại

    # Tìm giá tối ưu
    try:
        result = minimize_scalar(revenue_function, bounds=(search_min, search_max), method='bounded')
        optimal_price = result.x
        
        # Tính hệ số R-squared
        y_pred = model.predict(X)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Tính các coeff của mô hình
        coefficients = model.named_steps['linearregression'].coef_
        intercept = model.named_steps['linearregression'].intercept_
    except Exception as e:
        st.error(f"Lỗi khi tối ưu hóa giá: {e}")
        return model, None, None, np.nan

    return model, r_squared, coefficients, optimal_price

# ------------------------
# Hàm phân tích mối quan hệ giữa giá và nhu cầu
# ------------------------
def analyze_price_demand_relationship(data, product):
    """Phân tích mối quan hệ giữa giá và nhu cầu sản phẩm."""
    if 'Product' in data.columns:
        product_data = data[data['Product'] == product].copy()
    else:
        product_data = data.copy()  # Nếu không có cột Product, sử dụng tất cả dữ liệu
    
    if len(product_data) < 3:  # Cần ít nhất 3 điểm dữ liệu để đánh giá
        return None, "Không đủ dữ liệu để phân tích. Cần ít nhất 3 điểm dữ liệu."

    # Phân tích trên dữ liệu đã tổng hợp theo giá
    price_grouped = product_data.groupby('Price').agg({'QUANTITY': 'mean'}).reset_index()
    
    if len(price_grouped) < 3:
        return None, "Không đủ mức giá khác nhau để phân tích. Cần ít nhất 3 mức giá."
    
    model, r_squared, coeffs, _ = build_pricing_model(price_grouped)
    
    if not model:
        return None, "Không thể xây dựng mô hình để phân tích."
    
    analysis_result = {
        'model': model,
        'r_squared': r_squared,
        'data': price_grouped
    }
    
    insight = "Xem biểu đồ để hiểu mối quan hệ giữa giá và nhu cầu."
    if r_squared is not None:
        insight += f" Độ chính xác của mô hình (R²): {r_squared:.2f}"
    
    return analysis_result, insight

# ------------------------
# Hàm đo lường tác động của giá đến doanh thu
# ------------------------
def analyze_price_revenue_impact(data, product=None):
    """Đo lường tác động của giá đến doanh thu."""
    if product is not None and 'Product' in data.columns:
        product_data = data[data['Product'] == product].copy()
    else:
        product_data = data.copy()  # Nếu không có cột Product hoặc không có sản phẩm cụ thể, sử dụng tất cả dữ liệu
    
    if len(product_data) < 3:
        return None, "Không đủ dữ liệu để phân tích."

    # Phân tích trên dữ liệu đã tổng hợp theo giá
    price_grouped = product_data.groupby('Price').agg({
        'QUANTITY': 'mean'
    }).reset_index()
    
    if len(price_grouped) < 3:
        return None, "Không đủ mức giá khác nhau để phân tích. Cần ít nhất 3 mức giá."

    model, r_squared, _, optimal_price = build_pricing_model(price_grouped)
    
    if not model:
        return None, "Không thể xây dựng mô hình để phân tích."

    # Tạo các mức giá cho phân tích
    price_min = price_grouped['Price'].min()
    price_max = price_grouped['Price'].max()
    price_range = price_max - price_min
    
    # Tạo 100 mức giá phân bố đều trong khoảng giá mở rộng thêm 20%
    extended_min = max(0, price_min - 0.1 * price_range)
    extended_max = price_max + 0.1 * price_range
    prices = np.linspace(extended_min, extended_max, 100)
    
    # Tính doanh thu dự kiến ở mỗi mức giá
    revenues = []
    for price in prices:
        quantity = max(0, model.predict(np.array([[price]]))[0])  # Đảm bảo số lượng không âm
        revenue = price * quantity
        revenues.append((price, revenue, quantity))
    
    result = {
        'price_revenue_data': revenues,
        'optimal_price': optimal_price,
        'r_squared': r_squared,
        'original_data': price_grouped
    }
    
    message = f"Giá tối ưu dự đoán: {optimal_price:.2f}"
    if r_squared is not None:
        message += f" (Độ chính xác R²: {r_squared:.2f})"
    
    return result, message

# Hàm mới: Phân tích xu hướng theo thời gian
def analyze_time_trend(data, product=None, freq='M'):
    """Phân tích xu hướng giá và số lượng theo thời gian"""
    if product is not None and 'Product' in data.columns:
        product_data = data[data['Product'] == product].copy()
    else:
        product_data = data.copy()  # Nếu không có cột Product hoặc không có sản phẩm cụ thể, sử dụng tất cả dữ liệu
    
    if len(product_data) < 10:  # Cần đủ dữ liệu cho phân tích
        return None, "Không đủ dữ liệu để phân tích xu hướng theo thời gian"
    
    # Tổng hợp dữ liệu theo khoảng thời gian
    product_data.set_index('Date', inplace=True)
    time_series = product_data.resample(freq).agg({
        'QUANTITY': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    # Thêm cột doanh thu
    time_series['Revenue'] = time_series['QUANTITY'] * time_series['Price']
    
    # Phân tích xu hướng thời gian nếu có đủ dữ liệu
    trend_analysis = {}
    if len(time_series) >= 6:  # Cần ít nhất 6 điểm dữ liệu cho phân tích xu hướng
        try:
            # Phân tích thành phần xu hướng cho số lượng
            if not time_series['QUANTITY'].isna().any():
                decomposition = seasonal_decompose(
                    time_series['QUANTITY'], 
                    model='additive', 
                    period=min(4, len(time_series)//2)
                )
                trend_analysis['quantity_trend'] = decomposition.trend
                
            # Phân tích thành phần xu hướng cho giá
            if not time_series['Price'].isna().any():
                decomposition = seasonal_decompose(
                    time_series['Price'], 
                    model='additive', 
                    period=min(4, len(time_series)//2)
                )
                trend_analysis['price_trend'] = decomposition.trend
        except Exception as e:
            st.warning(f"Không thể phân tích thành phần xu hướng: {e}")
    
    result = {
        'time_series': time_series,
        'trend_analysis': trend_analysis
    }
    
    return result, "Xem biểu đồ để hiểu xu hướng theo thời gian"


# Hàm vẽ biểu đồ độ nhạy cảm giá
def plot_price_sensitivity(result):
    """Vẽ biểu đồ phân tích độ nhạy cảm giá."""
    if not result:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Biểu đồ độ co giãn giá theo sản phẩm
    elasticity_df = result['elasticity_df']
    sns.barplot(data=elasticity_df, x='Product', y='Average_Elasticity', ax=ax1)
    ax1.set_title('Độ co giãn giá trung bình theo sản phẩm')
    ax1.set_xlabel('Sản phẩm')
    ax1.set_ylabel('Độ co giãn giá')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Biểu đồ tầm quan trọng của các yếu tố
    feature_importance = result['feature_importance']
    sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax2)
    ax2.set_title('Tầm quan trọng của các yếu tố ảnh hưởng đến độ nhạy cảm giá')
    ax2.set_xlabel('Mức độ ảnh hưởng')
    ax2.set_ylabel('Yếu tố')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_competitor_price_impact(data, product):
    """Phân tích và vẽ biểu đồ tác động của giá đối thủ đến số lượng bán."""
    product_data = data[data['Product'] == product]
    if 'Competitor_Price' not in product_data.columns:
        return None, "Không có dữ liệu giá đối thủ để phân tích.", None

    X = product_data[['Competitor_Price']].values
    y = product_data['QUANTITY'].values

    if len(X) < 2:
        return None, "Không đủ dữ liệu để phân tích.", None

    # Huấn luyện mô hình hồi quy
    model = LinearRegression()
    model.fit(X, y)

    coef = model.coef_[0]
    intercept = model.intercept_

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='blue', alpha=0.6, label="Dữ liệu thực tế")
    ax.plot(X, model.predict(X), color='red', linewidth=2, label="Đường hồi quy")
    ax.set_xlabel("Competitor Price")
    ax.set_ylabel("Quantity Sold")
    ax.set_title(f"Ảnh hưởng của giá đối thủ đến lượng bán - {product}")
    ax.legend()
    ax.grid(True)

    result = {
        'coef': coef,
        'intercept': intercept
    }

    return result, f"Khi giá đối thủ tăng 1 đơn vị, số lượng bán thay đổi {coef:.2f} đơn vị.", fig
# Hàm vẽ biểu đồ dự báo giá
def plot_seasonal_forecast(result):
    """Vẽ biểu đồ dự báo giá theo mùa vụ."""
    if not result:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Biểu đồ giá lịch sử và dự báo
    historical_data = result['historical_data']
    forecast_data = result['forecast']
    
    ax1.plot(historical_data.index, historical_data['Price'], label='Giá lịch sử', color='blue')
    ax1.plot(forecast_data['Date'], forecast_data['Forecast_Price'], label='Dự báo', color='red', linestyle='--')
    ax1.set_title('Dự báo giá theo thời gian')
    ax1.set_xlabel('Ngày')
    ax1.set_ylabel('Giá')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Biểu đồ thành phần mùa vụ
    seasonal_data = result['seasonal_component']
    ax2.plot(seasonal_data.index, seasonal_data, color='green')
    ax2.set_title('Thành phần mùa vụ của giá')
    ax2.set_xlabel('Ngày')
    ax2.set_ylabel('Mức mùa vụ')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
# Function to create main app UI
def create_pricing_analysis_app():
    st.title("Phân Tích Giá và Tối Ưu Hóa Doanh Thu")
    
    # Upload files
    st.sidebar.header("Tải Dữ Liệu")
    sell_meta = st.sidebar.file_uploader("Tải file dữ liệu sản phẩm", type=["csv"])
    transactions = st.sidebar.file_uploader("Tải file dữ liệu giao dịch", type=["csv"])
    date_info = st.sidebar.file_uploader("Tải file dữ liệu ngày", type=["csv"])
    
    if sell_meta and transactions and date_info:
        data = load_data(sell_meta, transactions, date_info)
        
        if not data.empty:
            st.sidebar.success("Đã tải dữ liệu thành công!")
            data['Competitor_Price'] = data['Price'] * np.random.uniform(0.9, 1.1, size=len(data))

            # Show data summary
            with st.expander("Tổng quan dữ liệu"):
                st.write(f"Tổng số dòng dữ liệu: {len(data)}")
                st.write(f"Phạm vi thời gian: {data['Date'].min().date()} đến {data['Date'].max().date()}")
                st.write(f"Số lượng sản phẩm: {data['Product'].nunique()}")
                st.dataframe(data)
            
            # Analysis options
            st.sidebar.header("Phân Tích")
            analysis_type = st.sidebar.selectbox(
                "Chọn loại phân tích",
                ["Phân tích sản phẩm đơn lẻ", "Phân tích combo sản phẩm"]
            )
            
            if analysis_type == "Phân tích sản phẩm đơn lẻ":
                product_list = sorted(data['Product'].unique().tolist())
                selected_product = st.sidebar.selectbox("Chọn sản phẩm", product_list)
                
                if selected_product:
                    st.header(f"Phân tích cho sản phẩm: {selected_product}")
                    
                    tabs = st.tabs(["Tối ưu giá", "Xu hướng theo thời gian", "Dữ liệu chi tiết", "Độ nhạy cảm giá", "Dự báo mùa vụ","Phân tích đối thủ"])
                    
                    with tabs[0]:
                        price_demand_result, insight = analyze_price_demand_relationship(data, selected_product)
                        revenue_impact_result, revenue_message = analyze_price_revenue_impact(data, selected_product)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Mối quan hệ Giá - Nhu cầu")
                            st.write(insight)
                            if price_demand_result:
                                # Create plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_data = price_demand_result['data']
                                ax.scatter(plot_data['Price'], plot_data['QUANTITY'], s=50, alpha=0.7)
                                
                                # Plot predicted curve
                                x_range = np.linspace(plot_data['Price'].min(), plot_data['Price'].max(), 100)
                                y_pred = price_demand_result['model'].predict(x_range.reshape(-1, 1))
                                ax.plot(x_range, y_pred, 'r-', linewidth=2)
                                
                                ax.set_xlabel('Giá')
                                ax.set_ylabel('Số lượng bán')
                                ax.set_title('Mối quan hệ giữa Giá và Nhu cầu')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            
                        with col2:
                            st.subheader("Tác động của Giá đến Doanh thu")
                            st.write(revenue_message)
                            if revenue_impact_result:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Extract data
                                prices = [item[0] for item in revenue_impact_result['price_revenue_data']]
                                revenues = [item[1] for item in revenue_impact_result['price_revenue_data']]
                                quantities = [item[2] for item in revenue_impact_result['price_revenue_data']]
                                
                                # Plot revenue curve
                                ax.plot(prices, revenues, 'b-', linewidth=2)
                                
                                # Mark optimal price
                                opt_price = revenue_impact_result['optimal_price']
                                opt_index = np.argmin(np.abs(np.array(prices) - opt_price))
                                opt_revenue = revenues[opt_index]
                                
                                ax.plot(opt_price, opt_revenue, 'ro', markersize=10)
                                ax.annotate(f'Giá tối ưu: {opt_price:.2f}', 
                                            xy=(opt_price, opt_revenue),
                                            xytext=(5, 10), textcoords='offset points')
                                
                                # Plot original data points
                                original_data = revenue_impact_result['original_data']
                                original_revenue = original_data['Price'] * original_data['QUANTITY']
                                ax.scatter(original_data['Price'], original_revenue, color='green', alpha=0.7)
                                
                                ax.set_xlabel('Giá')
                                ax.set_ylabel('Doanh thu dự kiến')
                                ax.set_title('Mối quan hệ giữa Giá và Doanh thu')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                    
                    with tabs[1]:
                        time_result, time_message = analyze_time_trend(data, selected_product)
                        if time_result:
                            time_series = time_result['time_series']
                            
                            # Biểu đồ xu hướng theo thời gian
                            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                            
                            # Biểu đồ số lượng
                            ax1.plot(time_series['Date'], time_series['QUANTITY'], 'b-o')
                            ax1.set_ylabel('Số lượng')
                            ax1.set_title('Xu hướng số lượng bán theo thời gian')
                            ax1.grid(True, alpha=0.3)
                            
                            # Biểu đồ giá
                            ax2.plot(time_series['Date'], time_series['Price'], 'g-o')
                            ax2.set_ylabel('Giá trung bình')
                            ax2.set_title('Xu hướng giá theo thời gian')
                            ax2.grid(True, alpha=0.3)
                            
                            # Biểu đồ doanh thu
                            ax3.plot(time_series['Date'], time_series['Revenue'], 'r-o')
                            ax3.set_ylabel('Doanh thu')
                            ax3.set_title('Xu hướng doanh thu theo thời gian')
                            ax3.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Hiển thị bảng dữ liệu
                            st.subheader("Dữ liệu theo thời gian")
                            st.dataframe(time_series.set_index('Date'))
                        else:
                            st.write(time_message)
                    
                    with tabs[2]:
                        product_data = data[data['Product'] == selected_product].copy()
                        product_data['Revenue'] = product_data['Price'] * product_data['QUANTITY']
                        
                        st.subheader("Dữ liệu chi tiết")
                        st.dataframe(product_data)
                        
                        # Thống kê
                        st.subheader("Thống kê")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tổng số lượng bán", f"{product_data['QUANTITY'].sum():,.0f}")
                        col2.metric("Giá trung bình", f"{product_data['Price'].mean():,.2f}")
                        col3.metric("Tổng doanh thu", f"{product_data['Revenue'].sum():,.2f}")
                        
                    with tabs[3]:
                            st.subheader("Phân tích độ nhạy cảm giá")
                            sensitivity_result, sensitivity_message = analyze_price_sensitivity_factors(data)
                            st.write(sensitivity_message)
                            
                            if sensitivity_result:
                                # Vẽ biểu đồ
                                fig = plot_price_sensitivity(sensitivity_result)
                                if fig:
                                    st.pyplot(fig)
                                
                                # Hiển thị đề xuất
                                st.subheader("Đề xuất từ phân tích")
                                for insight in sensitivity_result['insights']:
                                    st.write(f"- {insight}")
                                
                                # Hiển thị bảng độ co giãn giá
                                st.subheader("Độ co giãn giá theo sản phẩm")
                                st.dataframe(sensitivity_result['elasticity_df'])
                            else:
                                st.warning(sensitivity_message)
                    with tabs[4]:
                        st.subheader("Dự báo xu hướng giá theo mùa vụ")
                        seasonal_result, seasonal_insights = forecast_seasonal_price(data, selected_product)
                        
                        if seasonal_result:
                            # Vẽ biểu đồ
                            fig = plot_seasonal_forecast(seasonal_result)
                            if fig:
                                st.pyplot(fig)
                            
                            # Hiển thị đề xuất
                            st.subheader("Thông tin chi tiết")
                            for insight in seasonal_insights:
                                st.write(f"- {insight}")
                            
                            # Hiển thị bảng dự báo
                            st.subheader("Dữ liệu dự báo")
                            st.dataframe(seasonal_result['forecast'])
                        else:
                            st.warning(seasonal_insights)
                    with tabs[5]:
                        st.subheader("Phân tích Giá Đối thủ")
                        competitor_result, competitor_message, competitor_fig = analyze_competitor_price_impact(data, selected_product)
                        st.write(competitor_message)
                        if competitor_fig:
                            st.pyplot(competitor_fig)


            elif analysis_type == "Phân tích combo sản phẩm":
                product_list = sorted(data['Product'].unique().tolist())
                selected_combo = st.sidebar.multiselect("Chọn sản phẩm cho combo", product_list)
                
                if selected_combo and len(selected_combo) >= 2:
                    st.header(f"Phân tích combo: {', '.join(selected_combo)}")
                    
                    combo_data = create_combo(data, selected_combo)
                    if not combo_data.empty:
                        # Đổi tên cột cho phù hợp với hàm phân tích
                        combo_analysis_data = combo_data.rename(columns={
                            'Combo_Price': 'Price', 
                            'Combo_Quantity': 'QUANTITY',
                            'Combo_Revenue': 'Revenue'
                        })
                        
                        tabs = st.tabs(["Tối ưu giá", "Xu hướng theo thời gian", "Dữ liệu chi tiết", "Độ nhạy cảm giá", "Dự báo mùa vụ"])
                        
                        with tabs[0]:
                            # Bây giờ chúng ta không cần tham số product vì đã làm việc trực tiếp với dữ liệu combo
                            combo_price_result, combo_message = analyze_price_revenue_impact(combo_analysis_data)
                            
                            st.subheader("Tối ưu giá combo")
                            st.write(combo_message)
                            
                            if combo_price_result:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Extract data
                                prices = [item[0] for item in combo_price_result['price_revenue_data']]
                                revenues = [item[1] for item in combo_price_result['price_revenue_data']]
                                
                                # Plot revenue curve
                                ax.plot(prices, revenues, 'b-', linewidth=2)
                                
                                # Mark optimal price
                                opt_price = combo_price_result['optimal_price']
                                opt_index = np.argmin(np.abs(np.array(prices) - opt_price))
                                opt_revenue = revenues[opt_index]
                                
                                ax.plot(opt_price, opt_revenue, 'ro', markersize=10)
                                ax.annotate(f'Giá combo tối ưu: {opt_price:.2f}', 
                                            xy=(opt_price, opt_revenue),
                                            xytext=(5, 10), textcoords='offset points')
                                
                                ax.set_xlabel('Giá combo')
                                ax.set_ylabel('Doanh thu dự kiến')
                                ax.set_title('Mối quan hệ giữa Giá combo và Doanh thu')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                        
                        with tabs[1]:
                            # Phân tích xu hướng thời gian cho combo không cần tham số product
                            combo_time_result, combo_time_message = analyze_time_trend(combo_analysis_data)
                            
                            if combo_time_result:
                                time_series = combo_time_result['time_series']
                                
                                # Biểu đồ xu hướng theo thời gian cho combo
                                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                                
                                # Biểu đồ số lượng
                                ax1.plot(time_series['Date'], time_series['QUANTITY'], 'b-o')
                                ax1.set_ylabel('Số lượng combo')
                                ax1.set_title('Xu hướng số lượng combo theo thời gian')
                                ax1.grid(True, alpha=0.3)
                                
                                # Biểu đồ giá
                                ax2.plot(time_series['Date'], time_series['Price'], 'g-o')
                                ax2.set_ylabel('Giá combo')
                                ax2.set_title('Xu hướng giá combo theo thời gian')
                                ax2.grid(True, alpha=0.3)
                                
                                # Biểu đồ doanh thu
                                ax3.plot(time_series['Date'], time_series['Revenue'], 'r-o')
                                ax3.set_ylabel('Doanh thu combo')
                                ax3.set_title('Xu hướng doanh thu combo theo thời gian')
                                ax3.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Hiển thị bảng dữ liệu
                                st.subheader("Dữ liệu theo thời gian")
                                st.dataframe(time_series.set_index('Date'))
                            else:
                                st.write(combo_time_message)
                        
                        with tabs[2]:
                            st.subheader("Dữ liệu chi tiết combo")
                            st.dataframe(combo_data)
                            
                            # Thống kê
                            st.subheader("Thống kê combo")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Tổng số lượng combo", f"{combo_data['Combo_Quantity'].sum():,.0f}")
                            col2.metric("Giá combo trung bình", f"{combo_data['Combo_Price'].mean():,.2f}")
                            col3.metric("Tổng doanh thu combo", f"{combo_data['Combo_Revenue'].mean():,.2f}")
                            col3.metric("Tổng doanh thu combo", f"{combo_data['Combo_Revenue'].sum():,.2f}")
                        # Thêm tab độ nhạy cảm giá
                        with tabs[3]:
                            st.subheader("Phân tích độ nhạy cảm giá")
                            sensitivity_result, sensitivity_message = analyze_price_sensitivity_factors1(
                                combo_analysis_data,
                                is_combo=True
                            )
                            st.write(sensitivity_message)
                            
                            if sensitivity_result:
                                # Vẽ biểu đồ
                                fig = plot_price_sensitivity(sensitivity_result)
                                if fig:
                                    st.pyplot(fig)
                                
                                # Hiển thị đề xuất
                                st.subheader("Đề xuất từ phân tích")
                                for insight in sensitivity_result['insights']:
                                    st.write(f"- {insight}")
                                
                                # Hiển thị bảng độ co giãn giá
                                st.subheader("Độ co giãn giá theo sản phẩm")
                                st.dataframe(sensitivity_result['elasticity_df'])
                            else:
                                st.warning(sensitivity_message)
                        # Tab Dự báo mùa vụ
                        with tabs[4]:
                            st.subheader("Dự báo xu hướng giá combo theo mùa vụ")
                            seasonal_result, seasonal_insights = forecast_seasonal_price1(
                                combo_analysis_data,
                                is_combo=True
                            )
                            if seasonal_result:
                                fig = plot_seasonal_forecast(seasonal_result)
                                if fig:
                                    st.pyplot(fig)
                                st.subheader("Thông tin chi tiết")
                                for insight in seasonal_insights:
                                    st.write(f"- {insight}")
                                st.subheader("Dữ liệu dự báo")
                                st.dataframe(seasonal_result['forecast'])
                            else:
                                st.warning(seasonal_insights)
                    else:
                        st.warning("Không thể tạo dữ liệu combo. Vui lòng kiểm tra lại các sản phẩm đã chọn.")
        else:
            st.error("Có lỗi khi xử lý dữ liệu. Vui lòng kiểm tra định dạng của các file đầu vào.")
    else:
        st.info("Vui lòng tải lên đầy đủ ba file dữ liệu để bắt đầu phân tích.")
        
        # Hiển thị hướng dẫn
        st.header("Hướng dẫn sử dụng")
        
        st.subheader("Chuẩn bị dữ liệu")
        st.write("""
        Ứng dụng yêu cầu ba file CSV đầu vào:
        1. **File dữ liệu sản phẩm (sell_meta)**: Chứa thông tin về sản phẩm với các cột SELL_ID, SELL_CATEGORY và Product.
        2. **File dữ liệu giao dịch (transactions)**: Chứa thông tin giao dịch với các cột Date/CALENDAR_DATE/Transaction_Date, SELL_ID, SELL_CATEGORY, QUANTITY và Price.
        3. **File dữ liệu ngày (date_info)**: Chứa thông tin ngày với cột Date/CALENDAR_DATE.
        """)
        
        st.subheader("Các phân tích")
        st.write("""
        Ứng dụng cung cấp hai loại phân tích:
        - **Phân tích sản phẩm đơn lẻ**: Phân tích giá tối ưu và xu hướng theo thời gian cho một sản phẩm.
        - **Phân tích combo sản phẩm**: Tạo combo từ nhiều sản phẩm và phân tích giá tối ưu cho combo.
        """)
        
        # Tạo dữ liệu mẫu
        st.subheader("Dữ liệu mẫu")
        st.write("Nếu bạn chưa có dữ liệu, bạn có thể tải xuống các file mẫu:")
        
        # Tạo dữ liệu mẫu để download
        if st.button("Tạo dữ liệu mẫu"):
            # Tạo dữ liệu mẫu cho sell_meta
            sell_meta_sample = pd.DataFrame({
                'SELL_ID': list(range(1, 6)),
                'SELL_CATEGORY': ['Food', 'Food', 'Beverage', 'Beverage', 'Dessert'],
                'Product': ['Burger', 'Pizza', 'Soda', 'Coffee', 'Ice Cream']
            })
            
            # Tạo dữ liệu mẫu cho date_info
            date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            date_info_sample = pd.DataFrame({
                'CALENDAR_DATE': date_range.strftime('%Y-%m-%d')
            })
            
            # Tạo dữ liệu mẫu cho transactions
            np.random.seed(42)
            dates = date_range[:90].strftime('%Y-%m-%d')  # Chỉ lấy 90 ngày đầu tiên
            transactions_sample = []
            
            for day in dates:
                for product_id in range(1, 6):
                    # Tạo biến động giá theo thời gian
                    price_base = [8.99, 12.99, 2.49, 3.99, 4.49][product_id-1]
                    price_variation = np.random.uniform(-0.5, 0.5)
                    price = price_base + price_variation
                    
                    # Tạo số lượng bán với phụ thuộc ngược vào giá
                    quantity_base = np.random.randint(10, 30)
                    price_effect = -2 * (price - price_base)  # Giá càng cao, số lượng càng giảm
                    quantity = max(1, int(quantity_base + price_effect + np.random.randint(-5, 5)))
                    
                    category = ['Food', 'Food', 'Beverage', 'Beverage', 'Dessert'][product_id-1]
                    
                    transactions_sample.append({
                        'Transaction_Date': day,
                        'SELL_ID': product_id,
                        'SELL_CATEGORY': category,
                        'QUANTITY': quantity,
                        'PRICE': round(price, 2)
                    })
            
            transactions_df = pd.DataFrame(transactions_sample)
            
            # Tạo CSV để download
            st.download_button(
                label="Tải file sell_meta.csv",
                data=sell_meta_sample.to_csv(index=False),
                file_name="sell_meta_sample.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="Tải file date_info.csv",
                data=date_info_sample.to_csv(index=False),
                file_name="date_info_sample.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="Tải file transactions.csv",
                data=transactions_df.to_csv(index=False),
                file_name="transactions_sample.csv",
                mime="text/csv"
            )

# Main function
def main():
    create_pricing_analysis_app()

if __name__ == "__main__":
    main()