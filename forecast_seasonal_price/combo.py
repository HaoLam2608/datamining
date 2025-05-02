from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
# Điều chỉnh hàm forecast_seasonal_price cho combo
def forecast_seasonal_price1(data, product=None, is_combo=False, periods=12):
    """Dự đoán xu hướng giá theo mùa vụ cho sản phẩm hoặc combo."""
    # Với combo, không cần lọc theo product vì dữ liệu đã là một combo duy nhất
    if is_combo:
        product_data = data.copy()
    else:
        if product is not None and 'Product' in data.columns:
            product_data = data[data['Product'] == product].copy()
        else:
            product_data = data.copy()
    
    if len(product_data) < 12:
        return None, "Không đủ dữ liệu để dự đoán xu hướng giá theo mùa vụ."
    
    # Tổng hợp dữ liệu theo tháng
    product_data.set_index('Date', inplace=True)
    price_col = 'Price'  # Luôn sử dụng cột 'Price' vì combo_analysis_data đã đổi tên
    monthly_data = product_data.resample('M').agg({price_col: 'mean'}).dropna()
    
    if len(monthly_data) < 12:
        return None, "Không đủ dữ liệu hàng tháng để phân tích mùa vụ."
    
    try:
        # Phân tích mùa vụ
        decomposition = seasonal_decompose(monthly_data[price_col], model='additive', period=12)
        seasonal = decomposition.seasonal
        
        # Phù hợp mô hình SARIMA
        model = SARIMAX(monthly_data[price_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        
        # Dự báo
        forecast = model_fit.forecast(steps=periods)
        forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='M')
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast_Price': forecast})
        
        result = {
            'historical_data': monthly_data,
            'seasonal_component': seasonal,
            'forecast': forecast_df,
            'model_summary': model_fit.summary()
        }
        
        insights = [
            f"Dự báo giá trung bình trong {periods} tháng tới: {forecast.mean():.2f}",
            "Xem biểu đồ để hiểu xu hướng mùa vụ và dự báo giá."
        ]
        
        return result, insights
    except Exception as e:
        return None, f"Lỗi khi phân tích mùa vụ: {e}"