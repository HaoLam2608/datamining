from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from calculate_price_elasticity.calculate_price_elasticity import calculate_price_elasticity

# Hàm phân tích yếu tố ảnh hưởng đến độ nhạy cảm giá
def analyze_price_sensitivity_factors(data):
    """Phân tích các yếu tố ảnh hưởng đến độ nhạy cảm giá."""
    if data.empty:
        return None, "Không có dữ liệu để phân tích độ nhạy cảm giá."
    
    # Tính độ co giãn giá
    elasticity_df = calculate_price_elasticity(data)
    if elasticity_df.empty:
        return None, "Không đủ dữ liệu để tính độ co giãn giá."
    
    # Chuẩn bị dữ liệu cho mô hình hồi quy
    features = []
    for product in elasticity_df['Product']:
        product_data = data[data['Product'] == product].copy()
        if product_data.empty:
            continue
        
        # Thêm các đặc trưng
        product_data['Day_of_Week'] = product_data['Date'].dt.dayofweek
        product_data['Month'] = product_data['Date'].dt.month
        product_data['Price_Level'] = product_data['Price'].mean()  # Mức giá trung bình
        
        avg_data = product_data.groupby('Product').agg({
            'Day_of_Week': 'mean',
            'Month': 'mean',
            'Price_Level': 'mean',
            'SELL_CATEGORY': 'first'
        }).reset_index()
        
        avg_data['Elasticity'] = elasticity_df[elasticity_df['Product'] == product]['Average_Elasticity'].iloc[0]
        features.append(avg_data)
    
    if not features:
        return None, "Không thể tạo đặc trưng để phân tích."
    
    feature_df = pd.concat(features)
    
    # Mã hóa biến phân loại
    feature_df = pd.get_dummies(feature_df, columns=['SELL_CATEGORY'])
    
    # Xây dựng mô hình Random Forest để xác định tầm quan trọng của các yếu tố
    X = feature_df.drop(columns=['Product', 'Elasticity'])
    y = feature_df['Elasticity']
    
    if len(X) < 2:
        return None, "Không đủ dữ liệu để xây dựng mô hình."
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Lấy tầm quan trọng của các đặc trưng
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    insights = []
    for _, row in feature_importance.iterrows():
        insights.append(f"Yếu tố {row['Feature']} có mức độ ảnh hưởng {row['Importance']:.2%} đến độ nhạy cảm giá.")
    
    result = {
        'elasticity_df': elasticity_df,
        'feature_importance': feature_importance,
        'insights': insights
    }
    
    return result, "Xem biểu đồ và đề xuất để hiểu các yếu tố ảnh hưởng đến độ nhạy cảm giá."
