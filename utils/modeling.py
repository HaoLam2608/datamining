from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def prepare_features(df):
    # Tạo biến ngày trong tuần từ CALENDAR_DATE
    if 'CALENDAR_DATE' in df.columns:
        df['DAY_OF_WEEK'] = df['CALENDAR_DATE'].dt.dayofweek
    else:
        df['DAY_OF_WEEK'] = 0  # Nếu không có ngày thì mặc định 0

    # Lấy các biến làm đầu vào
    X = df[['PRICE', 'DAY_OF_WEEK']]
    y = df['QUANTITY']
    return X, y

def train_random_forest_model(df):
    X, y = prepare_features(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_polynomial_model(df):
    """Huấn luyện mô hình đa thức bậc 2 để dự đoán số lượng dựa trên giá"""
    grp = df.groupby('PRICE')['QUANTITY'].sum().reset_index()
    X = grp[['PRICE']].values
    y = grp['QUANTITY'].values
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly_features, grp

def predict_revenue(model, poly_features, price):
    """Dự đoán doanh thu dựa trên giá"""
    price_poly = poly_features.transform(np.array([[price]]))
    qty = max(0, model.predict(price_poly)[0])
    return price * qty