import numpy as np
import pandas as pd
# Hàm tính độ co giãn giá
def calculate_price_elasticity(data):
    """Tính độ co giãn giá cho từng sản phẩm."""
    elasticity_results = []
    for product in data['Product'].unique():
        product_data = data[data['Product'] == product].copy()
        if len(product_data) < 5:  # Cần đủ dữ liệu
            continue
        
        # Tính phần trăm thay đổi
        product_data = product_data.sort_values('Date')
        product_data['Price_Change'] = product_data['Price'].pct_change() * 100
        product_data['Quantity_Change'] = product_data['QUANTITY'].pct_change() * 100
        
        # Loại bỏ các giá trị vô cực hoặc NaN
        product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price_Change', 'Quantity_Change'])
        
        # Tính độ co giãn giá
        product_data['Elasticity'] = product_data['Quantity_Change'] / product_data['Price_Change']
        product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Elasticity'])
        
        if not product_data.empty:
            avg_elasticity = product_data['Elasticity'].mean()
            elasticity_results.append({
                'Product': product,
                'Average_Elasticity': avg_elasticity,
                'Category': product_data['SELL_CATEGORY'].iloc[0],
                'Data_Points': len(product_data)
            })
    
    return pd.DataFrame(elasticity_results)