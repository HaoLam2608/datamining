import pandas as pd
import numpy as np

def calculate_price_elasticity1(data, is_combo=False):
    """Tính độ co giãn giá cho sản phẩm hoặc combo."""
    elasticity_results = []
    
    if is_combo:
        # Xử lý combo (chỉ có một "sản phẩm" là combo)
        if len(data) < 5:
            return pd.DataFrame()
        
        price_col = 'Price'  # Sử dụng 'Price' vì combo_analysis_data đã đổi tên
        quantity_col = 'QUANTITY'
        
        # Tính phần trăm thay đổi
        product_data = data.sort_values('Date').copy()
        product_data['Price_Change'] = product_data[price_col].pct_change() * 100
        product_data['Quantity_Change'] = product_data[quantity_col].pct_change() * 100
        
        # Loại bỏ các giá trị vô cực hoặc NaN
        product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price_Change', 'Quantity_Change'])
        
        # Tính độ co giãn giá
        product_data['Elasticity'] = product_data['Quantity_Change'] / product_data['Price_Change']
        product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Elasticity'])
        
        if not product_data.empty:
            avg_elasticity = product_data['Elasticity'].mean()
            elasticity_results.append({
                'Product': data['Product'].iloc[0] if 'Product' in data.columns else 'Combo',
                'Average_Elasticity': avg_elasticity,
                'Category': 'Combo',  # Luôn gán 'Combo' cho combo
                'Data_Points': len(product_data)
            })
    else:
        # Xử lý sản phẩm đơn lẻ
        for product in data['Product'].unique():
            product_data = data[data['Product'] == product].copy()
            if len(product_data) < 5:
                continue
            
            price_col = 'Price'
            quantity_col = 'QUANTITY'
            
            # Tính phần trăm thay đổi
            product_data = product_data.sort_values('Date')
            product_data['Price_Change'] = product_data[price_col].pct_change() * 100
            product_data['Quantity_Change'] = product_data[quantity_col].pct_change() * 100
            
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
                    'Category': product_data['SELL_CATEGORY'].iloc[0] if 'SELL_CATEGORY' in product_data.columns else 'Unknown',
                    'Data_Points': len(product_data)
                })
    
    return pd.DataFrame(elasticity_results)