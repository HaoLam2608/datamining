import pandas as pd
import numpy as np

def load_and_merge_data(u_meta, u_trans, u_date):
    """Tải và merge dữ liệu từ các file CSV"""
    sell_meta = pd.read_csv(u_meta)
    transaction = pd.read_csv(u_trans)
    date_info = pd.read_csv(u_date)

    # Chuẩn hóa tên cột
    sell_meta.columns = sell_meta.columns.str.strip()
    transaction.columns = transaction.columns.str.strip()
    date_info.columns = date_info.columns.str.strip()

    # Convert CALENDAR_DATE sang datetime
    transaction['CALENDAR_DATE'] = pd.to_datetime(transaction['CALENDAR_DATE'], errors='coerce')
    date_info['CALENDAR_DATE'] = pd.to_datetime(date_info['CALENDAR_DATE'], errors='coerce')

    # Merge dữ liệu
    merged = pd.merge(transaction, sell_meta, on=["SELL_ID", "SELL_CATEGORY"], how="left")
    merged = pd.merge(merged, date_info, on="CALENDAR_DATE", how="left")

    return merged, sell_meta, transaction, date_info

def clean_data(df):
    """Loại bỏ outliers và làm sạch dữ liệu"""
    df = df.dropna(subset=['PRICE', 'QUANTITY'])
    
    # Loại bỏ outliers bằng IQR
    Q1_price = df['PRICE'].quantile(0.25)
    Q3_price = df['PRICE'].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    lower_bound_price = Q1_price - 1.5 * IQR_price
    upper_bound_price = Q3_price + 1.5 * IQR_price

    Q1_qty = df['QUANTITY'].quantile(0.25)
    Q3_qty = df['QUANTITY'].quantile(0.75)
    IQR_qty = Q3_qty - Q1_qty
    lower_bound_qty = Q1_qty - 1.5 * IQR_qty
    upper_bound_qty = Q3_qty + 1.5 * IQR_qty

    df_clean = df[
        (df['PRICE'] >= lower_bound_price) & 
        (df['PRICE'] <= upper_bound_price) &
        (df['QUANTITY'] >= lower_bound_qty) & 
        (df['QUANTITY'] <= upper_bound_qty)
    ].copy()
    
    return df_clean