import streamlit as st
import pandas as pd
import altair as alt
from utils.data_processing import clean_data

def render_price_qty_tab(df_prod, combo_label):
    """Hiển thị nội dung tab So sánh giá & SL"""
    st.header("📊 Nhạy Cảm Giá – Dựa trên DateInfo")
    
    df = clean_data(df_prod)
    
    # Chuẩn hóa biến ngày tháng
    date_cols = []
    if 'HOLIDAY' in df.columns:
        df['IS_HOLIDAY'] = df['HOLIDAY'].notna().astype(int)
        date_cols.append('IS_HOLIDAY')
    
    if 'IS_WEEKEND' in df.columns:
        date_cols.append('IS_WEEKEND')
    elif 'CALENDAR_DATE' in df.columns:
        df['IS_WEEKEND'] = df['CALENDAR_DATE'].dt.weekday >= 5
        date_cols.append('IS_WEEKEND')
    
    if 'IS_SCHOOLBREAK' in df.columns:
        date_cols.append('IS_SCHOOLBREAK')
    
    if 'IS_OUTDOOR' in df.columns:
        date_cols.append('IS_OUTDOOR')
    
    if not date_cols:
        st.warning("Không tìm thấy cột ngày tháng phù hợp trong dữ liệu.")
        return
    
    # Tính elasticity
    df = df.sort_values('CALENDAR_DATE')
    df['ΔP'] = df['PRICE'].pct_change()
    df['ΔQ'] = df['QUANTITY'].pct_change()
    df = df.dropna(subset=['ΔP', 'ΔQ'])
    df = df[df['ΔP'] != 0]
    df['Elasticity'] = df['ΔQ'] / df['ΔP']
    
    # Loại bỏ outliers
    Q1_el = df['Elasticity'].quantile(0.25)
    Q3_el = df['Elasticity'].quantile(0.75)
    IQR_el = Q3_el - Q1_el
    lower_bound_el = Q1_el - 3 * IQR_el
    upper_bound_el = Q3_el + 3 * IQR_el
    df = df[(df['Elasticity'] >= lower_bound_el) & (df['Elasticity'] <= upper_bound_el)]
    
    # Phân tích elasticity theo yếu tố ngày tháng
    records = []
    for factor in date_cols:
        try:
            grp = df.groupby(factor)['Elasticity'].mean().reset_index()
            grp.columns = [factor, 'Elasticity_TB']
            for _, row in grp.iterrows():
                factor_value = "Có" if row[factor] == 1 else "Không"
                records.append({
                    'Yếu tố': factor.replace('IS_', ''),
                    'Giá trị': factor_value,
                    'Elasticity trung bình': round(row['Elasticity_TB'], 2)
                })
        except Exception as e:
            st.warning(f"Không thể tính elasticity cho yếu tố {factor}: {e}")
    
    if records:
        df_el = pd.DataFrame(records)
        st.subheader("Elasticity trung bình theo yếu tố")
        st.dataframe(df_el)
        
        for factor_data in df_el['Yếu tố'].unique():
            df_plot = df_el[df_el['Yếu tố'] == factor]
            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('Giá trị:O', title=factor),
                y=alt.Y('Elasticity trung bình:Q', title='Elasticity trung bình'),
                color=alt.Color('Giá trị:N', legend=None),
                tooltip=['Yếu tố', 'Giá trị', 'Elasticity trung bình']
            ).properties(title=f"Elasticity theo {factor}", width=300)
            st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Phân tích nhạy cảm giá theo điều kiện")
        for factor in df_el['Yếu tố'].unique():
            factor_data = df_el[df_el['Yếu tố'] == factor]
            if len(factor_data) >= 2:
                values = factor_data['Elasticity trung bình'].values
                diff = abs(values[0] - values[1])
                if diff > 0.5:
                    st.info(f"📌 **{factor}**: Có sự khác biệt lớn về nhạy cảm giá ({diff:.2f}). Nên điều chỉnh giá theo yếu tố này.")
                else:
                    st.write(f"📌 **{factor}**: Không có nhiều sự khác biệt về nhạy cảm giá ({diff:.2f}).")