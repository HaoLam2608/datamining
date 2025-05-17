import streamlit as st
import pandas as pd
import altair as alt
from utils.data_processing import clean_data

def classify_season(date):
    """Phân loại mùa dựa trên tháng"""
    if date.month in [3, 4, 5]:
        return 'Xuân'
    elif date.month in [6, 7, 8]:
        return 'Hạ'
    elif date.month in [9, 10, 11]:
        return 'Thu'
    else:
        return 'Đông'

def render_seasonal_trend_tab(df_prod, combo_label):
    """Hiển thị nội dung tab Xu hướng theo mùa"""
    st.header("🌸 Dự đoán xu hướng thay đổi giá theo mùa vụ")
    
    df_season = clean_data(df_prod)
    
    if 'CALENDAR_DATE' in df_season.columns:
        df_season['Season'] = df_season['CALENDAR_DATE'].apply(classify_season)
        season_avg = df_season.groupby('Season').agg({
            'PRICE': 'mean',
            'QUANTITY': 'mean'
        }).reset_index()
        season_avg['Revenue'] = season_avg['PRICE'] * season_avg['QUANTITY']
        
        if 'IS_WEEKEND' in df_season.columns:
            season_weekend_avg = df_season[df_season['IS_WEEKEND'] == 1].groupby('Season').agg({
                'PRICE': 'mean',
                'QUANTITY': 'mean'
            }).reset_index()
            season_weekend_avg = season_weekend_avg.rename(columns={
                'PRICE': 'PRICE_WEEKEND',
                'QUANTITY': 'QUANTITY_WEEKEND'
            })
            season_avg = pd.merge(season_avg, season_weekend_avg, on='Season', how='left')
        
        st.subheader("Phân tích theo mùa")
        st.dataframe(season_avg)
        
        # Vẽ biểu đồ
        chart1 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('PRICE:Q', title='Giá trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Giá trung bình theo mùa")
        
        chart2 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('QUANTITY:Q', title='Số lượng trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Số lượng trung bình theo mùa")
        
        chart3 = alt.Chart(season_avg).mark_bar().encode(
            x=alt.X('Season:O', title='Mùa', sort=['Xuân', 'Hạ', 'Thu', 'Đông']),
            y=alt.Y('Revenue:Q', title='Doanh thu trung bình'),
            color=alt.Color('Season:N', legend=None),
            tooltip=['Season', 'PRICE', 'QUANTITY', 'Revenue']
        ).properties(title="Doanh thu trung bình theo mùa")
        
        st.altair_chart(alt.vconcat(chart1, chart2, chart3), use_container_width=True)
        
        # Đề xuất điều chỉnh giá
        st.subheader("Đề xuất điều chỉnh giá theo mùa")
        max_revenue_season = season_avg.loc[season_avg['Revenue'].idxmax()]['Season']
        min_revenue_season = season_avg.loc[season_avg['Revenue'].idxmin()]['Season']
        max_revenue = season_avg['Revenue'].max()
        min_revenue = season_avg['Revenue'].min()
        revenue_diff_pct = ((max_revenue - min_revenue) / min_revenue) * 100 if min_revenue > 0 else 0
        
        if revenue_diff_pct > 20:
            st.success(f"✅ Có sự chênh lệch đáng kể về doanh thu giữa các mùa ({revenue_diff_pct:.2f}%). Nên điều chỉnh giá theo mùa.")
            for _, row in season_avg.iterrows():
                season = row['Season']
                if season == max_revenue_season:
                    st.info(f"📈 **{season}**: Doanh thu cao nhất. Có thể tăng giá 5-10%.")
                elif season == min_revenue_season:
                    st.warning(f"📉 **{season}**: Doanh thu thấp nhất. Nên khuyến mãi hoặc giảm giá 5-10%.")
                else:
                    if row['Revenue'] > season_avg['Revenue'].mean():
                        st.write(f"📊 **{season}**: Doanh thu khá tốt. Có thể giữ giá hoặc tăng nhẹ 2-5%.")
                    else:
                        st.write(f"📊 **{season}**: Doanh thu dưới trung bình. Có thể giảm nhẹ giá 2-5%.")
        else:
            st.info(f"ℹ️ Không có sự chênh lệch đáng kể về doanh thu giữa các mùa ({revenue_diff_pct:.2f}%). Có thể giữ nguyên giá.")
    else:
        st.warning("Không tìm thấy dữ liệu ngày tháng để phân tích theo mùa.")