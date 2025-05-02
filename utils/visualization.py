import altair as alt

def create_price_revenue_chart(data, pred_df, title="Doanh thu theo giá"):
    """Tạo biểu đồ doanh thu thực tế và dự đoán"""
    chart1 = alt.Chart(data).mark_circle(size=100).encode(
        x=alt.X('PRICE', title='Giá'),
        y=alt.Y('Revenue', title='Doanh thu thực tế'),
        tooltip=['PRICE', 'QUANTITY', 'Revenue']
    ).properties(title=title)
    
    chart2 = alt.Chart(pred_df).mark_line(color='red').encode(
        x=alt.X('Giá', title='Giá'),
        y=alt.Y('Doanh thu dự đoán', title='Doanh thu dự đoán'),
        tooltip=['Giá', 'Doanh thu dự đoán']
    )
    
    return chart1 + chart2