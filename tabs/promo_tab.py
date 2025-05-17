import streamlit as st
import pandas as pd
import altair as alt
from utils.modeling import train_polynomial_model, predict_revenue
from utils.data_processing import clean_data

def render_promo_tab(df_prod, combo_label, items):
    """Hiển thị nội dung tab Tối ưu CTKM"""
    st.header("🎯 Tối ưu hóa chương trình khuyến mãi dựa trên giá")
    
    df_clean = clean_data(df_prod)
    model, poly_features, _ = train_polynomial_model(df_clean)
    base_price = df_clean['PRICE'].mean()
    current_qty = df_clean['QUANTITY'].mean()
    
    promo_type = st.radio(
        "Loại chương trình khuyến mãi",
        ["Giảm giá trực tiếp", "Mua 1 tặng 1", "Combo giảm giá", "Giảm giá theo số lượng"],
        horizontal=True
    )
    
    if promo_type == "Giảm giá trực tiếp":
        st.subheader("Giảm giá trực tiếp")
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_direct_discount"
        )
        cost_price = base_price * (cost_pct / 100)
        
        discount_range = range(0, 55, 5)
        profit_results = []
        current_profit = (base_price - cost_price) * current_qty
        
        for d in discount_range:
            adj_price = base_price * (1 - d/100)
            adj_revenue = predict_revenue(model, poly_features, adj_price)
            adj_qty = adj_revenue / adj_price if adj_price > 0 else 0
            adj_profit = (adj_price - cost_price) * adj_qty
            profit_pct_change = ((adj_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
            profit_results.append({
                'Giảm giá (%)': d,
                'Giá sau giảm': round(adj_price, 2),
                'Số lượng dự đoán': round(adj_qty, 2),
                'Doanh thu dự đoán': round(adj_revenue, 2),
                'Lợi nhuận dự đoán': round(adj_profit, 2),
                'Thay đổi lợi nhuận (%)': round(profit_pct_change, 2)
            })
        
        profit_df = pd.DataFrame(profit_results)
        opt_profit_discount = profit_df.loc[profit_df['Lợi nhuận dự đoán'].idxmax()]
        
        st.dataframe(profit_df)
        st.success(f"✅ Mức giảm giá tối ưu cho lợi nhuận: **{opt_profit_discount['Giảm giá (%)']}%** - Lợi nhuận dự đoán: **{opt_profit_discount['Lợi nhuận dự đoán']:.2f}**")
        
        melted_df = profit_df.melt(
            id_vars=['Giảm giá (%)'], 
            value_vars=['Doanh thu dự đoán', 'Lợi nhuận dự đoán'], 
            var_name='Chỉ số', 
            value_name='Giá trị'
        )
        chart = alt.Chart(melted_df).mark_line(point=True).encode(
            x=alt.X('Giảm giá (%):Q', title='Giảm giá (%)'),
            y=alt.Y('Giá trị:Q', title='Giá trị'),
            color=alt.Color('Chỉ số:N', title='Chỉ số'),
            tooltip=['Giảm giá (%)', 'Chỉ số', 'Giá trị']
        ).properties(
            title='So sánh doanh thu và lợi nhuận theo mức giảm giá'
        )
        st.altair_chart(chart, use_container_width=True)
    
    elif promo_type == "Mua 1 tặng 1":
        st.subheader("Phân tích chương trình Mua 1 tặng 1")
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_bogo"
        )
        cost_price = base_price * (cost_pct / 100)
        
        effective_price = base_price * 0.5
        effective_revenue = predict_revenue(model, poly_features, effective_price)
        effective_qty = (effective_revenue / effective_price) * 2 if effective_price > 0 else 0
        effective_revenue = base_price * effective_qty / 2
        effective_cost = cost_price * effective_qty
        effective_profit = effective_revenue - effective_cost
        
        current_revenue = base_price * current_qty
        current_cost = cost_price * current_qty
        current_profit = current_revenue - current_cost
        profit_change = ((effective_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số lượng dự đoán", f"{effective_qty:.2f}")
            st.metric("Doanh thu dự đoán", f"{effective_revenue:.2f}")
        with col2:
            st.metric("Chi phí dự đoán", f"{effective_cost:.2f}")
            st.metric("Lợi nhuận dự đoán", f"{effective_profit:.2f}", f"{profit_change:.2f}%")
        
        st.subheader("Đánh giá chương trình")
        if profit_change > 0:
            st.success(f"✅ Chương trình Mua 1 tặng 1 dự kiến tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.error(f"❌ Chương trình Mua 1 tặng 1 dự kiến giảm lợi nhuận {abs(profit_change):.2f}%. Không nên áp dụng.")
    
    elif promo_type == "Combo giảm giá":
        st.subheader("Phân tích chương trình Combo giảm giá")
        second_product = st.selectbox("Chọn sản phẩm thứ 2 cho combo", items)
        combo_discount = st.slider("Giảm giá cho combo (%)", 5, 30, 15, 5)
        
        second_price = df_prod[df_prod['ITEM_NAME'] == second_product]['PRICE'].mean() if second_product in df_prod['ITEM_NAME'].values else base_price
        total_price = base_price + second_price
        combo_price = total_price * (1 - combo_discount / 100)
        
        st.write(f"Giá gốc của hai sản phẩm: {total_price:.2f}")
        st.write(f"Giá combo sau giảm: {combo_price:.2f}")
        
        equivalent_single_price = combo_price / 2
        equivalent_revenue = predict_revenue(model, poly_features, equivalent_single_price)
        estimated_combo_qty = (equivalent_revenue / equivalent_single_price) * 0.5 if equivalent_single_price > 0 else 0
        
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_combo_discount"
        )
        cost_price_first = base_price * (cost_pct / 100)
        cost_price_second = second_price * (cost_pct / 100)
        total_cost_per_combo = cost_price_first + cost_price_second
        combo_profit = (combo_price - total_cost_per_combo) * estimated_combo_qty
        
        current_revenue = base_price * current_qty
        current_profit = (base_price - cost_price_first) * current_qty
        profit_change = ((combo_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số lượng combo dự đoán", f"{estimated_combo_qty:.2f}")
        with col2:
            st.metric("Lợi nhuận combo dự đoán", f"{combo_profit:.2f}", f"{profit_change:.2f}%")
        
        st.subheader("Đánh giá chương trình Combo giảm giá")
        if profit_change > 0:
            st.success(f"✅ Chương trình combo giảm giá {combo_discount}% dự kiến tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.warning(f"⚠️ Chương trình combo giảm giá {combo_discount}% dự kiến giảm lợi nhuận {abs(profit_change):.2f}%. Cần cân nhắc thêm.")
    
    elif promo_type == "Giảm giá theo số lượng":
        st.subheader("Phân tích chương trình Giảm giá theo số lượng")
        min_qty = st.number_input("Số lượng tối thiểu để áp dụng giảm giá", min_value=2, value=3, step=1)
        qty_discount_pct = st.slider("Mức giảm giá khi mua từ số lượng tối thiểu (%)", 5, 30, 10, 5)
        
        cost_pct = st.slider(
            "Chi phí (% giá bán)", 30, 70, 50, 5, 
            help="Chi phí sản xuất/nhập hàng tính theo % giá bán",
            key="cost_pct_quantity_discount"
        )
        cost_price = base_price * (cost_pct / 100)
        
        discounted_price = base_price * (1 - qty_discount_pct / 100)
        discounted_revenue = predict_revenue(model, poly_features, discounted_price)
        discounted_qty = discounted_revenue / discounted_price if discounted_price > 0 else 0
        estimated_qty = discounted_qty * (min_qty / 2)
        revenue_qty_discount = discounted_price * estimated_qty
        profit_qty_discount = (discounted_price - cost_price) * estimated_qty
        
        current_revenue = base_price * current_qty
        current_profit = (base_price - cost_price) * current_qty
        profit_change = ((profit_qty_discount - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số lượng dự đoán", f"{estimated_qty:.2f}")
            st.metric("Doanh thu dự đoán", f"{revenue_qty_discount:.2f}")
        with col2:
            st.metric("Lợi nhuận dự đoán", f"{profit_qty_discount:.2f}", f"{profit_change:.2f}%")
        
        st.subheader("Đánh giá chương trình Giảm giá theo số lượng")
        if profit_change > 0:
            st.success(f"✅ Chương trình giảm giá {qty_discount_pct}% khi mua từ {min_qty} sản phẩm dự kiến tăng lợi nhuận {profit_change:.2f}%. Nên áp dụng.")
        else:
            st.warning(f"⚠️ Chương trình giảm giá {qty_discount_pct}% khi mua từ {min_qty} sản phẩm dự kiến giảm lợi nhuận {abs(profit_change):.2f}%. Cần cân nhắc thêm.")