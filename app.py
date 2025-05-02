import streamlit as st
import pandas as pd
from utils.data_processing import load_and_merge_data
from pages.data_tab import render_data_tab
from pages.optimal_price_tab import render_optimal_price_tab
from pages.price_analysis_tab import render_price_analysis_tab
from pages.price_change_tab import render_price_change_tab
from pages.competitor_tab import render_competitor_tab
from pages.price_qty_tab import render_price_qty_tab
from pages.seasonal_trend_tab import render_seasonal_trend_tab
from pages.discount_tab import render_discount_tab
from pages.promo_tab import render_promo_tab
from pages.adjust_product_tab import render_adjust_product_tab
from pages.personalized_pricing_tab import render_personalized_pricing_tab

st.set_page_config(page_title="Tối Ưu Giá Bán Cafe & Phân Tích Đối Thủ", layout="wide")
st.title("☕ Ứng dụng Tối Ưu & Phân Tích Giá Bán Cafe Shop (Cạnh Tranh)")

# Sidebar: Upload dữ liệu
st.sidebar.header("🚀 Upload dữ liệu")
u_meta = st.sidebar.file_uploader("Sell Meta Data (CSV)", type="csv")
u_trans = st.sidebar.file_uploader("Transaction Store (CSV)", type="csv")
u_date = st.sidebar.file_uploader("Date Info (CSV)", type="csv")

if not (u_meta and u_trans and u_date):
    st.sidebar.info("Vui lòng upload cả 3 file để bắt đầu!")
    st.info("Chào mừng bạn đến với ứng dụng phân tích giá bán cà phê! Vui lòng tải lên 3 tệp CSV để bắt đầu phân tích:")
    st.markdown("""
    - **Sell Meta Data**: Chứa thông tin về sản phẩm và danh mục
    - **Transaction Store**: Chứa dữ liệu giao dịch bán hàng
    - **Date Info**: Chứa thông tin về ngày (lễ, cuối tuần, mùa vụ)
    """)
    st.stop()

# Load và merge dữ liệu
try:
    merged, sell_meta, transaction, date_info = load_and_merge_data(u_meta, u_trans, u_date)
    st.sidebar.success("Dữ liệu đã được tải thành công!")
except Exception as e:
    st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    st.stop()

# Sidebar: Chọn sản phẩm
items = merged['ITEM_NAME'].dropna().unique().tolist()
selected_items = st.sidebar.multiselect("🛒 Chọn 1 hoặc 2 sản phẩm:", items, max_selections=2)
if not selected_items:
    st.sidebar.info("Chọn ít nhất 1 sản phẩm.")
    st.stop()

# Lọc dữ liệu theo sản phẩm
if len(selected_items) == 1:
    df_prod = merged[merged['ITEM_NAME'] == selected_items[0]].copy()
    combo_label = selected_items[0]
else:
    df_prod = merged[merged['ITEM_NAME'].isin(selected_items)].copy()
    combo_label = ' + '.join(selected_items)
    df_prod['ITEM_NAME'] = combo_label

if df_prod.empty:
    st.warning("Không có dữ liệu cho lựa chọn này.")
    st.stop()

# Tạo các tab
tabs = st.tabs([
    "📋 Dữ liệu", "📈 Giá tối ưu", "🔍 Phân tích giá", "💰 Thay đổi giá",
    "🏢 Đối thủ", "📊 So sánh giá & SL", "🌸 Xu hướng theo mùa", "📉 Giảm giá",
    "🎯 Tối ưu CTKM", "📦 Sản phẩm cần điều chỉnh", "👤 Định giá cá nhân hóa"
])

# Gọi hàm render cho từng tab
with tabs[0]:
    render_data_tab(df_prod)
with tabs[1]:
    render_optimal_price_tab(df_prod, combo_label)
with tabs[2]:
    render_price_analysis_tab(df_prod, combo_label)
with tabs[3]:
    render_price_change_tab(df_prod, combo_label)
with tabs[4]:
    render_competitor_tab(df_prod, combo_label)
with tabs[5]:
    render_price_qty_tab(df_prod, combo_label)
with tabs[6]:
    render_seasonal_trend_tab(df_prod, combo_label)
with tabs[7]:
    render_discount_tab(df_prod, combo_label)
with tabs[8]:
    render_promo_tab(df_prod, combo_label, items)
with tabs[9]:
    render_adjust_product_tab(merged)
with tabs[10]:
    render_personalized_pricing_tab(merged)