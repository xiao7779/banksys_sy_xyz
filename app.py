"""银行营销数据分析与预测系统 — Streamlit 入口。"""

import streamlit as st

# 页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="银行营销数据分析与预测",
    page_icon="🏦",
    layout="wide",
)

# 侧边栏导航
st.sidebar.title("🏦 银行营销系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "选择功能",
    ["📊 数据分析", "🔮 在线预测"],
)

st.sidebar.markdown("---")
st.sidebar.caption("基于银行营销数据的交互式分析与认购预测")

# 根据选择加载对应页面
if page == "📊 数据分析":
    from src.ui.page_analysis import render

    render()
elif page == "🔮 在线预测":
    from src.ui.page_prediction import render

    render()
