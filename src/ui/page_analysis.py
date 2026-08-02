"""数据分析页面：多维度交互式可视化探索。"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import (
    get_categorical_frequency,
    get_correlation_matrix,
    get_numeric_distribution,
    get_summary_stats,
)
from src.data_loader import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    load_data,
)

DATA_PATH = "data/train.csv"


def _load_data():
    """带缓存的加载数据。"""
    if "data" not in st.session_state:
        st.session_state.data = load_data(DATA_PATH)
    return st.session_state.data


def _render_overview(df, stats):
    """渲染数据总览卡片。"""
    st.header("📋 数据总览")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总样本数", f"{stats['total_rows']:,}")
    with col2:
        st.metric("特征数", stats["total_features"])
    with col3:
        st.metric("认购数 (yes)", f"{stats['num_subscribe_yes']:,}")
    with col4:
        st.metric("未认购数 (no)", f"{stats['num_subscribe_no']:,}")
    with col5:
        st.metric("认购率", f"{stats['subscribe_rate']:.2%}")

    st.markdown("---")


def _render_numeric_distribution(df):
    """渲染数值特征分布（直方图 + 箱线图）。"""
    st.header("📈 数值特征分布")

    selected_col = st.selectbox(
        "选择数值特征",
        [c for c in NUMERIC_COLUMNS if c in df.columns],
        key="numeric_dist_select",
    )

    if selected_col:
        dist = get_numeric_distribution(df, selected_col)
        if not dist:
            return

        yes_vals = dist[selected_col]["yes"]
        no_vals = dist[selected_col]["no"]

        # 使用 subplots：直方图 + 箱线图上下排列
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(f"{selected_col} 分布直方图", f"{selected_col} 箱线图"),
            vertical_spacing=0.15,
        )

        # 直方图
        fig.add_trace(
            go.Histogram(
                x=yes_vals,
                name="认购(yes)",
                marker_color="#2E86AB",
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=no_vals,
                name="未认购(no)",
                marker_color="#A23B72",
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text=selected_col, row=1, col=1)
        fig.update_yaxes(title_text="频次", row=1, col=1)

        # 箱线图
        fig.add_trace(
            go.Box(x=yes_vals, name="认购(yes)", marker_color="#2E86AB", boxmean="sd"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Box(x=no_vals, name="未认购(no)", marker_color="#A23B72", boxmean="sd"),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=selected_col, row=2, col=1)

        fig.update_layout(
            height=500,
            showlegend=True,
            barmode="overlay",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


def _render_categorical_frequency(df):
    """渲染类别特征频次与认购率。"""
    st.header("📊 类别特征分析")

    selected_col = st.selectbox(
        "选择类别特征",
        [c for c in CATEGORICAL_COLUMNS if c in df.columns],
        key="cat_freq_select",
    )

    if selected_col:
        freq = get_categorical_frequency(df, selected_col)
        if not freq["counts"]:
            return

        categories = list(freq["counts"].keys())
        counts = list(freq["counts"].values())
        rates = [freq["subscribe_rate"].get(c, 0) for c in categories]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"{selected_col} 频次分布", f"{selected_col} 各分类认购率"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                name="频次",
                marker_color="#2E86AB",
                text=counts,
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=categories,
                y=rates,
                name="认购率",
                marker_color="#A23B72",
                text=[f"{r:.1%}" for r in rates],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            template="plotly_white",
        )
        fig.update_yaxes(title_text="数量", row=1, col=1)
        fig.update_yaxes(title_text="认购率", tickformat=".0%", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


def _render_correlation(df):
    """渲染相关性热力图。"""
    st.header("🔗 数值特征相关性")

    corr_dict = get_correlation_matrix(df)
    if not corr_dict:
        st.info("无可用数值特征")
        return

    numeric_cols = list(corr_dict.keys())
    z_matrix = [[corr_dict[c1][c2] for c2 in numeric_cols] for c1 in numeric_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar={"title": "相关系数"},
        )
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis={"tickangle": 45},
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


def render():
    """渲染数据分析页面。"""
    st.title("📊 银行营销数据分析")

    try:
        df = _load_data()
    except FileNotFoundError:
        st.error("数据文件未找到。请确保 `data/train.csv` 存在。")
        return

    stats = get_summary_stats(df)

    _render_overview(df, stats)
    _render_numeric_distribution(df)
    _render_categorical_frequency(df)
    _render_correlation(df)
