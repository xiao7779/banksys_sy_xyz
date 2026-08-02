"""在线预测页面：点选表单输入客户特征，实时预测认购意向。"""

import streamlit as st

from src.data_loader import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    get_feature_domains,
    load_data,
)
from src.predict import predict

DATA_PATH = "data/train.csv"

# 数值特征的默认值（基于训练集中位数，供首次加载用）
NUMERIC_DEFAULTS = {
    "age": 38,
    "duration": 258,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "emp_var_rate": 1.1,
    "cons_price_index": 93.58,
    "cons_conf_index": -41.0,
    "lending_rate3m": 3.5,
    "nr_employed": 5130.0,
}


def _load_feature_domains():
    """加载类别特征的取值域。"""
    if "feature_domains" not in st.session_state:
        try:
            df = load_data(DATA_PATH)
            st.session_state.feature_domains = get_feature_domains(df)
        except FileNotFoundError:
            st.session_state.feature_domains = {}
    return st.session_state.feature_domains


def _render_input_form(domains: dict) -> dict:
    """渲染输入表单，返回用户填写的特征字典。"""
    st.markdown("### 📝 客户特征输入")
    st.caption("请填写以下客户信息，点击「预测」按钮查看认购意向")

    input_data = {}
    col1, col2 = st.columns(2)

    # 确定字段在两个列中的分配
    all_fields = []
    for field in FEATURE_COLUMNS:
        if field in CATEGORICAL_COLUMNS:
            field_type = "categorical"
        elif field in NUMERIC_COLUMNS:
            field_type = "numeric"
        else:
            field_type = "numeric"
        all_fields.append((field, field_type))

    mid = len(all_fields) // 2
    left_fields = all_fields[:mid]
    right_fields = all_fields[mid:]

    # 左栏
    with col1:
        for field, ftype in left_fields:
            input_data[field] = _render_field(field, ftype, domains)

    # 右栏
    with col2:
        for field, ftype in right_fields:
            input_data[field] = _render_field(field, ftype, domains)

    return input_data


def _render_field(field: str, ftype: str, domains: dict):
    """渲染单个输入字段。"""
    label = field.replace("_", " ").title()

    if ftype == "categorical":
        options = domains.get(field, [])
        if not options:
            options = ["unknown"]
        return st.selectbox(
            label,
            options=options,
            key=f"pred_{field}",
        )
    else:
        default = NUMERIC_DEFAULTS.get(field, 0)
        step = 0.01 if field.endswith(("_rate", "_index")) else 1

        if field == "duration":
            return st.number_input(
                f"{label} (可选,填0表示未知)",
                min_value=0,
                max_value=10000,
                value=0,
                step=1,
                key=f"pred_{field}",
                help="通话时长。实际业务中事前未知，填 0 表示使用默认值。",
            )

        return st.number_input(
            label,
            value=float(default),
            step=step,
            key=f"pred_{field}",
        )


def render():
    """渲染预测页面。"""
    st.title("🔮 在线认购预测")

    # 加载特征取值域
    domains = _load_feature_domains()

    if not domains:
        st.warning("未能加载训练数据的特征取值域，部分下拉选项可能不完整。")

    # 渲染输入表单
    input_data = _render_input_form(domains)

    st.markdown("---")

    # 预测按钮
    if st.button("🔍 预测认购意向", type="primary", use_container_width=True):
        _handle_prediction(input_data)


def _handle_prediction(input_data: dict):
    """处理预测请求。"""
    # duration 为 0 时使用中位数默认值
    if input_data.get("duration", 0) == 0:
        input_data["duration"] = NUMERIC_DEFAULTS["duration"]
        st.info(f"ℹ️ duration 未填写，已使用默认值: {NUMERIC_DEFAULTS['duration']}")

    try:
        with st.spinner("预测中..."):
            result = predict(input_data)

        # 显示结果
        st.markdown("---")
        st.markdown("### 📊 预测结果")

        col1, col2, col3 = st.columns(3)

        with col1:
            if result["prediction"] == "yes":
                st.success("✅ 预测：**会认购**")
            else:
                st.warning("❌ 预测：**不会认购**")

        with col2:
            st.metric(
                "认购概率",
                f"{result['probability']:.1%}",
            )

        with col3:
            st.metric(
                "置信度",
                f"{result['confidence']:.1%}",
            )

        # 概率可视化
        prob = result["probability"]
        st.progress(
            prob,
            text=f"认购概率: {prob:.1%}  |  不认购概率: {1 - prob:.1%}",
        )

    except FileNotFoundError:
        st.error("⚠️ 模型文件未找到。请先运行模型训练: python -m src.model_train")
    except ValueError as e:
        st.error(f"⚠️ 输入数据有误: {e}")
