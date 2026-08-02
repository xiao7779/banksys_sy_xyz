"""在线预测页面：点选表单输入客户特征，实时预测认购意向。"""

import streamlit as st

from src.data_loader import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    get_feature_domains,
    load_data,
)
from src.predict import predict

DATA_PATH = "data/train.csv"

NUMERIC_DEFAULTS = {
    "age": 38,
    "duration": 0,
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
    """加载类别特征的取值域（缓存到 session_state）。"""
    if "feature_domains" not in st.session_state:
        try:
            df = load_data(DATA_PATH)
            st.session_state.feature_domains = get_feature_domains(df)
        except FileNotFoundError:
            st.session_state.feature_domains = {}
    return st.session_state.feature_domains


def render():
    """渲染预测页面。"""
    st.title("🔮 在线认购预测")

    # 加载特征取值域
    domains = _load_feature_domains()
    if not domains:
        st.warning("未能加载训练数据的特征取值域，部分下拉选项可能不完整。")

    st.markdown("### 📝 客户特征输入")
    st.caption("请填写以下客户信息，点击「预测」按钮查看认购意向")

    # 使用 st.form 包裹所有输入和按钮
    with st.form("prediction_form"):
        input_data = {}

        # 两列布局
        left_cols, right_cols = st.columns(2)
        mid = len(FEATURE_COLUMNS) // 2

        for i, field in enumerate(FEATURE_COLUMNS):
            col = left_cols if i < mid else right_cols
            with col:
                label = field.replace("_", " ").title()

                if field in CATEGORICAL_COLUMNS:
                    options = domains.get(field, ["unknown"])
                    input_data[field] = st.selectbox(
                        label,
                        options=options,
                        key=f"pred_{field}",
                    )
                else:
                    is_float = field.endswith(("_rate", "_index"))
                    if is_float:
                        default_val = float(NUMERIC_DEFAULTS.get(field, 0.0))
                        step = 0.01
                    else:
                        default_val = int(NUMERIC_DEFAULTS.get(field, 0))
                        step = 1

                    if field == "duration":
                        input_data[field] = st.number_input(
                            f"{label} (可选，填0表示默认值)",
                            min_value=0,
                            max_value=10000,
                            value=0,
                            step=1,
                            key=f"pred_{field}",
                            help="通话时长。实际业务中事前未知，填 0 使用默认中位数值。",
                        )
                    else:
                        input_data[field] = st.number_input(
                            label,
                            value=default_val,
                            step=step,
                            key=f"pred_{field}",
                        )

        submitted = st.form_submit_button(
            "🔍 预测认购意向",
            type="primary",
            use_container_width=True,
        )

    # 表单提交后处理预测
    if submitted:
        _do_predict(input_data)


def _do_predict(input_data: dict):
    """执行预测并显示结果。"""
    # duration 为 0 时使用默认值
    if input_data.get("duration", 0) == 0:
        input_data["duration"] = NUMERIC_DEFAULTS["duration"]
        st.info(f"ℹ️ 通话时长未填写，已使用默认值: {NUMERIC_DEFAULTS['duration']}")

    try:
        with st.spinner("预测中..."):
            result = predict(input_data)

        st.markdown("---")
        st.markdown("### 📊 预测结果")

        c1, c2, c3 = st.columns(3)
        with c1:
            if result["prediction"] == "yes":
                st.success("✅ 预测：**会认购**")
            else:
                st.warning("❌ 预测：**不会认购**")
        with c2:
            st.metric("认购概率", f"{result['probability']:.1%}")
        with c3:
            st.metric("置信度", f"{result['confidence']:.1%}")

        prob = result["probability"]
        st.progress(prob, text=f"认购概率: {prob:.1%}  |  不认购概率: {1 - prob:.1%}")

    except FileNotFoundError:
        st.error("⚠️ 模型文件未找到。请先运行模型训练: `python -m src.model_train`")
    except ValueError as e:
        st.error(f"⚠️ 输入数据有误: {e}")
