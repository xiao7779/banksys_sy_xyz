"""预测模块：模型加载 + 推理接口。"""

from typing import Any

import pandas as pd

from src.data_loader import FEATURE_COLUMNS, NUMERIC_COLUMNS
from src.model_train import load_model


def predict(
    input_data: dict[str, Any],
    model_path: str | None = None,
) -> dict[str, Any]:
    """对单条输入数据进行预测。

    Args:
        input_data: 包含所有特征键值对的字典，如 {"age": 35, "job": "admin.", ...}
        model_path: 可选模型路径。

    Returns:
        {
            "prediction": "yes" | "no",
            "probability": float,  # 认购概率 (0~1)
            "confidence": float,   # 预测置信度 (max(prob, 1-prob))
        }

    Raises:
        FileNotFoundError: 模型文件不存在时抛出。
        ValueError: 输入数据缺少必需特征时抛出。
    """
    model_data = load_model(model_path)
    if model_data is None:
        raise FileNotFoundError("模型文件未找到。请先运行模型训练: python -m src.model_train")

    pipeline = model_data["pipeline"]
    threshold = model_data["threshold"]

    # 校验输入特征完整性
    missing = [f for f in FEATURE_COLUMNS if f not in input_data]
    if missing:
        raise ValueError(f"输入缺少以下特征: {missing}")

    # 构建单行 DataFrame
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    # 类型转换：确保数值列是数值类型
    for col in NUMERIC_COLUMNS:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # 预测（使用最优阈值）
    proba = pipeline.predict_proba(input_df)[0]  # [P(no), P(yes)]
    yes_prob = float(proba[1])
    prediction = "yes" if yes_prob >= threshold else "no"
    confidence = max(yes_prob, 1 - yes_prob)

    return {
        "prediction": prediction,
        "probability": yes_prob,
        "confidence": confidence,
    }
