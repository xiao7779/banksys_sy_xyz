"""数据加载模块：加载银行营销 CSV 数据，提供列信息查询。"""

from pathlib import Path

import pandas as pd

# 特征列元数据
NUMERIC_COLUMNS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

TARGET_COLUMN = "subscribe"

ID_COLUMN = "id"

# 所有特征列（不含 id 和 target）
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """加载 CSV 数据文件。

    Args:
        csv_path: CSV 文件路径。

    Returns:
        加载完成的 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    return pd.read_csv(path)


def get_column_info(df: pd.DataFrame) -> dict:
    """获取 DataFrame 列的基本信息。

    Returns:
        dict，包含:
        - total_rows: 总行数
        - total_features: 特征列数
        - numeric_columns: 数值特征列表
        - categorical_columns: 类别特征列表
        - target_column: 目标列名
        - has_missing: 是否存在缺失值
        - missing_counts: 各列缺失值数量（如有）
    """
    missing_counts = df.isnull().sum()
    has_missing = bool(missing_counts.sum() > 0)

    return {
        "total_rows": len(df),
        "total_features": len(FEATURE_COLUMNS),
        "numeric_columns": [c for c in NUMERIC_COLUMNS if c in df.columns],
        "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in df.columns],
        "target_column": TARGET_COLUMN,
        "has_missing": has_missing,
        "missing_counts": missing_counts[missing_counts > 0].to_dict() if has_missing else {},
    }


def get_subscribe_rate(df: pd.DataFrame) -> float:
    """计算认购率（subscribe == 'yes' 的占比）。"""
    if TARGET_COLUMN not in df.columns:
        return 0.0
    return (df[TARGET_COLUMN] == "yes").mean()


def get_feature_domains(df: pd.DataFrame) -> dict:
    """返回类别特征的取值域，供预测表单使用。

    Returns:
        {feature_name: [unique_values...], ...}
    """
    domains = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            # 排序以保证 UI 展示稳定
            domains[col] = sorted(df[col].dropna().unique().tolist())
    return domains
