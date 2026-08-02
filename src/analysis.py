"""数据分析模块：统计摘要、分组聚合、相关性分析。"""

import pandas as pd

from src.data_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN


def get_summary_stats(df: pd.DataFrame) -> dict:
    """计算数据集的关键统计摘要。

    Returns:
        dict 包含:
        - total_rows, total_features
        - num_subscribe_yes / num_subscribe_no
        - subscribe_rate: 整体认购率
        - numeric_summary: 数值特征的 describe()
        - categorical_summary: 类别特征的值计数
    """
    total = len(df)
    yes_count = int((df[TARGET_COLUMN] == "yes").sum())
    no_count = total - yes_count

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    return {
        "total_rows": total,
        "total_features": len(numeric_cols) + len(categorical_cols),
        "num_subscribe_yes": yes_count,
        "num_subscribe_no": no_count,
        "subscribe_rate": yes_count / total if total > 0 else 0.0,
        "numeric_summary": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "categorical_summary": {col: df[col].value_counts().to_dict() for col in categorical_cols},
    }


def get_numeric_distribution(df: pd.DataFrame, column: str) -> dict:
    """获取数值特征的分布数据（按 subscribe 分组）。

    Returns:
        {column: {
            "yes": [values...],
            "no": [values...],
        }}
    """
    if column not in df.columns:
        return {}

    yes_mask = df[TARGET_COLUMN] == "yes"
    return {
        column: {
            "yes": df.loc[yes_mask, column].dropna().tolist(),
            "no": df.loc[~yes_mask, column].dropna().tolist(),
        }
    }


def get_categorical_frequency(df: pd.DataFrame, column: str) -> dict:
    """获取类别特征的频次及认购率。

    Returns:
        {
            "counts": {category: count, ...},
            "subscribe_rate": {category: rate, ...}
        }
    """
    if column not in df.columns:
        return {"counts": {}, "subscribe_rate": {}}

    counts = df[column].value_counts().to_dict()
    subscribe_rate = (
        df.groupby(column)[TARGET_COLUMN].apply(lambda x: (x == "yes").mean()).to_dict()
    )

    return {"counts": counts, "subscribe_rate": subscribe_rate}


def get_correlation_matrix(df: pd.DataFrame) -> dict:
    """计算数值特征之间的相关性矩阵。

    Returns:
        {col1: {col2: correlation, ...}, ...}
    """
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not numeric_cols:
        return {}

    corr = df[numeric_cols].corr().round(4)
    return corr.to_dict()
