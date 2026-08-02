"""测试数据分析模块。"""

import pandas as pd
import pytest

from src.analysis import (
    get_categorical_frequency,
    get_correlation_matrix,
    get_numeric_distribution,
    get_summary_stats,
)


@pytest.fixture
def sample_df():
    """创建包含认购数据的样本 DataFrame。"""
    data = {
        "age": [51, 50, 48, 26, 35],
        "job": ["admin.", "services", "blue-collar", "entrepreneur", "admin."],
        "marital": ["divorced", "married", "divorced", "single", "married"],
        "education": [
            "professional.course",
            "high.school",
            "basic.9y",
            "high.school",
            "university.degree",
        ],
        "default": ["no", "unknown", "no", "yes", "no"],
        "housing": ["yes", "yes", "no", "yes", "no"],
        "loan": ["yes", "no", "no", "yes", "no"],
        "contact": ["cellular", "cellular", "cellular", "cellular", "telephone"],
        "month": ["aug", "may", "apr", "aug", "jun"],
        "day_of_week": ["mon", "mon", "wed", "fri", "tue"],
        "duration": [4621, 4715, 171, 359, 1200],
        "campaign": [1, 1, 0, 26, 3],
        "pdays": [112, 412, 1027, 998, 999],
        "previous": [2, 2, 1, 0, 0],
        "poutcome": ["failure", "nonexistent", "failure", "nonexistent", "success"],
        "emp_var_rate": [1.4, -1.8, -1.8, 1.4, -0.1],
        "cons_price_index": [90.81, 96.33, 96.33, 97.08, 93.20],
        "cons_conf_index": [-35.53, -40.58, -44.74, -35.55, -42.0],
        "lending_rate3m": [0.69, 4.05, 1.50, 5.11, 3.50],
        "nr_employed": [5219.74, 4974.79, 5022.61, 5222.87, 5090.0],
        "subscribe": ["no", "yes", "no", "yes", "no"],
    }
    return pd.DataFrame(data)


class TestGetSummaryStats:
    def test_basic_structure(self, sample_df):
        """Given 样本数据，When 获取统计摘要，Then 返回结构包含所有必要键。"""
        stats = get_summary_stats(sample_df)
        assert "total_rows" in stats
        assert "total_features" in stats
        assert "num_subscribe_yes" in stats
        assert "num_subscribe_no" in stats
        assert "subscribe_rate" in stats
        assert "numeric_summary" in stats
        assert "categorical_summary" in stats

    def test_total_rows(self, sample_df):
        """Given 5 行数据，When 获取摘要，Then total_rows == 5。"""
        stats = get_summary_stats(sample_df)
        assert stats["total_rows"] == 5

    def test_subscribe_counts(self, sample_df):
        """Given 2 yes 3 no，When 获取摘要，Then 计数正确。"""
        stats = get_summary_stats(sample_df)
        assert stats["num_subscribe_yes"] == 2
        assert stats["num_subscribe_no"] == 3

    def test_subscribe_rate(self, sample_df):
        """Given 2/5 认购，When 获取摘要，Then 认购率 == 0.4。"""
        stats = get_summary_stats(sample_df)
        assert stats["subscribe_rate"] == pytest.approx(0.4)

    def test_numeric_summary_has_correct_keys(self, sample_df):
        """Given 数值特征，When 获取摘要，Then numeric_summary 包含 describe 的统计量。"""
        stats = get_summary_stats(sample_df)
        age_stats = stats["numeric_summary"]["age"]
        assert "mean" in age_stats
        assert "count" in age_stats


class TestGetNumericDistribution:
    def test_returns_yes_no_groups(self, sample_df):
        """Given 数值列，When 获取分布，Then 返回 yes 和 no 分组数据。"""
        dist = get_numeric_distribution(sample_df, "age")
        assert "age" in dist
        assert "yes" in dist["age"]
        assert "no" in dist["age"]
        assert len(dist["age"]["yes"]) == 2
        assert len(dist["age"]["no"]) == 3

    def test_unknown_column_returns_empty(self, sample_df):
        """Given 不存在的列，When 获取分布，Then 返回空字典。"""
        dist = get_numeric_distribution(sample_df, "unknown_col")
        assert dist == {}


class TestGetCategoricalFrequency:
    def test_returns_counts_and_rates(self, sample_df):
        """Given 类别列，When 获取频次，Then 返回 counts 和 subscribe_rate。"""
        freq = get_categorical_frequency(sample_df, "job")
        assert "counts" in freq
        assert "subscribe_rate" in freq
        # admin. 出现 2 次，0 次 yes → rate 0.0
        assert freq["counts"]["admin."] == 2
        assert freq["subscribe_rate"]["admin."] == pytest.approx(0.0)

    def test_unknown_column(self, sample_df):
        """Given 不存在的列，When 获取频次，Then 返回空字典。"""
        freq = get_categorical_frequency(sample_df, "unknown")
        assert freq == {"counts": {}, "subscribe_rate": {}}


class TestGetCorrelationMatrix:
    def test_returns_dict_of_dicts(self, sample_df):
        """Given 数据，When 计算相关性矩阵，Then 返回嵌套字典。"""
        corr = get_correlation_matrix(sample_df)
        # 验证是一个嵌套字典
        assert isinstance(corr, dict)
        first_col = next(iter(corr))
        assert isinstance(corr[first_col], dict)

    def test_correlation_bounds(self, sample_df):
        """Given 数据，When 计算相关性，Then 所有值在 [-1, 1] 范围内。"""
        corr = get_correlation_matrix(sample_df)
        for col1 in corr:
            for val in corr[col1].values():
                assert -1.0 <= val <= 1.0
