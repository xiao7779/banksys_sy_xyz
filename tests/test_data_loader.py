"""测试数据加载模块。"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    get_column_info,
    get_feature_domains,
    get_subscribe_rate,
    load_data,
)


@pytest.fixture
def sample_csv_path():
    """创建包含银行营销数据结构的临时 CSV 文件。"""
    data = {
        "id": [1, 2, 3, 4, 5],
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
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


class TestLoadData:
    def test_load_valid_csv(self, sample_csv_path):
        """Given 有效的 CSV 文件路径，When 调用 load_data，Then 返回非空 DataFrame。"""
        df = load_data(sample_csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_nonexistent_file(self):
        """Given 不存在的文件路径，When 调用 load_data，Then 抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.csv")

    def test_all_expected_columns_present(self, sample_csv_path):
        """Given 银行营销 CSV，When 加载数据，Then 包含所有预期列。"""
        df = load_data(sample_csv_path)
        expected_cols = {*FEATURE_COLUMNS, TARGET_COLUMN, "id"}
        assert expected_cols.issubset(set(df.columns))


class TestGetColumnInfo:
    def test_returns_correct_structure(self, sample_csv_path):
        """Given 已加载的 DataFrame，When 调用 get_column_info，Then 返回结构包含必要键。"""
        df = load_data(sample_csv_path)
        info = get_column_info(df)
        assert "total_rows" in info
        assert "total_features" in info
        assert "numeric_columns" in info
        assert "categorical_columns" in info
        assert "target_column" in info
        assert "has_missing" in info
        assert "missing_counts" in info

    def test_total_rows(self, sample_csv_path):
        """Given 5 行数据，When 获取列信息，Then total_rows == 5。"""
        df = load_data(sample_csv_path)
        info = get_column_info(df)
        assert info["total_rows"] == 5

    def test_numeric_columns(self, sample_csv_path):
        """Given 数据，When 获取列信息，Then numeric_columns 包含预期数值列。"""
        df = load_data(sample_csv_path)
        info = get_column_info(df)
        for col in NUMERIC_COLUMNS:
            assert col in info["numeric_columns"]

    def test_categorical_columns(self, sample_csv_path):
        """Given 数据，When 获取列信息，Then categorical_columns 包含预期类别列。"""
        df = load_data(sample_csv_path)
        info = get_column_info(df)
        for col in CATEGORICAL_COLUMNS:
            assert col in info["categorical_columns"]

    def test_no_missing_in_sample(self, sample_csv_path):
        """Given 无缺失的样本数据，When 获取列信息，Then has_missing 为 False。"""
        df = load_data(sample_csv_path)
        info = get_column_info(df)
        assert info["has_missing"] is False


class TestGetSubscribeRate:
    def test_subscribe_rate(self, sample_csv_path):
        """Given 5 行(2 yes, 3 no)，When 计算认购率，Then 返回 0.4。"""
        df = load_data(sample_csv_path)
        rate = get_subscribe_rate(df)
        assert rate == pytest.approx(0.4)

    def test_no_subscribe_column(self):
        """Given 无 subscribe 列的 DataFrame，When 计算认购率，Then 返回 0。"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        rate = get_subscribe_rate(df)
        assert rate == 0.0


class TestGetFeatureDomains:
    def test_returns_dict_with_categorical_keys(self, sample_csv_path):
        """Given 数据，When 获取特征域，Then 返回包含所有类别特征的字典。"""
        df = load_data(sample_csv_path)
        domains = get_feature_domains(df)
        for col in CATEGORICAL_COLUMNS:
            assert col in domains

    def test_domains_are_sorted(self, sample_csv_path):
        """Given 数据，When 获取特征域，Then 每个特征的值列表已排序。"""
        df = load_data(sample_csv_path)
        domains = get_feature_domains(df)
        for values in domains.values():
            assert values == sorted(values)
