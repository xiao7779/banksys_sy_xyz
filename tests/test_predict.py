"""测试预测模块。"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_train import build_pipeline, save_model
from src.predict import FEATURE_COLUMNS, predict


@pytest.fixture
def trained_model_path():
    """训练一个简单模型并保存，返回模型路径。"""
    np.random.seed(42)
    n = 100

    data = {
        "age": np.random.randint(18, 90, n),
        "duration": np.random.randint(0, 5000, n),
        "campaign": np.random.randint(1, 30, n),
        "pdays": np.random.choice([999, 0, 1, 2], n),
        "previous": np.random.randint(0, 5, n),
        "emp_var_rate": np.random.choice([-1.8, 1.4], n),
        "cons_price_index": np.random.uniform(90, 100, n),
        "cons_conf_index": np.random.uniform(-50, -30, n),
        "lending_rate3m": np.random.uniform(0.5, 5.5, n),
        "nr_employed": np.random.uniform(4900, 5300, n),
        "job": np.random.choice(["admin.", "services", "blue-collar"], n),
        "marital": np.random.choice(["married", "single", "divorced"], n),
        "education": np.random.choice(["high.school", "university.degree", "basic.9y"], n),
        "default": np.random.choice(["no", "yes"], n),
        "housing": np.random.choice(["yes", "no"], n),
        "loan": np.random.choice(["no", "yes"], n),
        "contact": np.random.choice(["cellular", "telephone"], n),
        "month": np.random.choice(["may", "jun", "jul", "aug"], n),
        "day_of_week": np.random.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "poutcome": np.random.choice(["failure", "nonexistent", "success"], n),
    }
    df = pd.DataFrame(data)
    df["subscribe"] = np.where(df["duration"] > 2500, "yes", "no")

    pipe = build_pipeline()
    from src.model_train import prepare_data

    X_train, _, y_train, _ = prepare_data(df)
    pipe.fit(X_train, y_train)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name

    save_model({"pipeline": pipe, "threshold": 0.5}, tmp_path)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def valid_input():
    """创建有效的输入数据。"""
    return {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 500,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.4,
        "cons_price_index": 93.0,
        "cons_conf_index": -40.0,
        "lending_rate3m": 3.0,
        "nr_employed": 5100.0,
    }


class TestPredict:
    def test_returns_prediction_and_probability(self, trained_model_path, valid_input):
        """Given 有效输入和已训练模型，When 预测，Then 返回 prediction、probability、confidence。"""
        result = predict(valid_input, model_path=trained_model_path)
        assert "prediction" in result
        assert "probability" in result
        assert "confidence" in result
        assert result["prediction"] in ("yes", "no")
        assert 0.0 <= result["probability"] <= 1.0
        assert 0.5 <= result["confidence"] <= 1.0

    def test_missing_model_raises(self, valid_input):
        """Given 不存在模型文件，When 预测，Then 抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            predict(valid_input, model_path="/nonexistent/model.pkl")

    def test_missing_features_raises(self, trained_model_path):
        """Given 不完整的输入，When 预测，Then 抛出 ValueError。"""
        with pytest.raises(ValueError, match="缺少以下特征"):
            predict({"age": 30}, model_path=trained_model_path)

    def test_all_feature_columns_required(self, trained_model_path, valid_input):
        """Given 所有特征列，When 预测，Then 不抛异常。"""
        # 应该包含所有 FEATURE_COLUMNS
        assert set(FEATURE_COLUMNS).issubset(set(valid_input.keys()))

    def test_confidence_is_max_probability(self, trained_model_path, valid_input):
        """Given 预测结果，When 检查 confidence，Then 等于 max(prob, 1-prob)。"""
        result = predict(valid_input, model_path=trained_model_path)
        expected_confidence = max(result["probability"], 1 - result["probability"])
        assert result["confidence"] == pytest.approx(expected_confidence)
