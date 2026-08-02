"""测试模型训练模块。"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_train import (
    build_pipeline,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_pipeline,
)


@pytest.fixture
def train_df():
    """创建可用于训练的合成数据（200 行，包含所有特征列）。"""
    np.random.seed(42)
    n = 200

    data = {}
    # 数值特征
    data["age"] = np.random.randint(18, 90, n)
    data["duration"] = np.random.randint(0, 5000, n)
    data["campaign"] = np.random.randint(1, 30, n)
    data["pdays"] = np.random.choice([999, 0, 1, 2, 3, 5, 10], n)
    data["previous"] = np.random.randint(0, 5, n)
    data["emp_var_rate"] = np.random.choice([-1.8, 1.4, -0.1, 0.5], n)
    data["cons_price_index"] = np.random.uniform(90, 100, n)
    data["cons_conf_index"] = np.random.uniform(-50, -30, n)
    data["lending_rate3m"] = np.random.uniform(0.5, 5.5, n)
    data["nr_employed"] = np.random.uniform(4900, 5300, n)

    # 类别特征
    data["job"] = np.random.choice(
        ["admin.", "services", "blue-collar", "technician", "management"], n
    )
    data["marital"] = np.random.choice(["married", "single", "divorced"], n)
    data["education"] = np.random.choice(
        ["high.school", "university.degree", "basic.9y", "professional.course"], n
    )
    data["default"] = np.random.choice(["no", "yes", "unknown"], n)
    data["housing"] = np.random.choice(["yes", "no", "unknown"], n)
    data["loan"] = np.random.choice(["no", "yes", "unknown"], n)
    data["contact"] = np.random.choice(["cellular", "telephone"], n)
    data["month"] = np.random.choice(
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        n,
    )
    data["day_of_week"] = np.random.choice(["mon", "tue", "wed", "thu", "fri"], n)
    data["poutcome"] = np.random.choice(["failure", "nonexistent", "success"], n)

    # 生成有一定模式的目标（确保模型能学到东西）
    subscribe_prob = (
        (data["duration"] > 2000).astype(float) * 0.4
        + (pd.Series(data["poutcome"]) == "success").astype(float) * 0.3
        + (pd.Series(data["previous"]) > 2).astype(float) * 0.2
        + np.random.uniform(0, 0.2, n)
    )
    data["subscribe"] = np.where(subscribe_prob > 0.5, "yes", "no")

    return pd.DataFrame(data)


class TestBuildPipeline:
    def test_returns_pipeline(self):
        """Given 无参数，When 构建管道，Then 返回 Pipeline 对象。"""
        from sklearn.pipeline import Pipeline

        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor_and_classifier(self):
        """Given 管道构建完成，When 检查步骤，Then 包含 preprocessor 和 classifier。"""
        pipe = build_pipeline()
        steps = dict(pipe.named_steps)
        assert "preprocessor" in steps
        assert "classifier" in steps


class TestPrepareData:
    def test_returns_four_arrays(self, train_df):
        """Given 数据，When 准备数据，Then 返回 4 个元素 (X_train, X_test, y_train, y_test)。"""
        result = prepare_data(train_df)
        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert len(X_train) > 0 and len(X_test) > 0
        assert len(y_train) > 0 and len(y_test) > 0

    def test_stratified_split_preserves_ratio(self, train_df):
        """Given 数据，When 分层划分，Then 训练集和测试集的类别比例大致相同。"""
        _X_train, _X_test, y_train, y_test = prepare_data(train_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # 比例差异应在合理范围内
        assert abs(train_rate - test_rate) < 0.2

    def test_y_values_are_binary(self, train_df):
        """Given 数据，When 标签编码后，Then y 值为 0 或 1。"""
        _, _, y_train, y_test = prepare_data(train_df)
        for val in y_train:
            assert val in (0, 1)
        for val in y_test:
            assert val in (0, 1)


class TestTrainPipeline:
    def test_returns_model_and_metrics(self, train_df):
        """Given 训练数据，When 执行训练流程，Then 返回模型字典和指标字典。"""
        model, metrics = train_pipeline(train_df)
        assert isinstance(model, dict)
        assert "pipeline" in model
        assert "threshold" in model
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "f1" in metrics
        assert "optimal_f1" in metrics
        assert "optimal_threshold" in metrics

    def test_auc_in_range(self, train_df):
        """Given 合成数据，When 训练后评估，Then AUC 在 [0, 1] 范围内。"""
        _, metrics = train_pipeline(train_df)
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_f1_in_range(self, train_df):
        """Given 合成数据，When 训练后评估，Then F1 在 [0, 1] 范围内。"""
        _, metrics = train_pipeline(train_df)
        assert 0.0 <= metrics["f1"] <= 1.0


class TestEvaluateModel:
    def test_returns_required_keys(self, train_df):
        """Given 训练好的模型和测试数据，When 评估，Then 返回所有必要指标。"""
        X_train, X_test, y_train, y_test = prepare_data(train_df)
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert "optimal_threshold" in metrics
        assert "optimal_f1" in metrics
        assert "classification_report" in metrics


class TestSaveAndLoadModel:
    def test_save_and_load_roundtrip(self, train_df):
        """Given 训练好的模型，When 保存后加载，Then 加载的模型预测结果与原始模型一致。"""
        pipe = build_pipeline()
        X_train, X_test, y_train, _y_test = prepare_data(train_df)
        pipe.fit(X_train, y_train)

        model_data = {"pipeline": pipe, "threshold": 0.5}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_model(model_data, tmp_path)
            loaded = load_model(tmp_path)
            assert loaded is not None
            assert "pipeline" in loaded
            assert "threshold" in loaded

            loaded_pipe = loaded["pipeline"]
            original_preds = pipe.predict(X_test)[:10]
            loaded_preds = loaded_pipe.predict(X_test)[:10]
            assert np.array_equal(original_preds, loaded_preds)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_nonexistent_returns_none(self):
        """Given 不存在的模型路径，When 加载，Then 返回 None。"""
        result = load_model("/nonexistent/model.pkl")
        assert result is None
