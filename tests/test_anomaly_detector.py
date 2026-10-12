"""Unit tests for IsolationForestDetector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processor.anomaly_detector import DEFAULT_FEATURE_COLUMNS, IsolationForestDetector


@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Fixture providing clean training dataset with 100 normal records and 5 anomalies."""
    rng = np.random.default_rng(42)
    # 100 normal
    normal_df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:04d}" for i in range(100)],
            "amount": np.round(rng.normal(50.0, 10.0, 100), 2),
            "transaction_velocity": rng.integers(1, 4, 100),
            "distance_from_home": np.round(rng.uniform(1.0, 25.0, 100), 2),
            "device_risk_score": np.round(rng.uniform(0.01, 0.20, 100), 3),
            "is_anomaly": 0,
        }
    )
    # 5 anomalies
    anom_df = pd.DataFrame(
        {
            "transaction_id": [f"ANOM_{i:04d}" for i in range(5)],
            "amount": [8000.0, 9500.0, 12000.0, 5500.0, 7500.0],
            "transaction_velocity": [35, 45, 50, 40, 30],
            "distance_from_home": [3500.0, 4200.0, 6000.0, 2800.0, 5100.0],
            "device_risk_score": [0.95, 0.98, 0.92, 0.89, 0.96],
            "is_anomaly": 1,
        }
    )
    return pd.concat([normal_df, anom_df], ignore_index=True)


def test_init_default_parameters():
    detector = IsolationForestDetector()
    assert detector.contamination == 0.05
    assert detector.random_state == 42
    assert detector.n_estimators == 100
    assert detector.feature_columns == DEFAULT_FEATURE_COLUMNS
    assert detector.is_fitted is False
    assert detector.model is None


def test_preprocess_missing_feature_column_raises():
    detector = IsolationForestDetector()
    invalid_df = pd.DataFrame({"amount": [10.0], "transaction_velocity": [1]})
    with pytest.raises(ValueError, match="Missing required feature column"):
        detector.preprocess(invalid_df)


def test_preprocess_empty_df_raises():
    detector = IsolationForestDetector()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        detector.preprocess(pd.DataFrame())


def test_preprocess_imputation_handles_nan_and_strings():
    detector = IsolationForestDetector()
    df = pd.DataFrame(
        {
            "amount": [50.0, "invalid_num", np.nan, 70.0],
            "transaction_velocity": [1, 2, 3, 4],
            "distance_from_home": [10.0, 15.0, 20.0, 25.0],
            "device_risk_score": [0.1, 0.2, 0.3, 0.4],
        }
    )
    # Train preprocess to capture medians
    X_train, _ = detector.preprocess(df, is_training=True)
    assert not np.isnan(X_train).any()
    # Check median was correctly computed
    assert detector.imputer_medians["amount"] == 60.0  # median of [50.0, 70.0]


def test_predict_before_fit_raises():
    detector = IsolationForestDetector()
    df = pd.DataFrame(
        {
            "amount": [50.0],
            "transaction_velocity": [1],
            "distance_from_home": [10.0],
            "device_risk_score": [0.1],
        }
    )
    with pytest.raises(RuntimeError, match="Detector is not fitted"):
        detector.predict(df)


def test_fit_and_predict(sample_training_df: pd.DataFrame):
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(sample_training_df)
    assert detector.is_fitted is True
    assert detector.model is not None

    preds_df = detector.predict(sample_training_df)
    assert "is_anomaly" in preds_df.columns
    assert "decision_score" in preds_df.columns
    assert "anomaly_score" in preds_df.columns
    assert "severity" in preds_df.columns

    # Check that decision_score is lower for anomalies and severity is non-trivial
    anomalies = preds_df[preds_df["transaction_id"].str.startswith("ANOM_")]
    assert len(anomalies) > 0
    # True anomalies should have lower decision scores
    normal = preds_df[preds_df["transaction_id"].str.startswith("TXN_")]
    assert anomalies["decision_score"].mean() < normal["decision_score"].mean()
    assert anomalies["anomaly_score"].mean() > normal["anomaly_score"].mean()


def test_evaluate_metrics(sample_training_df: pd.DataFrame):
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(sample_training_df)
    metrics = detector.evaluate(sample_training_df, ground_truth_col="is_anomaly")

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["sample_count"] == len(sample_training_df)
    assert metrics["ground_truth_anomalies"] == 5
    assert metrics["roc_auc"] > 0.80  # Outliers are extreme, ROC-AUC should be high


def test_save_and_load_persistence(sample_training_df: pd.DataFrame):
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(sample_training_df)
    orig_preds = detector.predict(sample_training_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.joblib"
        detector.save(model_path)
        assert model_path.exists()

        loaded_detector = IsolationForestDetector.load(model_path)
        assert loaded_detector.is_fitted is True
        assert loaded_detector.contamination == detector.contamination
        assert loaded_detector.feature_columns == detector.feature_columns

        loaded_preds = loaded_detector.predict(sample_training_df)
        pd.testing.assert_series_equal(orig_preds["is_anomaly"], loaded_preds["is_anomaly"])
        pd.testing.assert_series_equal(orig_preds["decision_score"], loaded_preds["decision_score"])
