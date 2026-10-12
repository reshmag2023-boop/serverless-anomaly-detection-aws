"""Anomaly Detection Module using Scikit-Learn Isolation Forest.

This module provides a production-ready, modular Isolation Forest detector designed for
serverless AWS Lambda execution and local training.

Isolation Forest Scoring Mechanics:
-----------------------------------
In Scikit-learn's `IsolationForest`:
- `decision_function(X)` returns average anomaly score across trees:
    - Lower, negative values indicate anomalies (records isolated with very few splits).
    - Higher, positive values indicate normal, inlier records.
- `predict(X)` returns:
    - -1 for anomalies (anomalous records).
    - +1 for normal inliers.

To make results intuitive for downstream microservices, APIs, and business operators:
- `raw_decision_score`: Raw output from `decision_function`.
- `normalized_risk_score`: Calibrated 0.0 to 1.0 score where 1.0 is extreme anomaly and 0.0 is normal.
- `severity`: Categorical classification ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = [
    "amount",
    "transaction_velocity",
    "distance_from_home",
    "device_risk_score",
]


class IsolationForestDetector:
    """Wrapper around Scikit-learn's Isolation Forest with feature validation,

    imputation, scoring, and evaluation capabilities.
    """

    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers in the data (0.0 < contamination <= 0.5).
            random_state: Seed for reproducible tree partitioning.
            n_estimators: Number of isolation trees in the ensemble.
            max_samples: Number of samples to draw from X to train each tree.
            feature_columns: List of numeric feature names expected in the dataset.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS.copy()

        self.model: Optional[IsolationForest] = None
        self.imputer_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def _init_underlying_model(self) -> IsolationForest:
        """Create a new IsolationForest instance with configured hyperparameters."""
        return IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_jobs=-1,
        )

    def preprocess(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Preprocess tabular dataframe into sanitized numeric matrix for model.

        - Extracts configured feature columns.
        - Coerces values to numeric, turning invalid values into NaN.
        - Imputes missing values using stored median values (or calculates medians if training).

        Args:
            df: Input pandas DataFrame.
            is_training: Whether preprocessing is occurring during model training.

        Returns:
            Tuple of (sanitized numpy 2D array, list of validated feature column names).

        Raises:
            ValueError: If missing required feature columns or if dataset is empty.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot preprocess.")

        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature column(s): {missing_features}")

        # Extract only configured features
        features_df = df[self.feature_columns].copy()

        # Coerce all to numeric, turning invalid values (e.g. non-numeric strings) into NaN
        for col in self.feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        if is_training:
            # Calculate and store medians for imputation
            self.imputer_medians = {}
            for col in self.feature_columns:
                med = features_df[col].median()
                # Fallback to 0.0 if entire column is NaN
                self.imputer_medians[col] = float(0.0 if pd.isna(med) else med)

        # Impute missing values using stored medians
        for col in self.feature_columns:
            default_val = self.imputer_medians.get(col, 0.0)
            features_df[col] = features_df[col].fillna(default_val)

        return features_df.values, self.feature_columns

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """Fit Isolation Forest on training DataFrame.

        Args:
            df: Training DataFrame containing required feature columns.

        Returns:
            Self instance with fitted model.
        """
        X, _ = self.preprocess(df, is_training=True)
        self.model = self._init_underlying_model()
        self.model.fit(X)
        self.is_fitted = True
        logger.info(
            "Fitted IsolationForest detector on %d records with %d features.",
            len(X),
            len(self.feature_columns),
        )
        return self

    @staticmethod
    def _compute_risk_score(raw_decision_score: np.ndarray) -> np.ndarray:
        """Convert raw decision score to calibrated 0.0 - 1.0 risk score.

        In IsolationForest:
        - decision_score is centered around 0.
        - Negative values indicate anomalies.
        - More negative => more anomalous.

        We map decision_score through a sigmoid function:
            risk = 1 / (1 + exp(10 * raw_score))
        When score == 0: risk = 0.50
        When score == -0.15: risk = 0.82
        When score == -0.30: risk = 0.95 (extreme anomaly)
        When score == +0.15: risk = 0.18 (normal)
        """
        # Clip score to prevent overflow in exp
        clipped_score = np.clip(raw_decision_score, -5.0, 5.0)
        risk = 1.0 / (1.0 + np.exp(10.0 * clipped_score))
        return np.round(risk, 4)

    @staticmethod
    def _assign_severity(raw_decision_score: float) -> str:
        """Assign categorical severity level based on Isolation Forest decision score.

        Note: Lower decision scores indicate greater anomaly likelihood.
        """
        if raw_decision_score < -0.15:
            return "CRITICAL"
        elif raw_decision_score < -0.05:
            return "HIGH"
        elif raw_decision_score < 0.0:
            return "MEDIUM"
        else:
            return "LOW"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate anomaly labels, decision scores, and risk classifications for records.

        Args:
            df: DataFrame containing records to evaluate.

        Returns:
            DataFrame with original columns plus:
            - `is_anomaly`: bool (True if anomalous)
            - `decision_score`: float (raw decision_function output; lower = more anomalous)
            - `anomaly_score`: float (normalized 0.0 - 1.0 risk score; higher = more anomalous)
            - `severity`: str ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')

        Raises:
            RuntimeError: If called before model is fitted or loaded.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Detector is not fitted. Call fit() or load() before calling predict().")

        X, _ = self.preprocess(df, is_training=False)

        # Sklearn predict returns -1 for outlier/anomaly, 1 for inlier
        raw_preds = self.model.predict(X)
        raw_decision_scores = self.model.decision_function(X)
        risk_scores = self._compute_risk_score(raw_decision_scores)

        result_df = df.copy()
        result_df["is_anomaly"] = raw_preds == -1
        result_df["decision_score"] = np.round(raw_decision_scores, 4)
        result_df["anomaly_score"] = risk_scores
        result_df["severity"] = [self._assign_severity(s) for s in raw_decision_scores]

        return result_df

    def evaluate(self, df: pd.DataFrame, ground_truth_col: str = "is_anomaly") -> Dict[str, Any]:
        """Evaluate detector performance against ground-truth labeled synthetic data.

        Args:
            df: DataFrame containing features and ground-truth boolean/int column.
            ground_truth_col: Name of column containing 1/True for anomaly, 0/False for normal.

        Returns:
            Dictionary containing metrics: precision, recall, f1, roc_auc, confusion matrix.
        """
        if ground_truth_col not in df.columns:
            raise ValueError(f"Ground truth column '{ground_truth_col}' not found in DataFrame.")

        y_true = df[ground_truth_col].astype(int).values
        pred_df = self.predict(df)
        y_pred = pred_df["is_anomaly"].astype(int).values

        # Invert decision score for ROC-AUC because higher probability should predict label=1
        y_scores = pred_df["anomaly_score"].values

        # Calculate metrics
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred).tolist()

        evaluation_report = {
            "model_version": self.MODEL_VERSION,
            "sample_count": len(df),
            "ground_truth_anomalies": int(np.sum(y_true)),
            "detected_anomalies": int(np.sum(y_pred)),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0] if len(cm) > 1 else 0,
                "false_positive": cm[0][1] if len(cm) > 1 else 0,
                "false_negative": cm[1][0] if len(cm) > 1 else 0,
                "true_positive": cm[1][1] if len(cm) > 1 else 0,
            },
            "parameters": {
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "feature_columns": self.feature_columns,
            },
        }
        return evaluation_report

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize model artifact and metadata to disk using joblib.

        Args:
            filepath: Target file path (e.g. 'models/isolation_forest_model.joblib').

        Returns:
            Resolved Path of the saved file.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_version": self.MODEL_VERSION,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "feature_columns": self.feature_columns,
            "imputer_medians": self.imputer_medians,
            "model": self.model,
        }

        joblib.dump(payload, path, compress=3)
        logger.info("Saved model artifact to %s", path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestDetector":
        """Load serialized model artifact and metadata from disk.

        Args:
            filepath: Path to saved joblib file.

        Returns:
            Restored IsolationForestDetector instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        payload: Dict[str, Any] = joblib.load(path)

        detector = cls(
            contamination=payload.get("contamination", 0.05),
            random_state=payload.get("random_state", 42),
            n_estimators=payload.get("n_estimators", 100),
            max_samples=payload.get("max_samples", "auto"),
            feature_columns=payload.get("feature_columns", DEFAULT_FEATURE_COLUMNS),
        )
        detector.imputer_medians = payload.get("imputer_medians", {})
        detector.model = payload["model"]
        detector.is_fitted = True

        logger.info(
            "Loaded IsolationForest detector version %s from %s",
            payload.get("model_version", "unknown"),
            path,
        )
        return detector
