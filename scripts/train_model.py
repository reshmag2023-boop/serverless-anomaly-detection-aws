"""Model Training & Evaluation Script.

Trains a Scikit-Learn Isolation Forest model on versioned synthetic training data,
evaluates precision, recall, and ROC-AUC on labeled synthetic anomalies,
and serializes the production artifact for bundling with the AWS Lambda processor.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from scripts.generate_datasets import generate_synthetic_transactions  # noqa: E402
from src.processor.anomaly_detector import DEFAULT_FEATURE_COLUMNS, IsolationForestDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate(
    data_path: Path,
    model_output_path: Path,
    report_output_path: Path,
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Train Isolation Forest and evaluate on labeled data."""
    if not data_path.exists():
        logger.info("Training dataset %s not found. Generating default synthetic dataset...", data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_transactions(n_samples=2000, anomaly_ratio=0.05, random_seed=42)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    logger.info("Loaded dataset with %d records from %s", len(df), data_path)

    # Initialize and fit detector
    detector = IsolationForestDetector(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    detector.fit(df)

    # Evaluate if ground truth column is present
    eval_report = {}
    if "is_anomaly" in df.columns:
        eval_report = detector.evaluate(df, ground_truth_col="is_anomaly")
        logger.info("=== Model Evaluation Report ===")
        logger.info("Total Samples: %d", eval_report["sample_count"])
        logger.info("Ground Truth Anomalies: %d", eval_report["ground_truth_anomalies"])
        logger.info("Detected Anomalies: %d", eval_report["detected_anomalies"])
        logger.info("Precision: %.4f", eval_report["precision"])
        logger.info("Recall: %.4f", eval_report["recall"])
        logger.info("F1 Score: %.4f", eval_report["f1_score"])
        logger.info("ROC-AUC: %.4f", eval_report["roc_auc"])
        logger.info("Confusion Matrix: %s", eval_report["confusion_matrix"])
    else:
        logger.warning("No ground-truth 'is_anomaly' column present; skipping evaluation metrics.")

    # Save model artifact
    saved_path = detector.save(model_output_path)
    logger.info("Successfully serialized model artifact to: %s", saved_path)

    # Save evaluation report
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved evaluation report to: %s", report_output_path)

    return eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest Anomaly Detection Model.")
    parser.add_argument(
        "--data",
        default="data/synthetic_training_data.csv",
        help="Path to synthetic training CSV",
    )
    parser.add_argument(
        "--output",
        default="src/processor/models/isolation_forest_model.joblib",
        help="Path to output serialized joblib model",
    )
    parser.add_argument(
        "--report",
        default="data/model_evaluation_report.json",
        help="Path to output evaluation report JSON",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected contamination rate (0.01 to 0.5)",
    )
    args = parser.parse_args()

    train_and_evaluate(
        data_path=Path(args.data),
        model_output_path=Path(args.output),
        report_output_path=Path(args.report),
        contamination=args.contamination,
    )


if __name__ == "__main__":
    main()
