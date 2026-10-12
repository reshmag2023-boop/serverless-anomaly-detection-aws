"""AWS Lambda S3 Processor Handler.

Processes CSV files uploaded to Amazon S3:
1. Validates file metadata and content schema.
2. Checks idempotency/deduplication to avoid redundant reprocessing.
3. Scores records using Scikit-Learn Isolation Forest.
4. Stores detected anomalies and batch execution summary in Amazon DynamoDB.
5. Emits consolidated alert notifications via Amazon SNS when anomalies exceed thresholds.
6. Publishes custom Amazon CloudWatch metrics for pipeline monitoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from .anomaly_detector import IsolationForestDetector
from .validation import validate_csv_content, validate_file_metadata

# Structured logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

# AWS Clients (initialized outside handler for connection reuse across invocations)
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")
cloudwatch = boto3.client("cloudwatch")

# Environment configurations
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "AnomalyDetectionRecords")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ALERT_THRESHOLD = int(os.environ.get("ALERT_THRESHOLD", "1"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent / "models" / "isolation_forest_model.joblib"),
)
ALLOWED_PREFIX = os.environ.get("ALLOWED_PREFIX", "incoming/")
METRIC_NAMESPACE = "ServerlessAnomalyDetection"

# Cache loaded detector in global execution context across warm Lambda invocations
_CACHED_DETECTOR: Optional[IsolationForestDetector] = None


def get_detector() -> IsolationForestDetector:
    """Retrieve or initialize the Isolation Forest detector instance."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is not None:
        return _CACHED_DETECTOR

    model_file = Path(MODEL_PATH)
    if model_file.exists():
        try:
            logger.info("Loading pre-trained Isolation Forest model from: %s", model_file)
            _CACHED_DETECTOR = IsolationForestDetector.load(model_file)
            return _CACHED_DETECTOR
        except Exception as exc:
            logger.warning(
                "Failed to load model from %s (%s). Falling back to fresh default instance.",
                model_file,
                str(exc),
            )

    logger.info("Initializing baseline default Isolation Forest detector.")
    _CACHED_DETECTOR = IsolationForestDetector()
    return _CACHED_DETECTOR


def compute_batch_id(bucket: str, key: str, etag: str) -> str:
    """Generate deterministic batch ID for idempotency tracking."""
    clean_etag = etag.replace('"', "")
    raw = f"{bucket}/{key}/{clean_etag}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_already_processed(table: Any, batch_id: str) -> bool:
    """Check if batch was already processed in DynamoDB."""
    try:
        response = table.get_item(
            Key={"id": f"batch_{batch_id}"},
            ProjectionExpression="id, #st",
            ExpressionAttributeNames={"#st": "status"},
        )
        item = response.get("Item")
        if item and item.get("status") == "COMPLETED":
            return True
    except ClientError as exc:
        logger.warning("DynamoDB idempotency check failed: %s", exc)
    return False


def publish_cloudwatch_metrics(
    processed_count: int,
    anomaly_count: int,
    invalid_count: int,
    duration_ms: float,
) -> None:
    """Publish custom CloudWatch metrics for observability."""
    try:
        cloudwatch.put_metric_data(
            Namespace=METRIC_NAMESPACE,
            MetricData=[
                {
                    "MetricName": "ProcessedRecords",
                    "Value": processed_count,
                    "Unit": "Count",
                },
                {
                    "MetricName": "AnomaliesDetected",
                    "Value": anomaly_count,
                    "Unit": "Count",
                },
                {
                    "MetricName": "InvalidRecords",
                    "Value": invalid_count,
                    "Unit": "Count",
                },
                {
                    "MetricName": "ProcessingDurationMs",
                    "Value": duration_ms,
                    "Unit": "Milliseconds",
                },
            ],
        )
    except Exception as exc:
        logger.warning("Failed to publish CloudWatch metrics: %s", exc)


def send_sns_alert(
    bucket: str,
    key: str,
    batch_id: str,
    anomaly_count: int,
    sample_records: List[Dict[str, Any]],
    highest_severity: str,
) -> None:
    """Publish consolidated SNS notification when anomalies are detected."""
    if not SNS_TOPIC_ARN:
        logger.info("SNS_TOPIC_ARN is not configured; skipping notification.")
        return

    subject = f"[ALERT] {anomaly_count} Anomalies Detected in {key}"
    # Truncate subject if needed (AWS SNS max subject length is 100 characters)
    if len(subject) > 95:
        subject = subject[:92] + "..."

    payload = {
        "alert": "ANOMALY_DETECTION_TRIGGERED",
        "batch_id": batch_id,
        "s3_source": f"s3://{bucket}/{key}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_anomalies": anomaly_count,
        "highest_severity": highest_severity,
        "samples": sample_records[:5],
        "message": (
            f"Isolation Forest identified {anomaly_count} suspicious records in uploaded file "
            f"s3://{bucket}/{key}. Highest severity observed: {highest_severity}."
        ),
    }

    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=json.dumps(payload, indent=2),
        )
        logger.info("Published SNS anomaly alert to %s", SNS_TOPIC_ARN)
    except Exception as exc:
        logger.error("Failed to publish SNS alert: %s", exc)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for Amazon S3 ObjectCreated events."""
    start_time = time.time()
    logger.info("Processing S3 event with %d records.", len(event.get("Records", [])))

    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    processed_batches: List[Dict[str, Any]] = []

    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name")
        raw_key = s3_info.get("object", {}).get("key", "")
        key = urllib.parse.unquote_plus(raw_key)
        etag = s3_info.get("object", {}).get("eTag", "NO_ETAG")
        file_size = s3_info.get("object", {}).get("size", 0)

        if not bucket or not key:
            logger.warning("Skipping event record missing bucket or key: %s", record)
            continue

        # Prevent recursive triggers: only process allowed prefix and strictly .csv files
        if ALLOWED_PREFIX and not key.startswith(ALLOWED_PREFIX):
            logger.info("Skipping key '%s': does not match allowed prefix '%s'", key, ALLOWED_PREFIX)
            continue

        # Validate file metadata
        is_valid_meta, meta_err = validate_file_metadata(key, file_size)
        if not is_valid_meta:
            logger.warning("File metadata validation failed for %s: %s", key, meta_err)
            continue

        # Idempotency check
        batch_id = compute_batch_id(bucket, key, etag)
        if is_already_processed(table, batch_id):
            logger.info("Batch %s for s3://%s/%s already processed. Skipping.", batch_id, bucket, key)
            processed_batches.append({"batch_id": batch_id, "status": "ALREADY_PROCESSED", "key": key})
            continue

        logger.info("Starting processing for batch %s (s3://%s/%s)", batch_id, bucket, key)

        # Download S3 object
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            csv_bytes = response["Body"].read()
        except ClientError as exc:
            logger.error("Error retrieving object s3://%s/%s: %s", bucket, key, exc)
            continue

        # Validate CSV content
        validation_result = validate_csv_content(csv_bytes)
        now_iso = datetime.now(timezone.utc).isoformat()

        if not validation_result.is_valid_file:
            logger.error("Validation failed for %s: %s", key, validation_result.error_message)
            # Record failed batch summary in DynamoDB
            try:
                table.put_item(
                    Item={
                        "id": f"batch_{batch_id}",
                        "record_type": "BATCH_SUMMARY",
                        "created_at": now_iso,
                        "batch_id": batch_id,
                        "s3_bucket": bucket,
                        "s3_key": key,
                        "etag": etag,
                        "total_rows": validation_result.total_rows,
                        "valid_rows": 0,
                        "invalid_rows": validation_result.invalid_rows,
                        "anomalies_detected": 0,
                        "status": "VALIDATION_FAILED",
                        "error_message": validation_result.error_message or "Validation failed",
                    }
                )
            except Exception as ddb_err:
                logger.error("Failed to write validation failure to DynamoDB: %s", ddb_err)
            continue

        valid_df = validation_result.valid_df
        detector = get_detector()

        # If model is not yet fitted, fit on valid data if sufficient records exist
        if not detector.is_fitted:
            if len(valid_df) >= 20:
                logger.info("Fitting model on %d incoming records as fallback.", len(valid_df))
                detector.fit(valid_df)
            else:
                logger.warning("Insufficient rows (%d < 20) to fit fallback model.", len(valid_df))

        # Perform inference
        if detector.is_fitted:
            predictions_df = detector.predict(valid_df)
            anomalies_df = predictions_df[predictions_df["is_anomaly"]].copy()
        else:
            # Fallback when unfitted: treat none as anomaly to prevent crashing
            anomalies_df = valid_df.iloc[0:0].copy()

        anomaly_count = len(anomalies_df)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Batch %s: Processed %d records, detected %d anomalies in %.2f ms",
            batch_id,
            validation_result.valid_rows,
            anomaly_count,
            duration_ms,
        )

        # Batch write detected anomalies to DynamoDB
        highest_severity = "LOW"
        sample_records: List[Dict[str, Any]] = []

        if anomaly_count > 0:
            with table.batch_writer() as batch_writer:
                for idx, row in anomalies_df.iterrows():
                    severity = str(row.get("severity", "MEDIUM"))
                    if severity == "CRITICAL":
                        highest_severity = "CRITICAL"
                    elif severity == "HIGH" and highest_severity != "CRITICAL":
                        highest_severity = "HIGH"
                    elif severity == "MEDIUM" and highest_severity not in ("CRITICAL", "HIGH"):
                        highest_severity = "MEDIUM"

                    anomaly_id = f"anom_{batch_id}_{row['transaction_id']}"
                    item = {
                        "id": anomaly_id,
                        "record_type": "ANOMALY",
                        "created_at": now_iso,
                        "batch_id": batch_id,
                        "transaction_id": str(row["transaction_id"]),
                        "s3_source": f"s3://{bucket}/{key}",
                        "decision_score": Decimal(str(row["decision_score"])),
                        "anomaly_score": Decimal(str(row["anomaly_score"])),
                        "severity": severity,
                        "features": {col: Decimal(str(row[col])) for col in detector.feature_columns if col in row},
                        "original_data": {k: str(v) for k, v in row.to_dict().items() if not pd.isna(v)},
                    }
                    batch_writer.put_item(Item=item)

                    if len(sample_records) < 5:
                        sample_records.append(
                            {
                                "id": anomaly_id,
                                "transaction_id": str(row["transaction_id"]),
                                "decision_score": float(row["decision_score"]),
                                "anomaly_score": float(row["anomaly_score"]),
                                "severity": severity,
                            }
                        )

        # Store Batch Summary in DynamoDB
        anomaly_rate = Decimal(str(round(anomaly_count / max(validation_result.valid_rows, 1), 4)))
        batch_summary = {
            "id": f"batch_{batch_id}",
            "record_type": "BATCH_SUMMARY",
            "created_at": now_iso,
            "batch_id": batch_id,
            "s3_bucket": bucket,
            "s3_key": key,
            "etag": etag,
            "total_rows": validation_result.total_rows,
            "valid_rows": validation_result.valid_rows,
            "invalid_rows": validation_result.invalid_rows,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "processing_time_ms": Decimal(str(duration_ms)),
            "status": "COMPLETED",
        }
        try:
            table.put_item(Item=batch_summary)
        except Exception as exc:
            logger.error("Failed to write batch summary to DynamoDB: %s", exc)

        # Publish SNS Notification if threshold reached or critical anomaly found
        if anomaly_count >= ALERT_THRESHOLD or highest_severity == "CRITICAL":
            send_sns_alert(
                bucket=bucket,
                key=key,
                batch_id=batch_id,
                anomaly_count=anomaly_count,
                sample_records=sample_records,
                highest_severity=highest_severity,
            )

        # Publish CloudWatch custom metrics
        publish_cloudwatch_metrics(
            processed_count=validation_result.valid_rows,
            anomaly_count=anomaly_count,
            invalid_count=validation_result.invalid_rows,
            duration_ms=duration_ms,
        )

        processed_batches.append(
            {
                "batch_id": batch_id,
                "status": "COMPLETED",
                "valid_rows": validation_result.valid_rows,
                "invalid_rows": validation_result.invalid_rows,
                "anomalies_detected": anomaly_count,
            }
        )

    return {
        "statusCode": 200,
        "processed_batches": processed_batches,
    }
