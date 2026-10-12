"""Integration tests for S3 Processor Lambda Handler using Moto."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import boto3

from src.processor.handler import lambda_handler


def create_s3_event(bucket: str, key: str, etag: str = "test-etag", size: int = 1024) -> Dict[str, Any]:
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key, "eTag": etag, "size": size},
                },
            }
        ]
    }


def test_processor_success_with_anomalies(
    s3_setup: str,
    dynamodb_setup: str,
    sns_setup: str,
    cloudwatch_setup: None,
    sample_valid_csv_bytes: bytes,
):
    bucket = s3_setup
    key = "incoming/transactions_batch_01.csv"

    # Upload sample CSV to mocked S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket=bucket, Key=key, Body=sample_valid_csv_bytes)

    event = create_s3_event(bucket, key, etag="etag-123", size=len(sample_valid_csv_bytes))
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert len(response["processed_batches"]) == 1
    batch_info = response["processed_batches"][0]
    assert batch_info["status"] == "COMPLETED"
    assert batch_info["valid_rows"] == 20
    assert batch_info["anomalies_detected"] >= 1

    # Verify DynamoDB records
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table(dynamodb_setup)

    # 1. Check Batch Summary
    batch_id = batch_info["batch_id"]
    batch_res = table.get_item(Key={"id": f"batch_{batch_id}"})
    assert "Item" in batch_res
    item = batch_res["Item"]
    assert item["record_type"] == "BATCH_SUMMARY"
    assert item["status"] == "COMPLETED"
    assert item["valid_rows"] == 20
    assert item["total_rows"] == 20

    # 2. Check Anomaly item (TXN_0020)
    anom_res = table.get_item(Key={"id": f"anom_{batch_id}_TXN_0020"})
    assert "Item" in anom_res
    anom_item = anom_res["Item"]
    assert anom_item["record_type"] == "ANOMALY"
    assert anom_item["transaction_id"] == "TXN_0020"
    assert float(anom_item["decision_score"]) < 0.0  # Anomaly score is negative
    assert isinstance(anom_item["anomaly_score"], Decimal)


def test_processor_idempotency_prevents_duplicate_processing(
    s3_setup: str,
    dynamodb_setup: str,
    sns_setup: str,
    cloudwatch_setup: None,
    sample_valid_csv_bytes: bytes,
):
    bucket = s3_setup
    key = "incoming/duplicate_batch.csv"

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket=bucket, Key=key, Body=sample_valid_csv_bytes)

    event = create_s3_event(bucket, key, etag="etag-same", size=len(sample_valid_csv_bytes))

    # First invocation
    res1 = lambda_handler(event, None)
    assert res1["processed_batches"][0]["status"] == "COMPLETED"

    # Second invocation with same etag/key
    res2 = lambda_handler(event, None)
    assert res2["processed_batches"][0]["status"] == "ALREADY_PROCESSED"


def test_processor_handles_corrupt_csv_gracefully(
    s3_setup: str,
    dynamodb_setup: str,
    sns_setup: str,
    cloudwatch_setup: None,
):
    bucket = s3_setup
    key = "incoming/corrupted.csv"
    bad_bytes = b"header1,header2\nmissing,cols\n"

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket=bucket, Key=key, Body=bad_bytes)

    event = create_s3_event(bucket, key, etag="etag-bad", size=len(bad_bytes))
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    # Confirm error was recorded in DynamoDB batch summary without crashing Lambda
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table(dynamodb_setup)

    scan_res = table.scan()
    items = scan_res.get("Items", [])
    assert len(items) == 1
    assert items[0]["status"] == "VALIDATION_FAILED"


def test_processor_skips_disallowed_prefix(
    s3_setup: str,
    dynamodb_setup: str,
    sns_setup: str,
    cloudwatch_setup: None,
):
    bucket = s3_setup
    key = "processed/output.csv"
    data = b"some,data\n"

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket=bucket, Key=key, Body=data)

    event = create_s3_event(bucket, key, etag="etag-out", size=len(data))
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert len(response["processed_batches"]) == 0
