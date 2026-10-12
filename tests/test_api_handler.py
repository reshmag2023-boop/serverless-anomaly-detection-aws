"""Integration tests for API Gateway Lambda Handler using Moto."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict

import boto3

from src.api.handler import lambda_handler


def create_api_event(
    http_method: str = "GET",
    path: str = "/health",
    path_parameters: Dict[str, str] = None,
    query_parameters: Dict[str, str] = None,
) -> Dict[str, Any]:
    return {
        "httpMethod": http_method,
        "path": path,
        "pathParameters": path_parameters,
        "queryStringParameters": query_parameters,
        "headers": {"Accept": "application/json"},
    }


def seed_sample_records(table_name: str) -> None:
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table(table_name)

    # Insert 2 anomaly records
    table.put_item(
        Item={
            "id": "anom_batch1_TXN_001",
            "record_type": "ANOMALY",
            "created_at": "2026-10-12T10:00:00Z",
            "batch_id": "batch1",
            "transaction_id": "TXN_001",
            "decision_score": Decimal("-0.18"),
            "anomaly_score": Decimal("0.85"),
            "severity": "CRITICAL",
        }
    )
    table.put_item(
        Item={
            "id": "anom_batch1_TXN_002",
            "record_type": "ANOMALY",
            "created_at": "2026-10-12T10:05:00Z",
            "batch_id": "batch1",
            "transaction_id": "TXN_002",
            "decision_score": Decimal("-0.08"),
            "anomaly_score": Decimal("0.69"),
            "severity": "HIGH",
        }
    )

    # Insert batch summary
    table.put_item(
        Item={
            "id": "batch_batch1",
            "record_type": "BATCH_SUMMARY",
            "created_at": "2026-10-12T10:06:00Z",
            "batch_id": "batch1",
            "valid_rows": 100,
            "invalid_rows": 2,
            "anomalies_detected": 2,
            "status": "COMPLETED",
        }
    )


def test_api_health_endpoint():
    event = create_api_event(http_method="GET", path="/health")
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert "Access-Control-Allow-Origin" in response["headers"]
    body = json.loads(response["body"])
    assert body["status"] == "HEALTHY"
    assert "version" in body


def test_api_get_anomalies_list(dynamodb_setup: str):
    seed_sample_records(dynamodb_setup)

    event = create_api_event(http_method="GET", path="/anomalies", query_parameters={"limit": "10"})
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["count"] == 2
    assert len(body["items"]) == 2
    # Verify Decimal conversion worked smoothly
    assert body["items"][0]["decision_score"] in (-0.18, -0.08)


def test_api_get_anomalies_filter_severity(dynamodb_setup: str):
    seed_sample_records(dynamodb_setup)

    event = create_api_event(http_method="GET", path="/anomalies", query_parameters={"severity": "CRITICAL"})
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["count"] == 1
    assert body["items"][0]["severity"] == "CRITICAL"


def test_api_get_anomalies_invalid_params():
    event_bad_limit = create_api_event(http_method="GET", path="/anomalies", query_parameters={"limit": "999"})
    res1 = lambda_handler(event_bad_limit, None)
    assert res1["statusCode"] == 400

    event_bad_sev = create_api_event(http_method="GET", path="/anomalies", query_parameters={"severity": "INVALID_SEV"})
    res2 = lambda_handler(event_bad_sev, None)
    assert res2["statusCode"] == 400


def test_api_get_anomaly_by_id_found(dynamodb_setup: str):
    seed_sample_records(dynamodb_setup)

    event = create_api_event(
        http_method="GET",
        path="/anomalies/anom_batch1_TXN_001",
        path_parameters={"id": "anom_batch1_TXN_001"},
    )
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["data"]["id"] == "anom_batch1_TXN_001"
    assert body["data"]["transaction_id"] == "TXN_001"


def test_api_get_anomaly_by_id_not_found(dynamodb_setup: str):
    seed_sample_records(dynamodb_setup)

    event = create_api_event(
        http_method="GET",
        path="/anomalies/non_existent_id",
        path_parameters={"id": "non_existent_id"},
    )
    response = lambda_handler(event, None)

    assert response["statusCode"] == 404
    body = json.loads(response["body"])
    assert "error" in body


def test_api_get_summary(dynamodb_setup: str):
    seed_sample_records(dynamodb_setup)

    event = create_api_event(http_method="GET", path="/summary")
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    summary = body["summary"]
    assert summary["recent_batches_evaluated"] == 1
    assert summary["total_valid_rows_evaluated"] == 100
    assert summary["total_anomalies_detected"] == 2
    assert summary["overall_anomaly_percentage"] == 2.0


def test_api_cors_options():
    event = create_api_event(http_method="OPTIONS", path="/anomalies")
    response = lambda_handler(event, None)
    assert response["statusCode"] == 200
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"


def test_api_invalid_method():
    event = create_api_event(http_method="POST", path="/anomalies")
    response = lambda_handler(event, None)
    assert response["statusCode"] == 405


def test_api_unknown_route():
    event = create_api_event(http_method="GET", path="/unknown_route")
    response = lambda_handler(event, None)
    assert response["statusCode"] == 404
