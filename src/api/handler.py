"""AWS Lambda API Gateway REST Handler.

Exposes REST endpoints to query detected anomalies, batch processing summaries,
and service health status.

Endpoints:
- GET /health
- GET /anomalies
- GET /anomalies/{id}
- GET /summary
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "AnomalyDetectionRecords")
INDEX_NAME = os.environ.get("INDEX_NAME", "TypeCreatedAtIndex")
SERVICE_VERSION = "1.0.0"

dynamodb = boto3.resource("dynamodb")


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle DynamoDB Decimal types cleanly."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            if obj % 1 == 0:
                return int(obj)
            return float(obj)
        return super().default(obj)


def build_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Format standardized API Gateway proxy response with CORS headers."""
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
    }
    if headers:
        default_headers.update(headers)

    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body, cls=DecimalEncoder),
    }


def encode_next_token(last_evaluated_key: Optional[Dict[str, Any]]) -> Optional[str]:
    """Base64 encode DynamoDB LastEvaluatedKey for safe URL pagination."""
    if not last_evaluated_key:
        return None
    raw_json = json.dumps(last_evaluated_key, cls=DecimalEncoder)
    return base64.urlsafe_b64encode(raw_json.encode("utf-8")).decode("utf-8")


def decode_next_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Decode base64 pagination token into DynamoDB ExclusiveStartKey dict."""
    if not token:
        return None
    try:
        decoded_bytes = base64.urlsafe_b64decode(token.encode("utf-8"))
        data = json.loads(decoded_bytes.decode("utf-8"))
        # Re-convert numeric values to Decimal where needed
        return data
    except Exception as exc:
        logger.warning("Invalid pagination token: %s", exc)
        return None


def handle_health() -> Dict[str, Any]:
    """Handle GET /health."""
    return build_response(
        200,
        {
            "status": "HEALTHY",
            "service": "serverless-anomaly-detection-api",
            "version": SERVICE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def handle_get_anomaly_by_id(table: Any, anomaly_id: str) -> Dict[str, Any]:
    """Handle GET /anomalies/{id} via direct O(1) GetItem."""
    if not anomaly_id or anomaly_id.strip() == "":
        return build_response(400, {"error": "Anomaly ID must not be empty."})

    try:
        response = table.get_item(Key={"id": anomaly_id})
        item = response.get("Item")
        if not item:
            return build_response(
                404,
                {
                    "error": "Anomaly record not found",
                    "id": anomaly_id,
                },
            )
        return build_response(200, {"data": item})
    except ClientError as exc:
        logger.error("Error fetching anomaly %s: %s", anomaly_id, exc)
        return build_response(500, {"error": "Failed to retrieve anomaly record."})


def handle_get_anomalies(table: Any, query_params: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Handle GET /anomalies with pagination and optional severity filtering."""
    params = query_params or {}

    # Validate and bound limit parameter
    raw_limit = params.get("limit", "20")
    try:
        limit = int(raw_limit)
        if limit < 1 or limit > 100:
            return build_response(
                400,
                {"error": "Query parameter 'limit' must be an integer between 1 and 100."},
            )
    except ValueError:
        return build_response(400, {"error": "Query parameter 'limit' must be an integer."})

    severity_filter = params.get("severity")
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    if severity_filter and severity_filter.upper() not in valid_severities:
        return build_response(
            400,
            {"error": f"Invalid severity '{severity_filter}'. Must be one of: {sorted(valid_severities)}"},
        )

    next_token = params.get("next_token")
    exclusive_start_key = decode_next_token(next_token)

    query_kwargs: Dict[str, Any] = {
        "IndexName": INDEX_NAME,
        "KeyConditionExpression": Key("record_type").eq("ANOMALY"),
        "ScanIndexForward": False,  # Reverse chronological order
        "Limit": limit,
    }

    if exclusive_start_key:
        query_kwargs["ExclusiveStartKey"] = exclusive_start_key

    if severity_filter:
        from boto3.dynamodb.conditions import Attr

        query_kwargs["FilterExpression"] = Attr("severity").eq(severity_filter.upper())

    try:
        response = table.query(**query_kwargs)
        items = response.get("Items", [])
        last_key = response.get("LastEvaluatedKey")

        return build_response(
            200,
            {
                "count": len(items),
                "limit": limit,
                "next_token": encode_next_token(last_key),
                "items": items,
            },
        )
    except ClientError as exc:
        logger.error("Error querying anomalies: %s", exc)
        return build_response(500, {"error": "Failed to query anomaly records."})


def handle_get_summary(table: Any) -> Dict[str, Any]:
    """Handle GET /summary to return aggregate processing metrics across recent batches."""
    try:
        response = table.query(
            IndexName=INDEX_NAME,
            KeyConditionExpression=Key("record_type").eq("BATCH_SUMMARY"),
            ScanIndexForward=False,
            Limit=20,
        )
        batches = response.get("Items", [])

        total_batches = len(batches)
        total_rows_evaluated = 0
        total_anomalies_detected = 0
        total_invalid_rows = 0

        for b in batches:
            total_rows_evaluated += int(b.get("valid_rows", 0))
            total_anomalies_detected += int(b.get("anomalies_detected", 0))
            total_invalid_rows += int(b.get("invalid_rows", 0))

        overall_anomaly_rate = (
            round((total_anomalies_detected / total_rows_evaluated) * 100, 2) if total_rows_evaluated > 0 else 0.0
        )

        return build_response(
            200,
            {
                "summary": {
                    "recent_batches_evaluated": total_batches,
                    "total_valid_rows_evaluated": total_rows_evaluated,
                    "total_invalid_rows_isolated": total_invalid_rows,
                    "total_anomalies_detected": total_anomalies_detected,
                    "overall_anomaly_percentage": overall_anomaly_rate,
                    "latest_batch": batches[0] if batches else None,
                },
                "recent_batches": batches,
            },
        )
    except ClientError as exc:
        logger.error("Error querying summary: %s", exc)
        return build_response(500, {"error": "Failed to retrieve summary metrics."})


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point for Amazon API Gateway REST API requests."""
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    path_parameters = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}

    logger.info("API request: %s %s", http_method, path)

    # CORS Preflight
    if http_method == "OPTIONS":
        return build_response(200, {"message": "OK"})

    if http_method != "GET":
        return build_response(405, {"error": f"Method {http_method} not allowed."})

    table = dynamodb.Table(DYNAMODB_TABLE_NAME)

    # Clean path routing
    normalized_path = path.rstrip("/")

    if normalized_path in ("/health", ""):
        return handle_health()

    if normalized_path == "/summary":
        return handle_get_summary(table)

    if normalized_path == "/anomalies":
        return handle_get_anomalies(table, query_params)

    # Match /anomalies/{id}
    if path_parameters.get("id"):
        return handle_get_anomaly_by_id(table, path_parameters["id"])

    if normalized_path.startswith("/anomalies/"):
        anomaly_id = normalized_path.split("/")[-1]
        return handle_get_anomaly_by_id(table, anomaly_id)

    return build_response(
        404,
        {
            "error": "Resource not found",
            "path": path,
            "supported_endpoints": [
                "GET /health",
                "GET /anomalies",
                "GET /anomalies/{id}",
                "GET /summary",
            ],
        },
    )
