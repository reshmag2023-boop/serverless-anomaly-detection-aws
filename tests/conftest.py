"""Pytest fixtures and configuration for AWS mocks and test data."""

from __future__ import annotations

import os
from typing import Generator

import boto3
import pytest
from moto import mock_aws

# Set dummy AWS credentials before any boto3 import
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["DYNAMODB_TABLE_NAME"] = "TestAnomalyRecords"
os.environ["INDEX_NAME"] = "TypeCreatedAtIndex"
os.environ["ALERT_THRESHOLD"] = "1"
os.environ["ALLOWED_PREFIX"] = "incoming/"


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def s3_setup(aws_credentials) -> Generator[str, None, None]:
    """Provide mocked S3 bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-input-bucket"
        client.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.fixture(scope="function")
def dynamodb_setup(aws_credentials) -> Generator[str, None, None]:
    """Provide mocked DynamoDB table with GSI."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "TestAnomalyRecords"
        client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "record_type", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "TypeCreatedAtIndex",
                    "KeySchema": [
                        {"AttributeName": "record_type", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield table_name


@pytest.fixture(scope="function")
def sns_setup(aws_credentials) -> Generator[str, None, None]:
    """Provide mocked SNS Topic."""
    with mock_aws():
        client = boto3.client("sns", region_name="us-east-1")
        response = client.create_topic(Name="test-anomaly-alerts")
        topic_arn = response["TopicArn"]
        os.environ["SNS_TOPIC_ARN"] = topic_arn
        yield topic_arn


@pytest.fixture(scope="function")
def cloudwatch_setup(aws_credentials) -> Generator[None, None, None]:
    """Provide mocked CloudWatch."""
    with mock_aws():
        yield


@pytest.fixture(scope="function")
def sample_valid_csv_bytes() -> bytes:
    """Sample valid CSV content with 20 records (including a known outlier)."""
    rows = [
        "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score",
    ]
    # 19 Normal records
    for i in range(1, 20):
        rows.append(f"TXN_{i:04d},{35.50 + i},{2},{12.5},{0.12}")
    # 1 Extreme anomaly
    rows.append("TXN_0020,9999.00,45,4500.0,0.98")
    return "\n".join(rows).encode("utf-8")
