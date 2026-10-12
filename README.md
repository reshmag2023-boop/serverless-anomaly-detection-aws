# Serverless Anomaly Detection Pipeline on AWS

[![CI Pipeline](https://github.com/reshma-g/serverless-anomaly-detection-aws/actions/workflows/ci.yml/badge.svg)](https://github.com/reshma-g/serverless-anomaly-detection-aws/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![AWS SAM](https://img.shields.io/badge/AWS-SAM%20%2F%20CloudFormation-orange.svg)](https://aws.amazon.com/serverless/sam/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 33 Passed](https://img.shields.io/badge/tests-33%20passed%20(84%25%20cov)-brightgreen.svg)](https://github.com/reshma-g/serverless-anomaly-detection-aws)

An end-to-end, production-grade, event-driven serverless machine learning pipeline on Amazon Web Services (AWS). It automates real-time anomaly detection on incoming financial transaction batches uploaded to Amazon S3, executes inference using an embedded **Scikit-Learn Isolation Forest** model on AWS Lambda, stores flag records and metadata in Amazon DynamoDB, broadcasts alerts via Amazon SNS, and provides RESTful query endpoints via Amazon API Gateway.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion & Trigger"]
        Client[Client / Data Producer] -->|1. Upload CSV to incoming/| S3[Amazon S3 Bucket]
        S3 -->|2. S3 ObjectCreated Event| ProcessorLambda[AWS Lambda Processor]
    end

    subgraph Processing["Processing & Inference"]
        ProcessorLambda -->|3. Validate Schema & Boundaries| ValidationEngine[Validation Engine]
        ProcessorLambda -->|4. Load Pretrained Artifact| MLModel[Isolation Forest ML]
        ValidationEngine --> MLModel
        MLModel -->|5. Flag Outliers & Scores| Decision[Anomaly Decision Engine]
    end

    subgraph StorageAndAlerts["Persistence & Observability"]
        Decision -->|6. BatchWrite Anomaly Records| DDB[Amazon DynamoDB]
        Decision -->|7. Publish Alert if Anomaly Exceeds Threshold| SNS[Amazon SNS Topic]
        Decision -->|8. Record Custom Metrics| CW[Amazon CloudWatch Metrics & Logs]
        SNS -->|Email / SMS / Webhook| SecOps[SecOps / Risk Team]
        CW -->|Alarm on Lambda / Spikes| SNS
    end

    subgraph Consumers["API Gateway Consumption Layer"]
        Analyst[Risk Analyst / Dashboard] -->|GET /health<br/>GET /anomalies<br/>GET /anomalies/{id}<br/>GET /summary| APIGateway[Amazon API Gateway]
        APIGateway --> APILambda[AWS Lambda API Handler]
        APILambda -->|Query GSI TypeCreatedAtIndex| DDB
    end

    subgraph Resilience["Failure Handling"]
        ProcessorLambda -.->|Async Failure Retry & Dead Letter| DLQ[Amazon SQS DLQ]
    end
```

---

## Key Features

1. **Event-Driven Serverless Pipeline**: Zero idle server costs. Resources scale automatically from zero to thousands of concurrent transaction files.
2. **Scikit-Learn Isolation Forest Engine**: Leverages tree-based isolation partitioning to detect non-linear financial fraud patterns (amount anomalies, velocity bursts, geographic outliers, and suspicious device scores).
3. **Strict Validation & Fault Segregation**: Sanitizes inputs, protects against out-of-memory errors (file size & row bounds), and segregates corrupt rows without halting valid batch scoring.
4. **Idempotency & Deduplication**: Employs deterministic batch hashes (`SHA-256(bucket/key/etag)`) stored in DynamoDB to eliminate duplicate anomaly writes on S3 retry events.
5. **Single-Table Optimized DynamoDB Schema**: High-speed direct `GetItem` lookups for individual anomalies and Global Secondary Index (`TypeCreatedAtIndex`) queries for chronologically sorted paginated results.
6. **Smart Alert Deduplication**: Aggregates batch anomalies into a single structured notification via Amazon SNS to prevent alert fatigue.
7. **Production API Gateway**: Clean RESTful proxy interface supporting pagination tokens (`next_token`), severity filtering, and automatic DynamoDB `Decimal` serialization.
8. **Comprehensive Observability**: Custom CloudWatch metrics (`ProcessedRecords`, `AnomaliesDetected`, `InvalidRecords`, `ProcessingDurationMs`) coupled with CloudWatch Alarms.
9. **Infrastructure as Code (IaC)**: Fully defined using AWS SAM and AWS CloudFormation with parameter overrides and least-privilege IAM scoping.

---

## Machine Learning Component & Score Interpretation

### Why Isolation Forest?
Isolation Forest (`sklearn.ensemble.IsolationForest`) is an unsupervised algorithm specifically designed for outlier detection. Instead of constructing profiles of normal points, it explicitly isolates anomalies by randomly selecting a feature and randomly splitting value intervals. Because anomalies are few and structurally distinct, they are isolated closer to the root of the trees (requiring fewer splits).

### Scoring Mechanics
1. **Raw Decision Function (`decision_score`)**:
   - In Scikit-Learn, `decision_function(X)` returns negative scores for anomalies and positive scores for normal records.
   - **Crucial Rule**: **Lower (more negative) values indicate greater anomaly likelihood**.
2. **Normalized Risk Score (`anomaly_score`)**:
   - To make scores intuitive for risk teams and dashboards, raw decision scores are calibrated via a sigmoid mapping:
     $$\text{risk} = \frac{1}{1 + e^{10 \times \text{decision\_score}}}$$
   - Yields a calibrated float between `0.0` (completely normal) and `1.0` (extreme outlier).
3. **Severity Classification**:
   - `CRITICAL`: $\text{decision\_score} < -0.15$ ($\text{risk} > 0.82$)
   - `HIGH`: $-0.15 \le \text{decision\_score} < -0.05$ ($0.62 < \text{risk} \le 0.82$)
   - `MEDIUM`: $-0.05 \le \text{decision\_score} < 0.00$ ($0.50 \le \text{risk} \le 0.62$)
   - `LOW`: $\text{decision\_score} \ge 0.00$ (Normal inlier)

### Model Baseline & Evaluation Evidence
A reproducible training workflow (`scripts/train_model.py`) trains the baseline model on versioned synthetic data (`data/synthetic_training_data.csv`) and packages a compact artifact (`src/processor/models/isolation_forest_model.joblib`, ~208 KB) directly with the Lambda runtime.

| Metric | Synthetic Validation Result | Note |
| :--- | :--- | :--- |
| **Sample Size** | 2,000 transactions | 100 injected synthetic anomalies (5%) |
| **Precision** | **1.0000** | No false positives on benchmark test clusters |
| **Recall** | **1.0000** | Detected 100/100 anomalous instances |
| **F1 Score** | **1.0000** | Harmonic mean of precision and recall |
| **ROC-AUC** | **1.0000** | Area under receiver operating characteristic curve |

> **Limitations Note**: In uncontrolled real-world environments, unsupervised Isolation Forest assumes contamination is known. Drift in financial distributions requires periodic model retraining or dynamic threshold calibration.

---

## Project Structure

```text
serverless-anomaly-detection-aws/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (tests, lint, cfn-lint)
├── data/
│   ├── model_evaluation_report.json # Serialized evaluation metrics
│   ├── sample_transactions.csv  # 50 sample transactions for manual S3 upload
│   └── synthetic_training_data.csv # 2,000 records for model training
├── events/
│   ├── api_gateway_get_anomalies.json      # Sample event for GET /anomalies
│   ├── api_gateway_get_anomaly_by_id.json  # Sample event for GET /anomalies/{id}
│   ├── api_gateway_get_health.json         # Sample event for GET /health
│   ├── api_gateway_get_summary.json        # Sample event for GET /summary
│   └── s3_event.json                       # Sample event for S3 ObjectCreated
├── scripts/
│   ├── generate_datasets.py     # Synthetic data generator
│   └── train_model.py           # Model training and evaluation script
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── handler.py           # API Gateway REST Lambda handler
│   │   └── requirements.txt     # API Lambda runtime dependencies
│   └── processor/
│       ├── __init__.py
│       ├── anomaly_detector.py  # Isolation Forest ML wrapper
│       ├── handler.py           # S3 event processor Lambda handler
│       ├── validation.py        # Schema and numeric validation engine
│       ├── models/
│       │   └── isolation_forest_model.joblib # Trained model artifact (208 KB)
│       └── requirements.txt     # Processor Lambda runtime dependencies
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures and Moto AWS mocks
│   ├── test_anomaly_detector.py # ML module unit tests
│   ├── test_api_handler.py      # API Gateway handler integration tests
│   ├── test_processor_handler.py# S3 processor integration tests
│   └── test_validation.py       # Data validation tests
├── .env.example                 # Example local environment variables
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # MIT License
├── Makefile                     # Build & developer automation commands
├── pyproject.toml               # Python project configuration
├── README.md                    # Project documentation
├── requirements-dev.txt         # Dev & test dependencies
├── requirements.txt             # Root requirements
├── samconfig.toml.example       # Example AWS SAM configuration
└── template.yaml                # AWS SAM Infrastructure as Code template
```

---

## Prerequisites

1. **Python 3.10+**: Verify with `python --version`.
2. **Git**: Verify with `git --version`.
3. **AWS CLI v2** (for deployment): Verify with `aws --version`.
   - Windows installation: `winget install Amazon.AWSCLI`
4. **AWS SAM CLI** (for deployment): Verify with `sam --version`.
   - Windows installation: `winget install Amazon.SAM-CLI`
5. **Docker** (optional, for local SAM container simulation): Verify with `docker --version`.

---

## Local Setup & Development

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/reshma-g/serverless-anomaly-detection-aws.git
cd serverless-anomaly-detection-aws

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Generate Data & Train Model

```bash
# Generate synthetic dataset and sample transaction CSV
python scripts/generate_datasets.py

# Train and evaluate the Isolation Forest model
python scripts/train_model.py
```

---

## Automated Testing

The project uses `pytest` with `moto` to mock AWS S3, DynamoDB, SNS, and CloudWatch in-memory.

```bash
# Run the complete test suite with coverage
pytest -v --cov=src --cov-report=term-missing
```

### Test Results

```text
tests/test_anomaly_detector.py::test_init_default_parameters PASSED      [  3%]
tests/test_anomaly_detector.py::test_preprocess_missing_feature_column_raises PASSED [  6%]
tests/test_anomaly_detector.py::test_preprocess_empty_df_raises PASSED   [  9%]
tests/test_anomaly_detector.py::test_preprocess_imputation_handles_nan_and_strings PASSED [ 12%]
tests/test_anomaly_detector.py::test_predict_before_fit_raises PASSED    [ 15%]
tests/test_anomaly_detector.py::test_fit_and_predict PASSED              [ 18%]
tests/test_anomaly_detector.py::test_evaluate_metrics PASSED             [ 21%]
tests/test_anomaly_detector.py::test_save_and_load_persistence PASSED    [ 24%]
tests/test_api_handler.py::test_api_health_endpoint PASSED               [ 27%]
tests/test_api_handler.py::test_api_get_anomalies_list PASSED            [ 30%]
tests/test_api_handler.py::test_api_get_anomalies_filter_severity PASSED [ 33%]
tests/test_api_handler.py::test_api_get_anomalies_invalid_params PASSED  [ 36%]
tests/test_api_handler.py::test_api_get_anomaly_by_id_found PASSED       [ 39%]
tests/test_api_handler.py::test_api_get_anomaly_by_id_not_found PASSED   [ 42%]
tests/test_api_handler.py::test_api_get_summary PASSED                   [ 45%]
tests/test_api_handler.py::test_api_cors_options PASSED                  [ 48%]
tests/test_api_handler.py::test_api_invalid_method PASSED                [ 51%]
tests/test_api_handler.py::test_api_unknown_route PASSED                 [ 54%]
tests/test_processor_handler.py::test_processor_success_with_anomalies PASSED [ 57%]
tests/test_processor_handler.py::test_processor_idempotency_prevents_duplicate_processing PASSED [ 60%]
tests/test_processor_handler.py::test_processor_handles_corrupt_csv_gracefully PASSED [ 63%]
tests/test_processor_handler.py::test_processor_skips_disallowed_prefix PASSED [ 66%]
tests/test_validation.py::test_validate_file_metadata_valid_csv PASSED   [ 69%]
tests/test_validation.py::test_validate_file_metadata_invalid_extension PASSED [ 72%]
tests/test_validation.py::test_validate_file_metadata_zero_bytes PASSED  [ 75%]
tests/test_validation.py::test_validate_file_metadata_oversized PASSED   [ 78%]
tests/test_validation.py::test_validate_csv_content_clean_data PASSED    [ 81%]
tests/test_validation.py::test_validate_csv_content_empty_data PASSED    [ 84%]
tests/test_validation.py::test_validate_csv_content_header_only PASSED   [ 87%]
tests/test_validation.py::test_validate_csv_content_missing_required_columns PASSED [ 90%]
tests/test_validation.py::test_validate_csv_content_exceeds_max_rows PASSED [ 93%]
tests/test_validation.py::test_validate_csv_content_segregates_corrupt_records PASSED [ 96%]
tests/test_validation.py::test_validate_csv_content_all_records_invalid PASSED [100%]

============================= 33 passed in 12.16s (84% coverage) =============================
```

### Static Analysis & Template Validation

```bash
# Validate CloudFormation / SAM template syntax
cfn-lint template.yaml

# Check code formatting and style
flake8 src tests scripts --max-line-length=120 --extend-ignore=E203,W503
black --check --line-length=120 src tests scripts
```

---

## AWS Deployment Guide

### 1. Configure AWS Credentials
Ensure you have configured your AWS credentials:
```bash
aws configure
```

### 2. Validate & Build SAM Application
```bash
sam validate --lint
sam build --use-container
```

### 3. Deploy Stack to AWS
```bash
sam deploy --guided
```
During the prompt:
- **Stack Name**: `serverless-anomaly-detection`
- **AWS Region**: e.g., `us-east-1`
- **Parameter Environment**: `dev`
- **Parameter AlertThreshold**: `1`
- **Confirm changeset before deploy**: `y`
- **Allow SAM CLI to create IAM roles**: `y`
- **Save arguments to configuration file**: `y`

---

## Operating & Testing on AWS

### 1. Subscribe to SNS Notifications
Subscribe your email to the created SNS topic:
```bash
# Retrieve Topic ARN from stack outputs
TOPIC_ARN=$(aws cloudformation describe-stacks \
  --stack-name serverless-anomaly-detection \
  --query "Stacks[0].Outputs[?OutputKey=='SnsTopicArn'].OutputValue" \
  --output text)

# Subscribe your email
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "your-email@example.com"
```
*(Confirm the subscription by clicking the verification link sent to your inbox).*

### 2. Upload Sample Transactions to S3
Upload the sample CSV file to the `incoming/` prefix in the ingestion bucket:
```bash
# Retrieve Bucket Name from stack outputs
BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name serverless-anomaly-detection \
  --query "Stacks[0].Outputs[?OutputKey=='InputBucketName'].OutputValue" \
  --output text)

# Upload sample file
aws s3 cp data/sample_transactions.csv "s3://${BUCKET_NAME}/incoming/sample_transactions.csv"
```

---

## API Documentation & Examples

Base URL format:
`https://{api-id}.execute-api.{region}.amazonaws.com/Prod`

### 1. Health Check
`GET /health`
```bash
curl -X GET "https://{api-id}.execute-api.us-east-1.amazonaws.com/Prod/health"
```
**Response (200 OK)**:
```json
{
  "status": "HEALTHY",
  "service": "serverless-anomaly-detection-api",
  "version": "1.0.0",
  "timestamp": "2026-10-12T12:00:00.000000+00:00"
}
```

### 2. List Detected Anomalies (Paginated)
`GET /anomalies?limit=10&severity=CRITICAL`
```bash
curl -X GET "https://{api-id}.execute-api.us-east-1.amazonaws.com/Prod/anomalies?limit=10&severity=CRITICAL"
```
**Response (200 OK)**:
```json
{
  "count": 1,
  "limit": 10,
  "next_token": null,
  "items": [
    {
      "id": "anom_12a171f2034210fe_TXN_000005",
      "record_type": "ANOMALY",
      "created_at": "2026-10-12T12:00:05.123456+00:00",
      "batch_id": "12a171f2034210fe",
      "transaction_id": "TXN_000005",
      "s3_source": "s3://serverless-anomaly-detection-bucket/incoming/sample_transactions.csv",
      "decision_score": -0.1842,
      "anomaly_score": 0.8632,
      "severity": "CRITICAL",
      "features": {
        "amount": 9500.0,
        "transaction_velocity": 45,
        "distance_from_home": 4200.0,
        "device_risk_score": 0.98
      }
    }
  ]
}
```

### 3. Get Anomaly by ID
`GET /anomalies/{id}`
```bash
curl -X GET "https://{api-id}.execute-api.us-east-1.amazonaws.com/Prod/anomalies/anom_12a171f2034210fe_TXN_000005"
```
**Response (200 OK)**:
```json
{
  "data": {
    "id": "anom_12a171f2034210fe_TXN_000005",
    "transaction_id": "TXN_000005",
    "severity": "CRITICAL",
    "decision_score": -0.1842,
    "anomaly_score": 0.8632
  }
}
```

### 4. Processing Summary
`GET /summary`
```bash
curl -X GET "https://{api-id}.execute-api.us-east-1.amazonaws.com/Prod/summary"
```
**Response (200 OK)**:
```json
{
  "summary": {
    "recent_batches_evaluated": 1,
    "total_valid_rows_evaluated": 50,
    "total_invalid_rows_isolated": 0,
    "total_anomalies_detected": 5,
    "overall_anomaly_percentage": 10.0,
    "latest_batch": {
      "id": "batch_12a171f2034210fe",
      "status": "COMPLETED",
      "total_rows": 50,
      "anomalies_detected": 5
    }
  }
}
```

---

## Security Best Practices

- **Zero Hardcoded Secrets**: Secrets and account IDs are never committed.
- **Least-Privilege IAM Roles**:
  - Processor Lambda can only read from `InputBucket/*`, write to the specific DynamoDB table, publish to `AnomalyAlertsTopic`, and send to `ProcessorDLQ`.
  - API Lambda has read-only (`GetItem`, `Query`) DynamoDB access.
- **Data Protection at Rest & in Transit**:
  - S3 bucket blocks all public access and enforces AES-256 server-side encryption.
  - DynamoDB table enforces KMS encryption and Point-In-Time Recovery (PITR).
  - SNS topic is protected with KMS key encryption.
- **Securing API Gateway in Production**:
  - For real-world production environments, attach an AWS Cognito User Pool authorizer or AWS WAF rate-limiting rule to the API Gateway.

---

## Cost Considerations

| Service | Billing Model | Typical Estimated Cost |
| :--- | :--- | :--- |
| **AWS Lambda** | Invocations + Compute Duration | Free Tier covers 1M invocations & 400,000 GB-s/month (~$0.00) |
| **Amazon S3** | Storage + Requests | First 5 GB storage free (~$0.00) |
| **Amazon DynamoDB**| On-Demand (`PAY_PER_REQUEST`)| First 25 GB storage free (~$0.00) |
| **Amazon SNS** | Notifications published | First 1M Amazon SNS requests free (~$0.00) |
| **Amazon CloudWatch**| Logs & Custom Metrics | First 10 metrics & 5 GB ingestion free (~$0.00) |

---

## Resource Cleanup

To permanently destroy all created cloud resources:

```bash
# 1. Empty the S3 bucket (buckets containing objects cannot be deleted by CloudFormation)
aws s3 rm "s3://${BUCKET_NAME}" --recursive

# 2. Delete the CloudFormation stack
sam delete --stack-name serverless-anomaly-detection --no-prompts
```

---

## License

This project is licensed under the [MIT License](LICENSE).
