"""Data Validation Module for S3 CSV Ingestion.

Ensures strict schema adherence, sanitizes numeric values, bounds record sizes,
and safely segregates corrupted records so legitimate data can still be processed.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default limits to protect Lambda resources
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_ROWS = 50_000
ALLOWED_EXTENSIONS = {".csv"}

# Schema definition
REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",
    "transaction_velocity",
    "distance_from_home",
    "device_risk_score",
]

NUMERIC_COLUMNS = [
    "amount",
    "transaction_velocity",
    "distance_from_home",
    "device_risk_score",
]


@dataclass
class ValidationResult:
    """Structured result of dataset validation."""

    is_valid_file: bool
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    valid_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_records: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def validate_file_metadata(
    filename: str,
    file_size_bytes: int,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Tuple[bool, Optional[str]]:
    """Validate file extension and file size before downloading or parsing."""
    import os

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file extension '{ext}'. Only CSV files ({', '.join(ALLOWED_EXTENSIONS)}) are supported.",
        )

    if file_size_bytes == 0:
        return False, "File is empty (0 bytes)."

    if file_size_bytes > max_file_size_bytes:
        return (
            False,
            f"File size ({file_size_bytes} bytes) exceeds maximum limit of {max_file_size_bytes} bytes.",
        )

    return True, None


def validate_csv_content(
    csv_bytes_or_buffer: io.BytesIO | bytes | str,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ValidationResult:
    """Parse and validate CSV content.

    Performs:
    1. Parsing check (detects malformed/corrupt CSV structure).
    2. Empty file / empty row check.
    3. Row count limit enforcement.
    4. Required column presence verification.
    5. Per-row numeric feature sanitization:
       - Identifies unparseable strings, infinite numbers, or null transaction_ids.
       - Isolates corrupted records into `invalid_records` so valid rows can still be processed.

    Args:
        csv_bytes_or_buffer: Raw CSV data.
        required_columns: List of columns required to be present in CSV header.
        numeric_columns: List of columns that must contain valid numeric values.
        max_rows: Maximum allowed rows.

    Returns:
        ValidationResult object.
    """
    req_cols = required_columns or REQUIRED_COLUMNS
    num_cols = numeric_columns or NUMERIC_COLUMNS

    if isinstance(csv_bytes_or_buffer, bytes):
        buffer = io.BytesIO(csv_bytes_or_buffer)
    elif isinstance(csv_bytes_or_buffer, str):
        buffer = io.StringIO(csv_bytes_or_buffer)
    else:
        buffer = csv_bytes_or_buffer

    # Step 1: Parse CSV
    try:
        df = pd.read_csv(buffer, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        return ValidationResult(
            is_valid_file=False,
            error_message="CSV file is empty or contains no readable data.",
        )
    except Exception as exc:
        return ValidationResult(
            is_valid_file=False,
            error_message=f"Failed to parse CSV file: {str(exc)}",
        )

    # Step 2: Empty DataFrame check
    if df.empty or len(df) == 0:
        return ValidationResult(
            is_valid_file=False,
            error_message="CSV contains header only or zero data rows.",
        )

    # Step 3: Max rows check
    total_rows = len(df)
    if total_rows > max_rows:
        return ValidationResult(
            is_valid_file=False,
            total_rows=total_rows,
            error_message=f"CSV row count ({total_rows}) exceeds maximum permitted limit of {max_rows} rows.",
        )

    # Normalize column names (strip whitespace, lowercase check if needed)
    df.columns = [c.strip() for c in df.columns]

    # Step 4: Required columns check
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        return ValidationResult(
            is_valid_file=False,
            total_rows=total_rows,
            error_message=f"CSV missing mandatory required column(s): {missing_cols}. Found: {list(df.columns)}",
        )

    warnings: List[str] = []
    extra_cols = [c for c in df.columns if c not in req_cols]
    if extra_cols:
        warnings.append(f"CSV contains unmodeled columns that will be ignored: {extra_cols}")

    # Step 5: Row-level validation and sanitization
    valid_indices: List[int] = []
    invalid_records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_errors: List[str] = []

        # Check transaction_id exists and is not blank
        tid = row_dict.get("transaction_id")
        if tid is None or pd.isna(tid) or str(tid).strip() == "":
            row_errors.append("Missing or blank transaction_id")

        # Validate numeric features
        for col in num_cols:
            val = row_dict.get(col)
            if val is None or pd.isna(val) or str(val).strip() == "":
                row_errors.append(f"Null or missing value in numeric column '{col}'")
                continue

            try:
                numeric_val = float(val)
                if np.isnan(numeric_val) or np.isinf(numeric_val):
                    row_errors.append(f"Invalid numeric value '{val}' (NaN or Inf) in '{col}'")
                elif col == "amount" and numeric_val < 0:
                    row_errors.append(f"Negative transaction amount ({numeric_val}) is invalid")
                elif col == "device_risk_score" and not (0.0 <= numeric_val <= 1.0):
                    row_errors.append(f"device_risk_score ({numeric_val}) must be in range [0.0, 1.0]")
            except (ValueError, TypeError):
                row_errors.append(f"Cannot parse value '{val}' as numeric float in '{col}'")

        if row_errors:
            invalid_records.append(
                {
                    "row_index": int(idx),
                    "transaction_id": str(row_dict.get("transaction_id", f"UNKNOWN_ROW_{idx}")),
                    "errors": row_errors,
                    "raw_data": row_dict,
                }
            )
        else:
            valid_indices.append(idx)

    valid_rows_count = len(valid_indices)
    invalid_rows_count = len(invalid_records)

    if valid_rows_count == 0:
        return ValidationResult(
            is_valid_file=False,
            total_rows=total_rows,
            valid_rows=0,
            invalid_rows=invalid_rows_count,
            invalid_records=invalid_records,
            error_message="All records in the CSV file failed validation; no valid rows found.",
            warnings=warnings,
        )

    # Construct clean valid dataframe with proper numeric types
    valid_df = df.loc[valid_indices].copy()
    for col in num_cols:
        valid_df[col] = valid_df[col].astype(float)

    return ValidationResult(
        is_valid_file=True,
        total_rows=total_rows,
        valid_rows=valid_rows_count,
        invalid_rows=invalid_rows_count,
        valid_df=valid_df,
        invalid_records=invalid_records,
        warnings=warnings,
    )
