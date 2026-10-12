"""Unit tests for Data Validation Module."""

from __future__ import annotations

from src.processor.validation import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    validate_csv_content,
    validate_file_metadata,
)


def test_validate_file_metadata_valid_csv():
    is_valid, err = validate_file_metadata("incoming/data.csv", 1024)
    assert is_valid is True
    assert err is None


def test_validate_file_metadata_invalid_extension():
    is_valid, err = validate_file_metadata("incoming/data.xlsx", 1024)
    assert is_valid is False
    assert "Unsupported file extension" in err


def test_validate_file_metadata_zero_bytes():
    is_valid, err = validate_file_metadata("incoming/empty.csv", 0)
    assert is_valid is False
    assert "empty" in err.lower()


def test_validate_file_metadata_oversized():
    is_valid, err = validate_file_metadata("incoming/huge.csv", DEFAULT_MAX_FILE_SIZE_BYTES + 1)
    assert is_valid is False
    assert "exceeds maximum limit" in err


def test_validate_csv_content_clean_data():
    csv_text = (
        "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score\n"
        "TXN_1,50.25,2,10.5,0.15\n"
        "TXN_2,120.00,4,22.0,0.05\n"
    )
    result = validate_csv_content(csv_text)
    assert result.is_valid_file is True
    assert result.total_rows == 2
    assert result.valid_rows == 2
    assert result.invalid_rows == 0
    assert len(result.valid_df) == 2


def test_validate_csv_content_empty_data():
    result = validate_csv_content("")
    assert result.is_valid_file is False
    assert "empty" in result.error_message.lower()


def test_validate_csv_content_header_only():
    csv_text = "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score\n"
    result = validate_csv_content(csv_text)
    assert result.is_valid_file is False
    assert "zero data rows" in result.error_message.lower()


def test_validate_csv_content_missing_required_columns():
    csv_text = "transaction_id,amount,transaction_velocity\nTXN_1,10.0,1\n"
    result = validate_csv_content(csv_text)
    assert result.is_valid_file is False
    assert "missing mandatory required column" in result.error_message.lower()


def test_validate_csv_content_exceeds_max_rows():
    csv_text = "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score\n" + "\n".join(
        [f"TXN_{i},10.0,1,5.0,0.1" for i in range(15)]
    )
    result = validate_csv_content(csv_text, max_rows=10)
    assert result.is_valid_file is False
    assert "exceeds maximum permitted limit" in result.error_message.lower()


def test_validate_csv_content_segregates_corrupt_records():
    csv_text = (
        "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score\n"
        "TXN_VALID,50.0,2,10.0,0.2\n"
        "TXN_BAD_AMOUNT,-99.0,2,10.0,0.2\n"  # negative amount
        "TXN_BAD_RISK,40.0,1,5.0,1.5\n"  # risk score > 1.0
        "TXN_STR_VAL,NOT_A_NUM,1,5.0,0.2\n"  # unparseable string
        ",30.0,1,5.0,0.2\n"  # missing transaction_id
    )
    result = validate_csv_content(csv_text)
    assert result.is_valid_file is True
    assert result.total_rows == 5
    assert result.valid_rows == 1
    assert result.invalid_rows == 4
    assert len(result.invalid_records) == 4
    assert result.valid_df.iloc[0]["transaction_id"] == "TXN_VALID"


def test_validate_csv_content_all_records_invalid():
    csv_text = (
        "transaction_id,amount,transaction_velocity,distance_from_home,device_risk_score\n"
        "TXN_BAD_AMOUNT,-50.0,2,10.0,0.2\n"
    )
    result = validate_csv_content(csv_text)
    assert result.is_valid_file is False
    assert result.valid_rows == 0
    assert result.invalid_rows == 1
    assert "All records in the CSV file failed validation" in result.error_message
