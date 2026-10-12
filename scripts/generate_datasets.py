"""Synthetic Dataset Generator for Serverless Anomaly Detection Pipeline.

Generates reproducible synthetic transaction datasets containing known normal patterns
and injected anomaly clusters (unusual amounts, extreme velocity, geographic anomalies,
and high device risk scores).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_transactions(
    n_samples: int = 2000,
    anomaly_ratio: float = 0.05,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic financial transaction dataset with controlled anomalies.

    Args:
        n_samples: Total number of records to generate.
        anomaly_ratio: Fraction of records to inject as anomalies (default: 0.05 = 5%).
        random_seed: Seed for reproducible generation.

    Returns:
        DataFrame with transaction features and ground-truth 'is_anomaly' column.
    """
    rng = np.random.default_rng(random_seed)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Base timestamps spanning last 7 days
    base_time = datetime.now(timezone.utc) - timedelta(days=7)

    # 1. Normal Transactions
    # - Amount: Log-normal distribution around $45 (mean ~ 3.8, std ~ 0.7) -> $10 to $250
    normal_amounts = np.round(rng.lognormal(mean=3.8, sigma=0.7, size=n_normal), 2)
    normal_amounts = np.clip(normal_amounts, 2.50, 450.00)

    # - Transaction Velocity (txns in last hour): Poisson with lambda=2 -> mostly 1 to 4
    normal_velocity = rng.poisson(lam=2.0, size=n_normal) + 1

    # - Distance from home (miles): Gamma distribution -> mostly 1 to 30 miles
    normal_distance = np.round(rng.gamma(shape=2.0, scale=6.0, size=n_normal), 2)
    normal_distance = np.clip(normal_distance, 0.1, 75.0)

    # - Device Risk Score: Beta distribution skewed low -> 0.01 to 0.25
    normal_device_risk = np.round(rng.beta(a=1.5, b=8.0, size=n_normal), 3)

    normal_df = pd.DataFrame(
        {
            "amount": normal_amounts,
            "transaction_velocity": normal_velocity,
            "distance_from_home": normal_distance,
            "device_risk_score": normal_device_risk,
            "is_anomaly": 0,
        }
    )

    # 2. Anomalous Transactions (Simulating fraudulent patterns)
    # Type A: High-value sudden spikes ($1,500 - $12,000)
    # Type B: Velocity spikes (card testing / automated bot: 15 - 50 txns/hr)
    # Type C: Extreme impossible travel (1,200 - 8,000 miles)
    # Type D: Compromised device risk + high velocity
    anomaly_choices = ["amount_spike", "velocity_burst", "geo_anomaly", "compromised_device"]
    anomaly_types = rng.choice(anomaly_choices, size=n_anomalies)

    anom_amounts = []
    anom_velocity = []
    anom_distance = []
    anom_device_risk = []

    for a_type in anomaly_types:
        if a_type == "amount_spike":
            anom_amounts.append(round(rng.uniform(2500.0, 15000.0), 2))
            anom_velocity.append(rng.integers(1, 6))
            anom_distance.append(round(rng.uniform(5.0, 120.0), 2))
            anom_device_risk.append(round(rng.uniform(0.3, 0.8), 3))
        elif a_type == "velocity_burst":
            anom_amounts.append(round(rng.uniform(15.0, 200.0), 2))
            anom_velocity.append(int(rng.integers(15, 60)))
            anom_distance.append(round(rng.uniform(1.0, 30.0), 2))
            anom_device_risk.append(round(rng.uniform(0.4, 0.95), 3))
        elif a_type == "geo_anomaly":
            anom_amounts.append(round(rng.uniform(150.0, 3500.0), 2))
            anom_velocity.append(int(rng.integers(2, 8)))
            anom_distance.append(round(rng.uniform(1500.0, 9500.0), 2))
            anom_device_risk.append(round(rng.uniform(0.6, 0.98), 3))
        else:  # compromised_device
            anom_amounts.append(round(rng.uniform(400.0, 5000.0), 2))
            anom_velocity.append(int(rng.integers(8, 25)))
            anom_distance.append(round(rng.uniform(50.0, 500.0), 2))
            anom_device_risk.append(round(rng.uniform(0.85, 0.999), 3))

    anom_df = pd.DataFrame(
        {
            "amount": anom_amounts,
            "transaction_velocity": anom_velocity,
            "distance_from_home": anom_distance,
            "device_risk_score": anom_device_risk,
            "is_anomaly": 1,
        }
    )

    # Combine and shuffle
    df = pd.concat([normal_df, anom_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Generate metadata columns
    tx_ids = [f"TXN_{i+1:06d}" for i in range(len(df))]
    user_ids = [f"USR_{rng.integers(1000, 9999)}" for _ in range(len(df))]
    random_seconds = rng.integers(0, 7 * 86400, size=len(df))
    timestamps = [(base_time + timedelta(seconds=int(s))).isoformat() for s in random_seconds]

    df.insert(0, "transaction_id", tx_ids)
    df.insert(1, "user_id", user_ids)
    df.insert(2, "timestamp", timestamps)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transaction datasets.")
    parser.add_argument("--output-dir", default="data", help="Target output directory")
    parser.add_argument("--training-size", type=int, default=2000, help="Training set row count")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample test set row count")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic training dataset ({args.training_size} records)...")
    training_df = generate_synthetic_transactions(n_samples=args.training_size, anomaly_ratio=0.05, random_seed=42)
    training_path = out_dir / "synthetic_training_data.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Saved: {training_path} ({len(training_df)} records, {training_df['is_anomaly'].sum()} anomalies)")

    print(f"Generating realistic sample transaction file ({args.sample_size} records)...")
    sample_df = generate_synthetic_transactions(n_samples=args.sample_size, anomaly_ratio=0.10, random_seed=123)
    # Remove ground-truth is_anomaly for realistic incoming S3 payload simulation
    clean_sample_df = sample_df.drop(columns=["is_anomaly"])
    sample_path = out_dir / "sample_transactions.csv"
    clean_sample_df.to_csv(sample_path, index=False)
    print(f"Saved: {sample_path} ({len(clean_sample_df)} records)")


if __name__ == "__main__":
    main()
