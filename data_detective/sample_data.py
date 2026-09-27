"""
Data Detective Sample Datasets Generator.
Provides realistic business incident datasets for immediate out-of-the-box demonstration.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data")


def generate_ecommerce_incident_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates an e-commerce checkout incident dataset:
    On 2024-10-14, release 'v2.4' broke the iOS payment gateway in Europe,
    causing checkout conversion for iOS Europe to crash from ~4.8% to ~0.5%,
    with checkout latency surging from 320ms to 2400ms.
    """
    np.random.seed(seed)

    start_date = pd.Timestamp("2024-10-01")
    dates = [start_date + pd.Timedelta(days=int(d)) for d in np.random.uniform(0, 24, size=n_samples)]
    dates = sorted(dates)

    devices = np.random.choice(["iOS", "Android", "Desktop"], size=n_samples, p=[0.40, 0.35, 0.25])
    regions = np.random.choice(["Europe", "North America", "Asia-Pacific", "Latin America"], size=n_samples, p=[0.35, 0.35, 0.20, 0.10])
    channels = np.random.choice(["Organic Search", "Paid Search", "Direct", "Social Media"], size=n_samples, p=[0.35, 0.30, 0.20, 0.15])

    data = []
    incident_date = pd.Timestamp("2024-10-14")

    for i in range(n_samples):
        dt = dates[i]
        dev = devices[i]
        reg = regions[i]
        chan = channels[i]
        is_post_incident = (dt >= incident_date)

        version = "v2.4" if is_post_incident else "v2.3"

        # Baseline latency ~320ms
        latency = np.random.normal(320, 45)
        # Baseline conversion ~4.8%
        base_conv_prob = 0.048

        # Ground truth injection
        if is_post_incident and dev == "iOS" and reg == "Europe":
            # Broken payment gateway
            latency = np.random.normal(2450, 350)
            conv_prob = 0.005
        elif is_post_incident and dev == "iOS":
            # Slight ripple latency
            latency = np.random.normal(480, 80)
            conv_prob = 0.042
        else:
            conv_prob = base_conv_prob

        converted = 1 if np.random.rand() < conv_prob else 0
        cart_items = int(np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
        order_value = round(float(np.random.gamma(shape=3.0, scale=25.0)), 2) if converted else 0.0

        data.append({
            "session_id": f"SES-{100000 + i}",
            "date": dt.strftime("%Y-%m-%d"),
            "device": dev,
            "region": reg,
            "channel": chan,
            "app_version": version,
            "checkout_latency_ms": round(float(max(50.0, latency)), 1),
            "cart_items": cart_items,
            "converted": converted,
            "order_value": order_value
        })

    df = pd.DataFrame(data)
    return df


def generate_saas_churn_dataset(n_samples: int = 3500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a SaaS platform customer health and churn spike dataset:
    Enterprise customers experienced API performance degradation after a tier pricing change.
    """
    np.random.seed(seed)

    start_date = pd.Timestamp("2024-05-01")
    dates = [start_date + pd.Timedelta(days=int(d)) for d in np.random.uniform(0, 30, size=n_samples)]
    dates = sorted(dates)

    tiers = np.random.choice(["Starter", "Professional", "Enterprise"], size=n_samples, p=[0.55, 0.30, 0.15])
    industries = np.random.choice(["Fintech", "Healthcare", "E-Commerce", "SaaS"], size=n_samples, p=[0.30, 0.25, 0.25, 0.20])

    data = []
    spike_date = pd.Timestamp("2024-05-18")

    for i in range(n_samples):
        dt = dates[i]
        tier = tiers[i]
        ind = industries[i]
        is_spike = (dt >= spike_date)

        # Baseline churn probability ~3%
        churn_prob = 0.03
        api_latency = np.random.normal(85, 15)

        if is_spike and tier == "Enterprise":
            churn_prob = 0.22
            api_latency = np.random.normal(380, 60)
        elif is_spike:
            churn_prob = 0.045
            api_latency = np.random.normal(110, 25)

        churned = 1 if np.random.rand() < churn_prob else 0
        support_tickets = int(np.random.poisson(3 if churned else 1))
        mrr = round(float(np.random.normal(4500 if tier == "Enterprise" else (1200 if tier == "Professional" else 200), 50)), 2)

        data.append({
            "account_id": f"ACC-{20000 + i}",
            "date": dt.strftime("%Y-%m-%d"),
            "plan_tier": tier,
            "industry": ind,
            "api_latency_ms": round(float(max(20.0, api_latency)), 1),
            "support_tickets": support_tickets,
            "monthly_mrr": max(50.0, mrr),
            "churned": churned
        })

    return pd.DataFrame(data)


def ensure_sample_data_files() -> Tuple[Path, Path]:
    """
    Ensures sample CSV files exist on disk for immediate use.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ecom_path = DATA_DIR / "sample_ecommerce_incident.csv"
    saas_path = DATA_DIR / "sample_saas_churn.csv"

    if not ecom_path.exists():
        df_ecom = generate_ecommerce_incident_dataset()
        df_ecom.to_csv(ecom_path, index=False)

    if not saas_path.exists():
        df_saas = generate_saas_churn_dataset()
        df_saas.to_csv(saas_path, index=False)

    return ecom_path, saas_path


if __name__ == "__main__":
    p1, p2 = ensure_sample_data_files()
    print(f"Generated {p1} and {p2}")
