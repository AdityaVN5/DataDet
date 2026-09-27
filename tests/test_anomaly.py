import pandas as pd
import numpy as np
from data_detective.anomaly_detector import detect_rolling_anomalies, detect_isolation_forest_anomalies


def test_detect_rolling_anomalies():
    # 20 baseline days with metric ~100, then 5 anomaly days with metric ~20
    dates = pd.date_range("2024-01-01", periods=25)
    metrics = [100.0 + np.random.normal(0, 2) for _ in range(20)] + [20.0 + np.random.normal(0, 2) for _ in range(5)]
    df = pd.DataFrame({"date": dates, "conversion": metrics})

    points, window = detect_rolling_anomalies(df, metric_col="conversion", date_col="date")
    assert window is not None
    assert len(window.flagged_points) > 0
    assert window.direction == "drop"
    assert window.relative_pct_change < -50.0


def test_detect_isolation_forest_anomalies():
    np.random.seed(42)
    # Generate 200 normal points and 10 extreme outliers
    x1 = np.random.normal(50, 5, 200)
    x2 = np.random.normal(10, 2, 200)

    # Inliers
    df = pd.DataFrame({"metric": x1, "latency": x2})
    # Outliers
    outliers = pd.DataFrame({"metric": [10.0, 5.0, 12.0], "latency": [150.0, 180.0, 200.0]})
    df = pd.concat([df, outliers], ignore_index=True)

    points, window = detect_isolation_forest_anomalies(df, metric_col="metric", contamination=0.05)
    assert window is not None
    assert len(window.anomaly_indices) > 0
