import pandas as pd
import numpy as np
from data_detective.profiler import infer_column_type, detect_outliers_iqr, profile_dataset, clean_and_prepare_dataset
from data_detective.schemas import DataType


def test_infer_column_type():
    num_series = pd.Series([1.2, 3.4, 5.6, 7.8, 9.0])
    assert infer_column_type(num_series, "revenue") == DataType.NUMERICAL

    cat_series = pd.Series(["US", "EU", "APAC", "US", "EU"])
    assert infer_column_type(cat_series, "region") == DataType.CATEGORICAL

    date_series = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert infer_column_type(date_series, "created_at") == DataType.DATETIME

    bool_series = pd.Series([0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    assert infer_column_type(bool_series, "converted") == DataType.BOOLEAN


def test_outlier_detection_iqr():
    normal_data = pd.Series(np.random.normal(50, 5, 100))
    # Inject 2 obvious outliers
    normal_data[0] = 500.0
    normal_data[1] = -300.0
    outlier_count, outlier_pct, low, high = detect_outliers_iqr(normal_data)
    assert outlier_count >= 2
    assert outlier_pct > 0.0


def test_profile_dataset():
    df = pd.DataFrame({
        "order_id": [f"ID_{i}" for i in range(100)],
        "date": pd.date_range("2024-01-01", periods=100),
        "device": np.random.choice(["iOS", "Android"], size=100),
        "revenue": np.random.uniform(10, 100, size=100)
    })
    prof = profile_dataset(df)
    assert prof.n_rows == 100
    assert prof.n_cols == 4
    assert "revenue" in prof.candidate_metric_cols
    assert "date" in prof.candidate_date_cols
    assert "device" in prof.candidate_segment_cols
