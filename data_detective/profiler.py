"""
Data Detective Dataset Profiler and Cleaner.
Provides automated schema inference, missing value profiling, outlier detection,
and heuristic detection of candidate metrics, dates, and segment dimensions.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_detective.schemas import ColumnProfile, DatasetProfile, DataType


# Common keywords indicating dates, IDs, and business metrics
DATE_KEYWORDS = ["date", "time", "timestamp", "day", "created", "updated", "ts", "dt", "period", "month"]
ID_KEYWORDS = ["id", "uuid", "guid", "key", "token", "hash", "session_id", "user_id", "order_id", "transaction_id"]
METRIC_KEYWORDS = ["revenue", "sales", "gmv", "amount", "price", "conversion", "rate", "churn",
                   "latency", "duration", "clicks", "orders", "cost", "margin", "profit",
                   "sessions", "bounce", "errors", "failed", "success", "count", "value", "score"]
SEGMENT_KEYWORDS = ["category", "segment", "region", "country", "device", "platform", "os", "browser",
                    "channel", "tier", "plan", "status", "version", "source", "medium", "type", "group"]


def infer_column_type(series: pd.Series, col_name: str) -> DataType:
    """
    Infers the high-level semantic type of a column using data checks and heuristics.
    """
    clean_series = series.dropna()
    total_count = len(clean_series)
    if total_count == 0:
        return DataType.CATEGORICAL

    # Check for boolean
    if pd.api.types.is_bool_dtype(series):
        return DataType.BOOLEAN
    if set(clean_series.unique()).issubset({0, 1, 0.0, 1.0, "0", "1", "true", "false", "True", "False"}):
        if total_count > 10 and len(clean_series.unique()) <= 2:
            return DataType.BOOLEAN

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return DataType.DATETIME

    lower_col = col_name.lower()
    if any(k in lower_col for k in DATE_KEYWORDS):
        # Try parsing sample
        sample = clean_series.head(30).astype(str)
        try:
            pd.to_datetime(sample, errors="raise")
            return DataType.DATETIME
        except Exception:
            pass

    # Check for numerical
    if pd.api.types.is_numeric_dtype(series):
        # If numeric but integer with 100% unique values and matches id keywords
        if any(k in lower_col for k in ID_KEYWORDS) and clean_series.nunique() / total_count > 0.9:
            return DataType.TEXT_ID
        return DataType.NUMERICAL

    # String / Object types
    unique_count = clean_series.nunique()
    unique_ratio = unique_count / total_count

    # Check if string date
    if unique_ratio > 0.05 and any(k in lower_col for k in DATE_KEYWORDS):
        try:
            pd.to_datetime(clean_series.head(50), errors="raise")
            return DataType.DATETIME
        except Exception:
            pass

    # Check for IDs / High cardinality text
    if any(k in lower_col for k in ID_KEYWORDS) or (unique_ratio > 0.85 and total_count > 50):
        return DataType.TEXT_ID

    # Categorical
    if unique_count <= 100 or unique_ratio < 0.3:
        return DataType.CATEGORICAL

    return DataType.TEXT_ID


def detect_outliers_iqr(series: pd.Series) -> Tuple[int, float, float, float]:
    """
    Detects outliers using the standard Interquartile Range (IQR) method.
    Returns (outlier_count, outlier_pct, lower_bound, upper_bound).
    """
    valid = series.dropna()
    if len(valid) < 5:
        return 0, 0.0, 0.0, 0.0

    q25 = float(np.percentile(valid, 25))
    q75 = float(np.percentile(valid, 75))
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outliers = (valid < lower_bound) | (valid > upper_bound)
    count = int(outliers.sum())
    pct = (count / len(valid)) * 100.0
    return count, pct, lower_bound, upper_bound


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Executes a comprehensive data profiling pass on the input DataFrame.
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / n_rows * 100.0) if n_rows > 0 else 0.0

    columns_profile: Dict[str, ColumnProfile] = {}
    candidate_dates: List[str] = []
    candidate_metrics: List[str] = []
    candidate_segments: List[str] = []

    for col in df.columns:
        series = df[col]
        total_count = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1

        inferred_type = infer_column_type(series, str(col))
        lower_col = str(col).lower()
        is_id = inferred_type == DataType.TEXT_ID or any(k in lower_col for k in ID_KEYWORDS)

        # Statistical metrics for numerical columns
        min_val = None
        max_val = None
        mean_val = None
        median_val = None
        std_val = None
        iqr_val = None
        outlier_count = 0
        outlier_pct = 0.0
        top_cats = {}

        if inferred_type in [DataType.NUMERICAL, DataType.BOOLEAN]:
            numeric_series = pd.to_numeric(series, errors="coerce")
            valid = numeric_series.dropna()
            if len(valid) > 0:
                min_val = float(valid.min())
                max_val = float(valid.max())
                mean_val = float(valid.mean())
                median_val = float(valid.median())
                std_val = float(valid.std()) if len(valid) > 1 else 0.0
                q25 = float(np.percentile(valid, 25))
                q75 = float(np.percentile(valid, 75))
                iqr_val = q75 - q25
                outlier_count, outlier_pct, _, _ = detect_outliers_iqr(valid)

        if inferred_type in [DataType.CATEGORICAL, DataType.BOOLEAN]:
            val_counts = series.value_counts(dropna=True).head(5).to_dict()
            top_cats = {str(k): int(v) for k, v in val_counts.items()}

        col_prof = ColumnProfile(
            name=str(col),
            inferred_type=inferred_type,
            raw_dtype=str(series.dtype),
            total_count=total_count,
            missing_count=missing_count,
            missing_pct=round(missing_pct, 2),
            unique_count=unique_count,
            is_constant=is_constant,
            is_id_candidate=is_id,
            min_val=round(min_val, 4) if min_val is not None else None,
            max_val=round(max_val, 4) if max_val is not None else None,
            mean_val=round(mean_val, 4) if mean_val is not None else None,
            median_val=round(median_val, 4) if median_val is not None else None,
            std_val=round(std_val, 4) if std_val is not None else None,
            iqr=round(iqr_val, 4) if iqr_val is not None else None,
            outlier_count=outlier_count,
            outlier_pct=round(outlier_pct, 2),
            top_categories=top_cats,
        )
        columns_profile[str(col)] = col_prof

        # Classify candidates
        if inferred_type == DataType.DATETIME:
            candidate_dates.append(str(col))
        elif inferred_type in [DataType.NUMERICAL, DataType.BOOLEAN] and not is_constant and not is_id:
            candidate_metrics.append(str(col))

        if inferred_type in [DataType.CATEGORICAL, DataType.BOOLEAN] and 2 <= unique_count <= 80 and not is_id:
            candidate_segments.append(str(col))

    # Priority sorting of candidate metrics
    def metric_priority(m: str) -> int:
        low = m.lower()
        if any(k in low for k in ["conversion", "churn", "revenue", "sales", "gmv", "rate", "latency", "error"]):
            return 0
        if any(k in low for k in ["amount", "count", "score", "price", "duration"]):
            return 1
        return 2

    candidate_metrics.sort(key=metric_priority)

    # Priority sorting of segments
    def segment_priority(s: str) -> int:
        low = s.lower()
        if any(k in low for k in ["device", "os", "platform", "country", "region", "channel", "category", "plan"]):
            return 0
        return 1

    candidate_segments.sort(key=segment_priority)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=duplicate_rows,
        duplicate_pct=round(duplicate_pct, 2),
        columns=columns_profile,
        candidate_date_cols=candidate_dates,
        candidate_metric_cols=candidate_metrics,
        candidate_segment_cols=candidate_segments,
    )


def clean_and_prepare_dataset(
    df: pd.DataFrame,
    profile: DatasetProfile,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Cleans string whitespaces, parses datetime column if specified,
    and returns a standardized working copy.
    """
    df_clean = df.copy()

    # Clean string columns: strip whitespaces
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Coerce date column
    if date_col and date_col in df_clean.columns:
        try:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
            df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)
        except Exception:
            pass

    return df_clean
