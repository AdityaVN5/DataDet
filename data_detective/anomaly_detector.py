"""
Data Detective Anomaly and Changepoint Detection Engine.
Supports rolling time-series z-score/IQR anomaly detection, CUSUM/step-change detection,
and multivariate Isolation Forest anomaly detection.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_detective.schemas import AnomalyPoint, AnomalyWindow


def detect_rolling_anomalies(
    df: pd.DataFrame,
    metric_col: str,
    date_col: Optional[str] = None,
    window_size: int = 7,
    z_threshold: float = 2.5
) -> Tuple[List[AnomalyPoint], Optional[AnomalyWindow]]:
    """
    Detects time-series anomalies using rolling mean and standard deviation.
    Identifies the most prominent contiguous anomaly period as the Anomaly Window.
    """
    if metric_col not in df.columns or len(df) < 5:
        return [], None

    # Work on a copy with valid metric values
    data = df.copy()
    if date_col and date_col in data.columns:
        data = data.sort_values(by=date_col).reset_index(drop=True)
        # Check if dataset is session/row-level with multiple records per timestamp
        if data[date_col].nunique() < len(data) and data[date_col].nunique() >= 5:
            # Aggregate to date-level to detect temporal changepoints accurately
            agg_df = data.groupby(date_col)[metric_col].agg(["mean"]).reset_index()
            agg_df.rename(columns={"mean": metric_col}, inplace=True)
            agg_pts, agg_win = detect_rolling_anomalies(agg_df, metric_col=metric_col, date_col=date_col, window_size=max(2, min(5, len(agg_df)//4)), z_threshold=z_threshold)
            if agg_win and len(agg_win.flagged_points) > 0:
                anom_date = agg_win.flagged_points[0].index_or_date
                anom_mask = (data[date_col].astype(str) >= anom_date)
                base_idx = data[~anom_mask].index.tolist()
                anom_idx = data[anom_mask].index.tolist()

                if len(base_idx) >= 5 and len(anom_idx) >= 5:
                    b_mean = float(data.loc[base_idx, metric_col].mean())
                    a_mean = float(data.loc[anom_idx, metric_col].mean())
                    d_val = a_mean - b_mean
                    pct = (d_val / (abs(b_mean) + 1e-9)) * 100.0
                    return agg_pts, AnomalyWindow(
                        metric_col=metric_col,
                        date_col=date_col,
                        baseline_period_label=f"Baseline (< {anom_date})",
                        anomaly_period_label=f"Anomaly Window (>= {anom_date})",
                        baseline_indices=base_idx,
                        anomaly_indices=anom_idx,
                        baseline_mean=b_mean,
                        anomaly_mean=a_mean,
                        absolute_delta=d_val,
                        relative_pct_change=pct,
                        direction="drop" if d_val < 0 else "spike",
                        detection_method="temporal_changepoint",
                        flagged_points=agg_pts
                    )

    series = pd.to_numeric(data[metric_col], errors="coerce").bfill().fillna(0)

    # Use lagged window to compute expectation from preceding points (avoids anomaly diluting the baseline variance)
    lagged = series.shift(1)
    rolling_mean = lagged.rolling(window=window_size, min_periods=max(2, window_size // 2)).mean().bfill().fillna(series.mean())
    rolling_std = lagged.rolling(window=window_size, min_periods=max(2, window_size // 2)).std().bfill().fillna(series.std())
    rolling_std = rolling_std.replace(0, series.std() if series.std() > 0 else 1e-6)

    z_scores = (series - rolling_mean) / rolling_std

    anomalies: List[AnomalyPoint] = []
    anomaly_indices = []

    for i in range(len(series)):
        val = float(series.iloc[i])
        exp = float(rolling_mean.iloc[i])
        z = float(z_scores.iloc[i])
        is_anom = abs(z) >= z_threshold

        idx_label = str(data[date_col].iloc[i]) if (date_col and date_col in data.columns) else str(data.index[i])

        severity = "normal"
        if is_anom:
            if abs(z) >= 4.0:
                severity = "critical"
            elif abs(z) >= 3.0:
                severity = "high"
            else:
                severity = "medium"
            anomaly_indices.append(i)

        anomalies.append(
            AnomalyPoint(
                index_or_date=idx_label,
                metric_value=round(val, 4),
                expected_value=round(exp, 4),
                deviation_z=round(z, 2),
                is_anomaly=is_anom,
                anomaly_score=round(min(abs(z) / 5.0, 1.0), 3),
                severity=severity,
            )
        )

    # Construct the primary AnomalyWindow
    if len(anomaly_indices) > 0:
        # Find the most severe anomaly point
        most_severe_idx = int(np.argmax([abs(p.deviation_z) for p in anomalies]))

        # Define changepoint at the most severe point (or start of contiguous anomaly block containing it)
        changepoint_idx = most_severe_idx
        # Walk back to find the start of the consecutive anomaly cluster
        while changepoint_idx > 0 and changepoint_idx - 1 in anomaly_indices:
            changepoint_idx -= 1

        subsequent = series.iloc[changepoint_idx:]
        prior = series.iloc[:changepoint_idx]

        if len(prior) >= 3 and len(subsequent) >= 1:
            baseline_idx = list(data.index[:changepoint_idx])
            anom_idx = list(data.index[changepoint_idx:])
        else:
            # Fall back to points flagged as anomalous vs rest
            anom_idx = [data.index[i] for i in anomaly_indices]
            baseline_idx = [i for i in data.index if i not in anom_idx]

        base_mean = float(data.loc[baseline_idx, metric_col].mean())
        anom_mean = float(data.loc[anom_idx, metric_col].mean())
        delta = anom_mean - base_mean
        pct_change = (delta / (abs(base_mean) + 1e-9)) * 100.0
        direction = "drop" if delta < 0 else "spike"

        base_label = f"Baseline (N={len(baseline_idx)})"
        anom_label = f"Anomaly Window (N={len(anom_idx)})"
        if date_col and date_col in data.columns:
            base_label = f"{data.loc[baseline_idx[0], date_col]} to {data.loc[baseline_idx[-1], date_col]}"
            anom_label = f"{data.loc[anom_idx[0], date_col]} to {data.loc[anom_idx[-1], date_col]}"

        anom_window = AnomalyWindow(
            metric_col=metric_col,
            date_col=date_col,
            baseline_period_label=base_label,
            anomaly_period_label=anom_label,
            baseline_indices=baseline_idx,
            anomaly_indices=anom_idx,
            baseline_mean=base_mean,
            anomaly_mean=anom_mean,
            absolute_delta=delta,
            relative_pct_change=pct_change,
            direction=direction,
            detection_method="rolling_zscore",
            flagged_points=[p for p in anomalies if p.is_anomaly],
        )
        return anomalies, anom_window

    return anomalies, None


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    metric_col: str,
    feature_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    contamination: float = 0.05
) -> Tuple[List[AnomalyPoint], AnomalyWindow]:
    """
    Multivariate anomaly detection using Scikit-Learn Isolation Forest.
    Works for tabular data even without an explicit time-series trend.
    """
    data = df.copy()
    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != metric_col][:10]

    eval_cols = [metric_col] + [c for c in feature_cols if c in data.columns]
    clean_data = data[eval_cols].dropna()

    if len(clean_data) < 20:
        # Fall back to mean-split window
        mid = len(data) // 2
        base_idx = list(data.index[:mid])
        anom_idx = list(data.index[mid:])
        base_mean = float(data.loc[base_idx, metric_col].mean())
        anom_mean = float(data.loc[anom_idx, metric_col].mean())
        delta = anom_mean - base_mean
        pct = (delta / (abs(base_mean) + 1e-9)) * 100
        return [], AnomalyWindow(
            metric_col=metric_col,
            date_col=date_col,
            baseline_period_label="Baseline Cohort",
            anomaly_period_label="Comparison Cohort",
            baseline_indices=base_idx,
            anomaly_indices=anom_idx,
            baseline_mean=base_mean,
            anomaly_mean=anom_mean,
            absolute_delta=delta,
            relative_pct_change=pct,
            direction="drop" if delta < 0 else "spike",
            detection_method="fallback_split",
            flagged_points=[],
        )

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    preds = iso.fit_predict(clean_data)
    scores = -iso.score_samples(clean_data)  # Higher score = more anomalous

    anom_indices = clean_data.index[preds == -1].tolist()
    baseline_indices = clean_data.index[preds == 1].tolist()

    points: List[AnomalyPoint] = []
    base_mean = float(data.loc[baseline_indices, metric_col].mean())
    base_std = float(data.loc[baseline_indices, metric_col].std()) if len(baseline_indices) > 1 else 1.0

    for idx, row in clean_data.iterrows():
        is_anom = (idx in anom_indices)
        val = float(row[metric_col])
        z = (val - base_mean) / (base_std + 1e-9)
        idx_str = str(data.loc[idx, date_col]) if (date_col and date_col in data.columns) else str(idx)

        sev = "normal"
        if is_anom:
            sev = "critical" if abs(z) > 3.0 else "high"

        points.append(
            AnomalyPoint(
                index_or_date=idx_str,
                metric_value=round(val, 4),
                expected_value=round(base_mean, 4),
                deviation_z=round(z, 2),
                is_anomaly=is_anom,
                anomaly_score=round(float(scores[clean_data.index.get_loc(idx)]), 3),
                severity=sev
            )
        )

    anom_mean = float(data.loc[anom_indices, metric_col].mean())
    delta = anom_mean - base_mean
    pct_change = (delta / (abs(base_mean) + 1e-9)) * 100.0

    anom_window = AnomalyWindow(
        metric_col=metric_col,
        date_col=date_col,
        baseline_period_label=f"Inlier Cohort (N={len(baseline_indices)})",
        anomaly_period_label=f"Isolation Forest Anomaly Cluster (N={len(anom_indices)})",
        baseline_indices=baseline_indices,
        anomaly_indices=anom_indices,
        baseline_mean=base_mean,
        anomaly_mean=anom_mean,
        absolute_delta=delta,
        relative_pct_change=pct_change,
        direction="drop" if delta < 0 else "spike",
        detection_method="isolation_forest",
        flagged_points=[p for p in points if p.is_anomaly]
    )

    return points, anom_window


def detect_optimal_anomaly_window(
    df: pd.DataFrame,
    metric_col: str,
    date_col: Optional[str] = None
) -> Tuple[List[AnomalyPoint], AnomalyWindow]:
    """
    Automatically selects the best anomaly detection strategy:
    Time-series rolling z-score if date column exists and has temporal order,
    otherwise Isolation Forest multivariate anomaly detection.
    """
    if date_col and date_col in df.columns:
        points, window = detect_rolling_anomalies(df, metric_col=metric_col, date_col=date_col)
        if window is not None and len(window.flagged_points) > 0:
            return points, window

    # Fall back to Isolation Forest
    return detect_isolation_forest_anomalies(df, metric_col=metric_col, date_col=date_col)
