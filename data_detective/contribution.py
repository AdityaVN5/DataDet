"""
Data Detective Contribution and Decomposition Engine.
Implements Rate-Mix Decomposition (Kitagawa-Oaxaca-Blinder decomposition)
to partition metric delta into within-segment performance shifts and between-segment mix shifts.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_detective.schemas import SegmentContribution, AnomalyWindow


def calculate_segment_contributions(
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
    anomaly_window: AnomalyWindow
) -> List[SegmentContribution]:
    """
    Decomposes the metric change between baseline and anomaly periods across segments of a dimension.
    Uses Rate-Mix Decomposition:
    Delta M = Sum_s [ w_mean_s * Delta m_s + m_mean_s * Delta w_s ]
    """
    base_df = df.loc[anomaly_window.baseline_indices]
    anom_df = df.loc[anomaly_window.anomaly_indices]

    total_base_vol = len(base_df)
    total_anom_vol = len(anom_df)

    if total_base_vol == 0 or total_anom_vol == 0:
        return []

    overall_base_rate = float(base_df[metric_col].mean())
    overall_anom_rate = float(anom_df[metric_col].mean())
    total_delta = overall_anom_rate - overall_base_rate

    # Unique segments in both periods
    base_counts = base_df[dimension_col].value_counts()
    anom_counts = anom_df[dimension_col].value_counts()
    all_segments = list(set(base_counts.index).union(set(anom_counts.index)))

    contributions: List[SegmentContribution] = []

    for seg in all_segments:
        base_seg = base_df[base_df[dimension_col] == seg]
        anom_seg = anom_df[anom_df[dimension_col] == seg]

        b_vol = len(base_seg)
        a_vol = len(anom_seg)

        # Segment weights (volume share)
        w_base = b_vol / total_base_vol if total_base_vol > 0 else 0.0
        w_anom = a_vol / total_anom_vol if total_anom_vol > 0 else 0.0

        # Segment rates
        m_base = float(base_seg[metric_col].mean()) if b_vol > 0 else overall_base_rate
        m_anom = float(anom_seg[metric_col].mean()) if a_vol > 0 else overall_anom_rate

        delta_m = m_anom - m_base
        delta_w = w_anom - w_base

        w_mean = (w_base + w_anom) / 2.0
        m_mean = (m_base + m_anom) / 2.0

        # Absolute contribution = Rate Effect + Mix Effect
        rate_effect = w_mean * delta_m
        mix_effect = m_mean * delta_w
        abs_contrib = rate_effect + mix_effect

        pct_of_total = (abs_contrib / total_delta * 100.0) if abs(total_delta) > 1e-9 else 0.0

        # Primary culprit heuristic: contributes substantially in same direction
        is_culprit = False
        if abs(pct_of_total) >= 25.0 and (abs_contrib * total_delta > 0):
            is_culprit = True

        contributions.append(
            SegmentContribution(
                dimension=dimension_col,
                segment_value=str(seg),
                baseline_volume=b_vol,
                anomaly_volume=a_vol,
                baseline_share=w_base,
                anomaly_share=w_anom,
                baseline_rate=m_base,
                anomaly_rate=m_anom,
                rate_change=delta_m,
                absolute_contribution=abs_contrib,
                pct_of_total_delta=pct_of_total,
                is_primary_culprit=is_culprit,
            )
        )

    # Sort contributions by absolute impact
    contributions.sort(key=lambda c: abs(c.absolute_contribution), reverse=True)
    return contributions


def decompose_all_dimensions(
    df: pd.DataFrame,
    metric_col: str,
    dimension_cols: List[str],
    anomaly_window: AnomalyWindow
) -> Dict[str, List[SegmentContribution]]:
    """
    Computes decomposition across all candidate categorical dimensions.
    Returns mapping of dimension_name -> list of segment contributions.
    """
    results: Dict[str, List[SegmentContribution]] = {}
    for dim in dimension_cols:
        if dim in df.columns and dim != metric_col and dim != anomaly_window.date_col:
            contribs = calculate_segment_contributions(df, metric_col, dim, anomaly_window)
            if contribs:
                results[dim] = contribs
    return results
