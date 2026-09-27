"""
Data Detective Counterfactual and Scenario Simulation Engine.
Computes baseline-adjusted counterfactual projections with 95% confidence intervals
and explicit epistemic caveats (SUTVA, no-spillover, historical stationarity).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_detective.schemas import CounterfactualResult, AnomalyWindow, SegmentContribution


def simulate_segment_restoration_counterfactual(
    df: pd.DataFrame,
    anomaly_window: AnomalyWindow,
    contribution: SegmentContribution
) -> CounterfactualResult:
    """
    Simulates: "What if segment X had maintained its pre-incident baseline performance?"
    Computes expected aggregate metric recovery and 95% confidence intervals.
    """
    metric_col = anomaly_window.metric_col
    dim = contribution.dimension
    seg_val = contribution.segment_value

    base_df = df.loc[anomaly_window.baseline_indices]
    anom_df = df.loc[anomaly_window.anomaly_indices]

    observed_anom_metric = anomaly_window.anomaly_mean
    baseline_metric = anomaly_window.baseline_mean
    total_delta = baseline_metric - observed_anom_metric  # total drop to recover

    # Baseline segment stats
    b_seg = base_df[base_df[dim] == seg_val][metric_col].dropna()
    a_seg = anom_df[anom_df[dim] == seg_val][metric_col].dropna()

    m_base = float(b_seg.mean()) if len(b_seg) > 0 else contribution.baseline_rate
    m_anom = float(a_seg.mean()) if len(a_seg) > 0 else contribution.anomaly_rate
    var_base = float(b_seg.var(ddof=1)) if len(b_seg) > 1 else 0.01
    var_anom = float(a_seg.var(ddof=1)) if len(a_seg) > 1 else 0.01

    w_anom = contribution.anomaly_share
    # Counterfactual improvement: restoring segment from m_anom to m_base
    restoration_delta = w_anom * (m_base - m_anom)
    counterfactual_metric = observed_anom_metric + restoration_delta

    # Total gap between baseline and anomaly
    gap = abs(baseline_metric - observed_anom_metric)
    recovery_pct = (abs(restoration_delta) / gap * 100.0) if gap > 1e-9 else 0.0

    # Uncertainty: SE(restoration_delta) ~ w_anom * sqrt(var_base/n_base + var_anom/n_anom)
    n_base = max(len(b_seg), 2)
    n_anom = max(len(a_seg), 2)
    se_restoration = w_anom * math.sqrt(var_base / n_base + var_anom / n_anom)

    margin = 1.96 * se_restoration
    ci_lower = counterfactual_metric - margin
    ci_upper = counterfactual_metric + margin

    caveats = [
        "SUTVA Assumption: Assumes no spillover or substitution effects occurred between this segment and others.",
        "Baseline Stationarity: Assumes macroeconomic, seasonal, and organic baseline rates would have persisted unchanged.",
        "Partial Equilibrium: Real-world remediation (e.g. bugfix or rollback) might have side-effects not modeled in linear restoration."
    ]

    return CounterfactualResult(
        scenario_name=f"Full Restoration of '{dim}={seg_val}'",
        dimension=dim,
        segment_value=str(seg_val),
        target_metric=metric_col,
        observed_anomaly_metric=observed_anom_metric,
        counterfactual_metric=counterfactual_metric,
        restoration_delta=restoration_delta,
        recovery_percentage=recovery_pct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        assumptions_and_caveats=caveats
    )


def generate_top_counterfactual_scenarios(
    df: pd.DataFrame,
    anomaly_window: AnomalyWindow,
    top_contributions: List[SegmentContribution],
    max_scenarios: int = 4
) -> List[CounterfactualResult]:
    """
    Generates counterfactual restoration scenarios for top contributing degraded segments.
    """
    scenarios: List[CounterfactualResult] = []
    seen = set()

    # Prioritize primary culprits and segments whose rate degraded in the direction of the anomaly
    degraded = [
        c for c in top_contributions
        if c.is_primary_culprit or (anomaly_window.direction == "drop" and c.rate_change < 0) or (anomaly_window.direction == "spike" and c.rate_change > 0)
    ]
    candidates = degraded if degraded else top_contributions

    for c in candidates:
        key = (c.dimension, c.segment_value)
        if key in seen:
            continue
        seen.add(key)

        scenario = simulate_segment_restoration_counterfactual(df, anomaly_window, c)
        scenarios.append(scenario)
        if len(scenarios) >= max_scenarios:
            break

    return scenarios
