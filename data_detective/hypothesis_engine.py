"""
Data Detective Hypothesis Generation and Testing Engine.
Generates testable scientific hypotheses across segments, mix shifts, and covariates,
tests them statistically, and ranks candidate explanations with epistemic modesty.
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_detective.schemas import (
    Hypothesis,
    HypothesisStatus,
    HypothesisType,
    StatisticalEvidence,
    AnomalyWindow,
    DatasetProfile
)
from data_detective.stats_tester import (
    test_two_sample_difference,
    test_categorical_distribution_shift,
    test_covariate_correlation_and_shift,
    evaluate_hypothesis_status
)
from data_detective.contribution import calculate_segment_contributions


STANDARD_CAVEATS = [
    "Observational Association: This finding represents a statistical relationship and does not prove direct causation.",
    "Potential Confounding: External or unmeasured variables not in this dataset may account for this pattern.",
    "Multiple Testing: High-dimensional hypothesis exploration incurs risk of false discovery; verify with holdout or A/B experiment."
]


def generate_and_test_hypotheses(
    df: pd.DataFrame,
    profile: DatasetProfile,
    anomaly_window: AnomalyWindow,
    top_n_segments: int = 5
) -> List[Hypothesis]:
    """
    Formulates and executes testable scientific hypotheses against the dataset.
    Ranks resulting candidate explanations based on plausibility and effect magnitude.
    """
    hypotheses: List[Hypothesis] = []
    metric_col = anomaly_window.metric_col
    base_indices = anomaly_window.baseline_indices
    anom_indices = anomaly_window.anomaly_indices

    base_df = df.loc[base_indices]
    anom_df = df.loc[anom_indices]

    # 1. SEGMENT PERFORMANCE DEGRADATION HYPOTHESES
    for dim in profile.candidate_segment_cols:
        if dim == metric_col or dim == anomaly_window.date_col:
            continue

        contribs = calculate_segment_contributions(df, metric_col, dim, anomaly_window)
        # Select top contributors to investigate
        for c in contribs[:top_n_segments]:
            seg_val = c.segment_value
            b_sub = base_df[base_df[dim] == seg_val][metric_col]
            a_sub = anom_df[anom_df[dim] == seg_val][metric_col]

            if len(b_sub) >= 5 and len(a_sub) >= 5:
                # Test whether performance in this segment dropped
                evidence = [test_two_sample_difference(b_sub, a_sub, f"{metric_col} | {dim}={seg_val}")]
                expected_dir = "negative" if anomaly_window.direction == "drop" else "positive"
                status, score = evaluate_hypothesis_status(evidence, expected_direction=expected_dir)

                # Boost score by contribution weight
                adjusted_score = min(100.0, score * 0.7 + min(abs(c.pct_of_total_delta), 100.0) * 0.3)

                h = Hypothesis(
                    id=f"HYP-SEG-{uuid.uuid4().hex[:6].upper()}",
                    title=f"Performance drop localized to {dim}='{seg_val}'",
                    hypothesis_type=HypothesisType.SEGMENT_PERFORMANCE,
                    target_metric=metric_col,
                    dimension=dim,
                    dimension_value=str(seg_val),
                    description=(
                        f"The observed metric shift was driven by degradation within the '{seg_val}' "
                        f"segment of '{dim}', where average {metric_col} changed from {c.baseline_rate:.4f} "
                        f"to {c.anomaly_rate:.4f} (accounting for {c.pct_of_total_delta:.1f}% of total delta)."
                    ),
                    rationale=f"Segments accounting for >15% of overall metric delta are prime candidates for localized degradation.",
                    test_plan=f"Compare distribution of {metric_col} within {dim}='{seg_val}' before vs during anomaly window using two-sample difference testing.",
                    evidence=evidence,
                    status=status,
                    plausibility_score=adjusted_score,
                    caveats=STANDARD_CAVEATS.copy(),
                    segment_stats=c.to_dict()
                )
                hypotheses.append(h)

        # 2. MIX SHIFT HYPOTHESES (Simpson's Paradox / Volume Composition)
        b_cat = base_df[dim].dropna()
        a_cat = anom_df[dim].dropna()
        if len(b_cat) >= 10 and len(a_cat) >= 10 and b_cat.nunique() >= 2:
            mix_evidence = [test_categorical_distribution_shift(b_cat, a_cat, dim)]
            status, score = evaluate_hypothesis_status(mix_evidence)

            h_mix = Hypothesis(
                id=f"HYP-MIX-{uuid.uuid4().hex[:6].upper()}",
                title=f"Traffic / Population mix shift across '{dim}'",
                hypothesis_type=HypothesisType.MIX_SHIFT,
                target_metric=metric_col,
                dimension=dim,
                dimension_value=None,
                description=(
                    f"A significant shift occurred in the relative volume distribution across categories of '{dim}', "
                    f"potentially diluting the aggregate metric (Simpson's paradox / composition effect)."
                ),
                rationale="Changes in customer or traffic proportions can shift aggregate rates even without individual cohort degradation.",
                test_plan=f"Chi-square test of independence on {dim} category proportions between baseline and anomaly periods.",
                evidence=mix_evidence,
                status=status,
                plausibility_score=score,
                caveats=STANDARD_CAVEATS.copy()
            )
            hypotheses.append(h_mix)

    # 3. NUMERICAL COVARIATE DRIFT HYPOTHESES
    for cov_col in profile.candidate_metric_cols:
        if cov_col == metric_col or cov_col == anomaly_window.date_col:
            continue

        cov_evidence = test_covariate_correlation_and_shift(
            df=df,
            covariate_col=cov_col,
            metric_col=metric_col,
            baseline_indices=base_indices,
            anomaly_indices=anom_indices
        )

        if cov_evidence:
            # Check shift evidence
            shift_ev = [e for e in cov_evidence if "Shift" in e.test_name]
            corr_ev = [e for e in cov_evidence if "Correlation" in e.test_name]

            eval_ev = shift_ev if shift_ev else corr_ev
            status, score = evaluate_hypothesis_status(eval_ev)

            # Adjust score if both correlated and shifted
            has_corr = any(abs(e.effect_size_value) > 0.25 and e.p_value < 0.05 for e in corr_ev)
            has_shift = any(e.p_value < 0.05 for e in shift_ev)
            if has_corr and has_shift:
                status = HypothesisStatus.SUPPORTED
                score = min(92.0, score + 20.0)

            h_cov = Hypothesis(
                id=f"HYP-COV-{uuid.uuid4().hex[:6].upper()}",
                title=f"Drift in covariate '{cov_col}' correlated with {metric_col}",
                hypothesis_type=HypothesisType.COVARIATE_DRIFT,
                target_metric=metric_col,
                dimension=cov_col,
                dimension_value=None,
                description=(
                    f"Numerical driver '{cov_col}' underwent a statistically significant shift during the anomaly period "
                    f"and exhibits statistical correlation with '{metric_col}'."
                ),
                rationale="Unplanned shifts in operational covariates (e.g. latency, discounts, error rates) frequently precede metric degradation.",
                test_plan=f"Pearson and Spearman correlation testing combined with Welch's t-test on '{cov_col}'.",
                evidence=cov_evidence,
                status=status,
                plausibility_score=score,
                caveats=STANDARD_CAVEATS + [f"Correlation between {cov_col} and {metric_col} may be mediated by a third unobserved factor."]
            )
            hypotheses.append(h_cov)

    # 4. CROSS-SEGMENT INTERACTION (Multi-dimensional root cause)
    if len(profile.candidate_segment_cols) >= 2:
        d1, d2 = profile.candidate_segment_cols[0], profile.candidate_segment_cols[1]
        combo_series_base = base_df[d1].astype(str) + " × " + base_df[d2].astype(str)
        combo_series_anom = anom_df[d1].astype(str) + " × " + anom_df[d2].astype(str)

        # Check top cross-segment with largest drop
        top_combos = combo_series_anom.value_counts().head(3).index.tolist()
        for combo in top_combos:
            b_mask = combo_series_base == combo
            a_mask = combo_series_anom == combo
            b_vals = base_df.loc[b_mask[b_mask].index, metric_col]
            a_vals = anom_df.loc[a_mask[a_mask].index, metric_col]

            if len(b_vals) >= 8 and len(a_vals) >= 8:
                ev = [test_two_sample_difference(b_vals, a_vals, f"{metric_col} | {combo}")]
                status, score = evaluate_hypothesis_status(ev)
                if status in [HypothesisStatus.SUPPORTED, HypothesisStatus.WEAK]:
                    h_combo = Hypothesis(
                        id=f"HYP-INT-{uuid.uuid4().hex[:6].upper()}",
                        title=f"Interaction anomaly in cohort '{combo}'",
                        hypothesis_type=HypothesisType.MULTIVARIATE_ANOMALY,
                        target_metric=metric_col,
                        dimension=f"{d1} × {d2}",
                        dimension_value=str(combo),
                        description=(
                            f"The intersection cohort '{combo}' suffered a severe localized performance change, "
                            f"indicating a multi-dimensional failure or niche vulnerability."
                        ),
                        rationale="System bugs, pricing anomalies, or tracking failures often isolate to specific multi-dimensional intersections (e.g. iOS in Germany).",
                        test_plan=f"Two-sample hypothesis test on interaction group '{combo}' across {metric_col}.",
                        evidence=ev,
                        status=status,
                        plausibility_score=score + 5.0,
                        caveats=STANDARD_CAVEATS + ["Small sample size warning: multi-dimensional intersections have wider confidence intervals."]
                    )
                    hypotheses.append(h_combo)

    # Sort hypotheses: Supported first, then by plausibility score
    status_order = {
        HypothesisStatus.SUPPORTED: 0,
        HypothesisStatus.WEAK: 1,
        HypothesisStatus.INCONCLUSIVE: 2,
        HypothesisStatus.REJECTED: 3
    }
    hypotheses.sort(key=lambda h: (status_order.get(h.status, 4), -h.plausibility_score))

    # Assign ranks
    for rank_idx, h in enumerate(hypotheses, start=1):
        h.rank = rank_idx

    return hypotheses
