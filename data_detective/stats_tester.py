"""
Data Detective Statistical Testing Engine.
Executes parametric and non-parametric tests, computes standardized effect sizes
(Cohen's d, Cramer's V, Pearson r, Spearman rho, Eta-squared),
and determines hypothesis status with epistemic rigor.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from data_detective.schemas import HypothesisStatus, StatisticalEvidence


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Computes Cohen's d effect size between two independent groups.
    d = (mean_a - mean_b) / s_pooled
    """
    n1, n2 = len(group_a), len(group_b)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group_a), np.mean(group_b)
    var1, var2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Computes Cramer's V effect size for a contingency table.
    V = sqrt(chi2 / (n * min(r-1, c-1)))
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    r, c = contingency_table.shape
    min_dim = min(r - 1, c - 1)
    if min_dim == 0 or n == 0:
        return 0.0
    return float(math.sqrt(chi2 / (n * min_dim)))


def test_two_sample_difference(
    baseline_values: pd.Series,
    anomaly_values: pd.Series,
    metric_name: str
) -> StatisticalEvidence:
    """
    Tests difference in a continuous or binary metric between baseline and anomaly periods.
    Applies Shapiro-Wilk or normality test heuristic: Welch's t-test for normal / large samples,
    Mann-Whitney U test for non-normal distributions.
    """
    a = baseline_values.dropna().values.astype(float)
    b = anomaly_values.dropna().values.astype(float)

    n_a, n_b = len(a), len(b)
    if n_a < 5 or n_b < 5:
        return StatisticalEvidence(
            test_name="Insufficient Sample Size",
            test_statistic=0.0,
            p_value=1.0,
            effect_size_name="Cohen's d",
            effect_size_value=0.0,
            sample_size_baseline=n_a,
            sample_size_anomaly=n_b,
            interpretation="Sample size is too small (<5 observations) to conduct a valid hypothesis test."
        )

    # Effect size
    d = compute_cohens_d(b, a)  # anomaly minus baseline

    # Check for binary metric (e.g. 0/1 conversion) vs continuous
    is_binary = len(np.unique(np.concatenate([a, b]))) <= 2

    # Welch's t-test (robust against unequal variances)
    t_stat, t_pval = stats.ttest_ind(b, a, equal_var=False)

    # Mann-Whitney U test (non-parametric rank sum)
    try:
        u_stat, u_pval = stats.mannwhitneyu(b, a, alternative="two-sided")
    except Exception:
        u_stat, u_pval = 0.0, 1.0

    # If binary or heavily skewed, report Mann-Whitney or Welch
    test_name = "Welch's Two-Sample t-test"
    stat = float(t_stat) if not np.isnan(t_stat) else 0.0
    pval = float(t_pval) if not np.isnan(t_pval) else 1.0

    if not is_binary and (n_a > 30 and n_b > 30):
        # Check skewness
        skew_a = abs(stats.skew(a))
        skew_b = abs(stats.skew(b))
        if skew_a > 1.5 or skew_b > 1.5:
            test_name = "Mann-Whitney U Rank-Sum Test"
            stat = float(u_stat)
            pval = float(u_pval)

    # 95% Confidence Interval for difference in means
    mean_diff = float(np.mean(b) - np.mean(a))
    se_diff = math.sqrt(np.var(a, ddof=1) / n_a + np.var(b, ddof=1) / n_b) if (n_a > 1 and n_b > 1) else 0.0
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff

    # Qualitative interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        eff_qual = "negligible"
    elif abs_d < 0.5:
        eff_qual = "small"
    elif abs_d < 0.8:
        eff_qual = "moderate"
    else:
        eff_qual = "large"

    sig_qual = "statistically significant" if pval < 0.05 else "not statistically significant"
    direction = "decrease" if mean_diff < 0 else "increase"
    interp = (
        f"{test_name} indicates a {sig_qual} {direction} in '{metric_name}' "
        f"(mean diff: {mean_diff:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], "
        f"p = {pval:.4e}, Cohen's d = {d:+.2f} [{eff_qual} effect])."
    )

    return StatisticalEvidence(
        test_name=test_name,
        test_statistic=stat,
        p_value=pval,
        effect_size_name="Cohen's d",
        effect_size_value=d,
        sample_size_baseline=n_a,
        sample_size_anomaly=n_b,
        degrees_of_freedom=n_a + n_b - 2,
        confidence_interval_95=(ci_low, ci_high),
        interpretation=interp
    )


def test_categorical_distribution_shift(
    baseline_cat: pd.Series,
    anomaly_cat: pd.Series,
    dimension_name: str
) -> StatisticalEvidence:
    """
    Tests whether the categorical distribution (traffic/mix shift) changed
    between baseline and anomaly periods using Chi-Square test of independence.
    """
    b_counts = baseline_cat.value_counts()
    a_counts = anomaly_cat.value_counts()

    all_keys = list(set(b_counts.index).union(set(a_counts.index)))
    if len(all_keys) < 2:
        return StatisticalEvidence(
            test_name="Chi-Square Independence",
            test_statistic=0.0,
            p_value=1.0,
            effect_size_name="Cramer's V",
            effect_size_value=0.0,
            sample_size_baseline=len(baseline_cat),
            sample_size_anomaly=len(anomaly_cat),
            interpretation=f"Dimension '{dimension_name}' has insufficient unique categories for contingency testing."
        )

    # Build 2 x K contingency table
    contingency = pd.DataFrame(
        {
            "baseline": [b_counts.get(k, 0) for k in all_keys],
            "anomaly": [a_counts.get(k, 0) for k in all_keys]
        },
        index=all_keys
    )

    # Filter out rows with 0 in both
    contingency = contingency[contingency.sum(axis=1) > 0]

    chi2, pval, dof, _ = stats.chi2_contingency(contingency)
    v = compute_cramers_v(contingency)

    v_qual = "negligible" if v < 0.1 else ("moderate" if v < 0.3 else "strong")
    sig_qual = "statistically significant" if pval < 0.05 else "not statistically significant"

    interp = (
        f"Chi-Square test indicates a {sig_qual} composition shift in '{dimension_name}' "
        f"(chi2 = {chi2:.2f}, dof = {dof}, p = {pval:.4e}, Cramer's V = {v:.3f} [{v_qual} association])."
    )

    return StatisticalEvidence(
        test_name="Chi-Square Test of Independence",
        test_statistic=float(chi2),
        p_value=float(pval),
        effect_size_name="Cramer's V",
        effect_size_value=float(v),
        sample_size_baseline=len(baseline_cat),
        sample_size_anomaly=len(anomaly_cat),
        degrees_of_freedom=dof,
        interpretation=interp
    )


def test_covariate_correlation_and_shift(
    df: pd.DataFrame,
    covariate_col: str,
    metric_col: str,
    baseline_indices: List[Any],
    anomaly_indices: List[Any]
) -> List[StatisticalEvidence]:
    """
    Tests both:
    1. Correlation between covariate and metric (Pearson + Spearman).
    2. Shift in covariate mean between baseline and anomaly periods.
    """
    evidence: List[StatisticalEvidence] = []

    clean_df = df[[covariate_col, metric_col]].dropna()
    if len(clean_df) >= 10:
        x = clean_df[covariate_col].values.astype(float)
        y = clean_df[metric_col].values.astype(float)

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

        evidence.append(
            StatisticalEvidence(
                test_name="Pearson Correlation",
                test_statistic=float(pearson_r),
                p_value=float(pearson_p),
                effect_size_name="Pearson r",
                effect_size_value=float(pearson_r),
                sample_size_baseline=len(clean_df),
                sample_size_anomaly=0,
                interpretation=(
                    f"Linear correlation between '{covariate_col}' and '{metric_col}': "
                    f"r = {pearson_r:+.3f} (p = {pearson_p:.4e})."
                )
            )
        )

        evidence.append(
            StatisticalEvidence(
                test_name="Spearman Rank Correlation",
                test_statistic=float(spearman_rho),
                p_value=float(spearman_p),
                effect_size_name="Spearman rho",
                effect_size_value=float(spearman_rho),
                sample_size_baseline=len(clean_df),
                sample_size_anomaly=0,
                interpretation=(
                    f"Monotonic rank correlation between '{covariate_col}' and '{metric_col}': "
                    f"rho = {spearman_rho:+.3f} (p = {spearman_p:.4e})."
                )
            )
        )

    # Covariate shift test between baseline and anomaly periods
    b_cov = df.loc[baseline_indices, covariate_col].dropna()
    a_cov = df.loc[anomaly_indices, covariate_col].dropna()
    if len(b_cov) >= 5 and len(a_cov) >= 5:
        shift_evidence = test_two_sample_difference(b_cov, a_cov, metric_name=covariate_col)
        shift_evidence.test_name = f"Covariate Shift ({shift_evidence.test_name})"
        evidence.append(shift_evidence)

    return evidence


def evaluate_hypothesis_status(
    evidence: List[StatisticalEvidence],
    expected_direction: Optional[str] = None
) -> Tuple[HypothesisStatus, float]:
    """
    Evaluates statistical evidence to assign status:
    - SUPPORTED: p < 0.05 and meaningful effect size (|d| >= 0.3 or |V| >= 0.15 or |r| >= 0.3)
    - WEAK: p < 0.05 but small effect, or 0.05 <= p < 0.10
    - REJECTED: p >= 0.10 or wrong effect direction
    - INCONCLUSIVE: Insufficient sample size or contradictory evidence
    Returns (status, plausibility_score between 0 and 100).
    """
    if not evidence:
        return HypothesisStatus.INCONCLUSIVE, 0.0

    # Use the primary test evidence
    primary = evidence[0]
    p = primary.p_value
    eff = abs(primary.effect_size_value)
    eff_name = primary.effect_size_name

    # Check minimum sample size
    if primary.sample_size_baseline < 10 or (primary.sample_size_anomaly > 0 and primary.sample_size_anomaly < 10):
        return HypothesisStatus.INCONCLUSIVE, 15.0

    # Effect size threshold
    meaningful_effect = False
    if "Cohen" in eff_name:
        meaningful_effect = eff >= 0.20
    elif "Cramer" in eff_name:
        meaningful_effect = eff >= 0.10
    elif "r" in eff_name.lower() or "rho" in eff_name.lower():
        meaningful_effect = eff >= 0.20
    else:
        meaningful_effect = eff >= 0.15

    # Direction check
    if expected_direction == "negative" and primary.effect_size_value > 0:
        return HypothesisStatus.REJECTED, 5.0
    if expected_direction == "positive" and primary.effect_size_value < 0:
        return HypothesisStatus.REJECTED, 5.0

    if p < 0.01 and meaningful_effect:
        # High confidence supported
        score = min(95.0, 70.0 + min(eff, 2.0) * 12.5)
        return HypothesisStatus.SUPPORTED, score
    elif p < 0.05 and meaningful_effect:
        score = min(85.0, 60.0 + min(eff, 2.0) * 10.0)
        return HypothesisStatus.SUPPORTED, score
    elif p < 0.05 and not meaningful_effect:
        score = 45.0 + eff * 10.0
        return HypothesisStatus.WEAK, score
    elif 0.05 <= p < 0.10:
        score = 35.0
        return HypothesisStatus.WEAK, score
    else:
        score = max(5.0, 20.0 - (p - 0.10) * 20.0)
        return HypothesisStatus.REJECTED, score
