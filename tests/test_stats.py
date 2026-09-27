import numpy as np
import pandas as pd
from data_detective.stats_tester import (
    compute_cohens_d,
    compute_cramers_v,
    test_two_sample_difference as run_two_sample_difference,
    test_categorical_distribution_shift as run_categorical_shift,
    evaluate_hypothesis_status
)
from data_detective.schemas import HypothesisStatus


def test_compute_cohens_d():
    g1 = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
    g2 = np.array([20.0, 21.0, 22.0, 20.5, 21.5])
    d = compute_cohens_d(g2, g1)
    assert d > 5.0  # Large positive difference


def test_two_sample_difference_calc():
    # Significant drop
    base = pd.Series(np.random.normal(100, 5, 50))
    anom = pd.Series(np.random.normal(50, 5, 50))
    ev = run_two_sample_difference(base, anom, "revenue")
    assert ev.p_value < 0.001
    assert ev.effect_size_value < -2.0

    status, score = evaluate_hypothesis_status([ev], expected_direction="negative")
    assert status == HypothesisStatus.SUPPORTED
    assert score > 70.0


def test_categorical_distribution_shift_calc():
    b_cat = pd.Series(["A"] * 80 + ["B"] * 20)
    # Severe mix shift in anomaly
    a_cat = pd.Series(["A"] * 20 + ["B"] * 80)
    ev = run_categorical_shift(b_cat, a_cat, "traffic_source")
    assert ev.p_value < 0.001
    assert ev.effect_size_value > 0.4
