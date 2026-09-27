import pandas as pd
import numpy as np
from data_detective.contribution import calculate_segment_contributions
from data_detective.schemas import AnomalyWindow


def test_calculate_segment_contributions():
    # Baseline: 100 iOS (mean conv 0.10) and 100 Android (mean conv 0.10)
    # Anomaly: 100 iOS (mean conv 0.01) and 100 Android (mean conv 0.10)
    df = pd.DataFrame({
        "device": ["iOS"] * 100 + ["Android"] * 100 + ["iOS"] * 100 + ["Android"] * 100,
        "conversion": [0.10] * 100 + [0.10] * 100 + [0.01] * 100 + [0.10] * 100
    })

    window = AnomalyWindow(
        metric_col="conversion",
        date_col=None,
        baseline_period_label="Baseline",
        anomaly_period_label="Anomaly",
        baseline_indices=list(range(200)),
        anomaly_indices=list(range(200, 400)),
        baseline_mean=0.10,
        anomaly_mean=0.055,
        absolute_delta=-0.045,
        relative_pct_change=-45.0,
        direction="drop",
        detection_method="test"
    )

    contribs = calculate_segment_contributions(df, "conversion", "device", window)
    assert len(contribs) == 2

    ios_contrib = next(c for c in contribs if c.segment_value == "iOS")
    android_contrib = next(c for c in contribs if c.segment_value == "Android")

    # iOS should account for the vast majority of the drop and be flagged as primary culprit
    assert ios_contrib.is_primary_culprit is True
    assert ios_contrib.pct_of_total_delta > 80.0
    assert abs(android_contrib.pct_of_total_delta) < 10.0
