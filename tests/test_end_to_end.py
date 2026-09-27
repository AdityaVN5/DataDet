import pandas as pd
import numpy as np
from pathlib import Path

from data_detective.sample_data import generate_ecommerce_incident_dataset
from data_detective.profiler import profile_dataset, clean_and_prepare_dataset
from data_detective.anomaly_detector import detect_optimal_anomaly_window
from data_detective.hypothesis_engine import generate_and_test_hypotheses
from data_detective.contribution import decompose_all_dimensions
from data_detective.evidence_graph import build_evidence_graph
from data_detective.counterfactual import generate_top_counterfactual_scenarios
from data_detective.report_generator import generate_deterministic_narrative
from data_detective.storage import init_db, save_investigation, list_investigations, load_investigation
from data_detective.schemas import InvestigationRun, HypothesisStatus


def test_end_to_end_investigation(tmp_path: Path):
    # 1. Generate realistic data
    df = generate_ecommerce_incident_dataset(n_samples=1000, seed=123)

    # 2. Profile
    profile = profile_dataset(df)
    assert "converted" in profile.candidate_metric_cols
    assert "date" in profile.candidate_date_cols

    # 3. Clean and prepare
    df_clean = clean_and_prepare_dataset(df, profile, date_col="date")

    # 4. Anomaly detection
    points, window = detect_optimal_anomaly_window(df_clean, metric_col="converted", date_col="date")
    assert window is not None
    assert len(window.baseline_indices) > 0
    assert len(window.anomaly_indices) > 0

    # 5. Formulate & test hypotheses
    hypotheses = generate_and_test_hypotheses(df_clean, profile, window)
    assert len(hypotheses) > 0

    # Check that at least one hypothesis is supported
    supported = [h for h in hypotheses if h.status == HypothesisStatus.SUPPORTED]
    assert len(supported) > 0

    # 6. Oaxaca-Blinder Rate-Mix Decomposition
    decompositions = decompose_all_dimensions(
        df=df_clean,
        metric_col="converted",
        dimension_cols=profile.candidate_segment_cols,
        anomaly_window=window
    )
    assert "device" in decompositions

    flat_contribs = []
    for d_contribs in decompositions.values():
        flat_contribs.extend(d_contribs)

    # 7. Evidence Graph
    G = build_evidence_graph("converted", hypotheses, flat_contribs)
    assert len(G.nodes) > 5
    assert len(G.edges) > 5

    # 8. Counterfactual analysis
    scenarios = generate_top_counterfactual_scenarios(df_clean, window, flat_contribs)
    assert len(scenarios) > 0
    assert scenarios[0].recovery_percentage > 0

    # 9. Narrative report
    narrative = generate_deterministic_narrative(
        run_title="Test E2E Incident",
        metric_col="converted",
        anomaly_window_dict=window.to_dict(),
        top_hypotheses=hypotheses,
        top_contributions=flat_contribs,
        top_scenarios=scenarios
    )
    assert "Root-Cause Investigation Report" in narrative
    assert "Executive Summary" in narrative

    # 10. SQLite persistence
    db_file = tmp_path / "test_detective.db"
    run_obj = InvestigationRun(
        id="RUN-TEST-001",
        title="E2E Test Run",
        created_at="2026-09-27 12:00:00",
        dataset_name="sample_ecommerce",
        row_count=len(df_clean),
        metric_col="converted",
        date_col="date",
        anomaly_window=window,
        hypotheses=hypotheses,
        contributions=flat_contribs,
        counterfactuals=scenarios,
        audit_trail=[],
        executive_summary=narrative
    )
    save_investigation(run_obj, db_path=db_file)
    runs = list_investigations(db_path=db_file)
    assert len(runs) == 1
    assert runs[0]["id"] == "RUN-TEST-001"

    loaded = load_investigation("RUN-TEST-001", db_path=db_file)
    assert loaded is not None
    assert loaded["metric_col"] == "converted"
