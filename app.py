"""
Data Detective: Evidence-Driven Automated Root-Cause Analysis Platform.
Streamlit Application Entry Point.
"""

from __future__ import annotations
import datetime
import io
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_detective.schemas import (
    InvestigationRun,
    Hypothesis,
    HypothesisStatus,
    HypothesisType,
    AuditLogEntry,
    AnomalyWindow,
    SegmentContribution,
    CounterfactualResult
)
from data_detective.profiler import profile_dataset, clean_and_prepare_dataset
from data_detective.anomaly_detector import detect_optimal_anomaly_window, detect_rolling_anomalies
from data_detective.hypothesis_engine import generate_and_test_hypotheses
from data_detective.contribution import decompose_all_dimensions, calculate_segment_contributions
from data_detective.evidence_graph import build_evidence_graph, plot_evidence_graph_plotly
from data_detective.counterfactual import generate_top_counterfactual_scenarios, simulate_segment_restoration_counterfactual
from data_detective.report_generator import generate_deterministic_narrative, generate_llm_enhanced_summary, export_report_html
from data_detective.storage import init_db, save_investigation, list_investigations, load_investigation, delete_investigation
from data_detective.sample_data import ensure_sample_data_files


from dotenv import load_dotenv

# Load environment variables (.env / .env.example)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Data Detective | Root Cause Analysis",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light-Mode Styling for maximum readability and visual contrast
st.markdown("""
<style>
    /* Clean, Modern Light Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-delta-neg {
        color: #dc2626;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .metric-delta-pos {
        color: #16a34a;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .status-badge-supported {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #047857;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .status-badge-weak {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        color: #b45309;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .status-badge-rejected {
        background-color: #fef2f2;
        border: 1px solid #ef4444;
        color: #b91c1c;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .status-badge-inconclusive {
        background-color: #f3f4f6;
        border: 1px solid #9ca3af;
        color: #4b5563;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .culprit-banner {
        background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
        border-left: 5px solid #ef4444;
        border: 1px solid #fecaca;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 14px 0;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.08);
    }
    .audit-step {
        border-left: 3px solid #2563eb;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .epistemic-banner {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        border: 1px solid #bfdbfe;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize DB on start
init_db()

# Session State Initialization
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "profile" not in st.session_state:
    st.session_state.profile = None
if "selected_metric" not in st.session_state:
    st.session_state.selected_metric = None
if "selected_date" not in st.session_state:
    st.session_state.selected_date = None
if "selected_dimensions" not in st.session_state:
    st.session_state.selected_dimensions = []
if "anomaly_window" not in st.session_state:
    st.session_state.anomaly_window = None
if "anomaly_points" not in st.session_state:
    st.session_state.anomaly_points = []
if "investigation" not in st.session_state:
    st.session_state.investigation = None
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []


def log_audit(step: str, action: str, details: str):
    """Appends an event to the investigation audit log."""
    entry = AuditLogEntry(
        timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
        step=step,
        action=action,
        details=details
    )
    st.session_state.audit_trail.append(entry)


# Auto-load default incident dataset on first launch for immediate out-of-the-box readiness
if st.session_state.current_df is None:
    ecom_p, saas_p = ensure_sample_data_files()
    df_loaded = pd.read_csv(ecom_p)
    st.session_state.current_df = df_loaded
    st.session_state.dataset_name = ecom_p.name
    st.session_state.profile = profile_dataset(df_loaded)
    if st.session_state.profile.candidate_metric_cols:
        st.session_state.selected_metric = st.session_state.profile.candidate_metric_cols[0]
    if st.session_state.profile.candidate_date_cols:
        st.session_state.selected_date = st.session_state.profile.candidate_date_cols[0]
    st.session_state.selected_dimensions = st.session_state.profile.candidate_segment_cols
    log_audit("Ingestion", "Auto-Init", f"Auto-loaded default incident '{ecom_p.name}' ({len(df_loaded)} rows)")


# ---------------- SIDEBAR NAVIGATION & DATA LOADER ----------------
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
            <span style='font-size: 2.2rem;'>🕵️</span>
            <div>
                <h2 style='margin:0; font-size: 1.3rem; color: #60a5fa;'>Data Detective</h2>
                <span style='font-size: 0.75rem; color: #94a3b8;'>Automated Root-Cause Analysis</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("1. Data Ingestion")

    data_source_mode = st.radio(
        "Source",
        ["Use Pre-built Incident", "Upload CSV / Excel"],
        index=0,
        horizontal=True
    )

    if data_source_mode == "Use Pre-built Incident":
        sample_choice = st.selectbox(
            "Select Scenario",
            [
                "E-Commerce: Post-Release Checkout Drop",
                "SaaS: Enterprise Customer Churn Spike"
            ]
        )
        if st.button("🚀 Load Sample Dataset", use_container_width=True):
            ecom_p, saas_p = ensure_sample_data_files()
            target_path = ecom_p if "E-Commerce" in sample_choice else saas_p
            df_loaded = pd.read_csv(target_path)
            st.session_state.current_df = df_loaded
            st.session_state.dataset_name = target_path.name
            st.session_state.profile = profile_dataset(df_loaded)
            # Default selections
            if st.session_state.profile.candidate_metric_cols:
                st.session_state.selected_metric = st.session_state.profile.candidate_metric_cols[0]
            if st.session_state.profile.candidate_date_cols:
                st.session_state.selected_date = st.session_state.profile.candidate_date_cols[0]
            st.session_state.selected_dimensions = st.session_state.profile.candidate_segment_cols
            st.session_state.anomaly_window = None
            st.session_state.investigation = None
            st.session_state.audit_trail = []
            log_audit("Ingestion", "Load Sample", f"Loaded sample dataset '{target_path.name}' ({len(df_loaded)} rows)")
            st.success(f"Loaded {target_path.name}!")
            st.rerun()

    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_excel(uploaded_file)
                st.session_state.current_df = df_up
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.profile = profile_dataset(df_up)
                if st.session_state.profile.candidate_metric_cols:
                    st.session_state.selected_metric = st.session_state.profile.candidate_metric_cols[0]
                if st.session_state.profile.candidate_date_cols:
                    st.session_state.selected_date = st.session_state.profile.candidate_date_cols[0]
                st.session_state.selected_dimensions = st.session_state.profile.candidate_segment_cols
                st.session_state.anomaly_window = None
                st.session_state.investigation = None
                st.session_state.audit_trail = []
                log_audit("Ingestion", "Upload File", f"User uploaded '{uploaded_file.name}' ({len(df_up)} rows)")
                st.success(f"Uploaded {uploaded_file.name} successfully!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    st.markdown("---")
    st.subheader("2. Investigation Workflow")
    nav_tab = st.radio(
        "Workflow Step",
        [
            "📁 Data Profile & Configuration",
            "⚡ Anomaly Detection",
            "🔬 Hypotheses & Evidence",
            "🕸️ Evidence Graph & Decomposition",
            "🔮 Scenarios & Audit Timeline",
            "📑 Investigation Report & History"
        ]
    )

    st.markdown("---")
    st.caption("Data Detective v1.0 • Evidence-Driven Diagnostic Engine")


# ---------------- TAB 1: DATA PROFILE & CONFIGURATION ----------------
if nav_tab == "📁 Data Profile & Configuration":
    st.title("📁 Dataset Profiling & Anomaly Targeting")
    st.markdown("Automatic column type inference, missing value inspection, and candidate dimension detection.")

    if st.session_state.current_df is None:
        st.info("👈 Please load a sample incident or upload your dataset from the sidebar to begin.")
        st.stop()

    df = st.session_state.current_df
    prof = st.session_state.profile

    # KPI summary bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Observations</div>
                <div class='metric-value'>{prof.n_rows:,}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>Rows in dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Dimensions & Attributes</div>
                <div class='metric-value'>{prof.n_cols}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>Columns analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Duplicate Records</div>
                <div class='metric-value'>{prof.duplicate_rows}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>{prof.duplicate_pct:.1f}% duplication rate</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        total_missing = sum(col.missing_count for col in prof.columns.values())
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Missing Values</div>
                <div class='metric-value'>{total_missing}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>Across all fields</div>
            </div>
        """, unsafe_allow_html=True)

    # Configuration Form
    st.subheader("🎯 Investigative Target Configuration")
    conf_col1, conf_col2, conf_col3 = st.columns(3)

    with conf_col1:
        metric_opts = prof.candidate_metric_cols if prof.candidate_metric_cols else list(df.columns)
        def_idx = 0
        if st.session_state.selected_metric in metric_opts:
            def_idx = metric_opts.index(st.session_state.selected_metric)
        selected_m = st.selectbox("Primary Target Metric (Y)", metric_opts, index=def_idx)
        st.session_state.selected_metric = selected_m

    with conf_col2:
        date_opts = ["None (Tabular Mode)"] + prof.candidate_date_cols
        d_idx = 0
        if st.session_state.selected_date and st.session_state.selected_date in date_opts:
            d_idx = date_opts.index(st.session_state.selected_date)
        selected_d = st.selectbox("Temporal / Date Column", date_opts, index=d_idx)
        st.session_state.selected_date = None if selected_d == "None (Tabular Mode)" else selected_d

    with conf_col3:
        seg_opts = [c for c in prof.candidate_segment_cols if c != selected_m]
        def_segs = [s for s in st.session_state.selected_dimensions if s in seg_opts]
        selected_segs = st.multiselect("Candidate Segment Dimensions (X)", seg_opts, default=def_segs)
        st.session_state.selected_dimensions = selected_segs

    # Column Schema Inspection Table
    st.subheader("📋 Inferred Schema & Data Health")
    schema_rows = []
    for col_name, c_prof in prof.columns.items():
        schema_rows.append({
            "Column": col_name,
            "Inferred Type": c_prof.inferred_type.value.upper(),
            "Raw Dtype": c_prof.raw_dtype,
            "Missing %": f"{c_prof.missing_pct}% ({c_prof.missing_count})",
            "Uniques": c_prof.unique_count,
            "Outliers (IQR)": f"{c_prof.outlier_count} ({c_prof.outlier_pct}%)",
            "Mean / Top Value": str(c_prof.mean_val if c_prof.mean_val is not None else list(c_prof.top_categories.keys())[:2])
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    with st.expander("🔍 Preview Raw Dataset Sample (First 15 Rows)"):
        st.dataframe(df.head(15), use_container_width=True)


# ---------------- TAB 2: ANOMALY DETECTION ----------------
elif nav_tab == "⚡ Anomaly Detection":
    st.title("⚡ Anomaly & Changepoint Detection")
    st.markdown("Locates critical metric shifts, structural changepoints, or multivariate anomaly clusters.")

    if st.session_state.current_df is None:
        st.warning("Please load a dataset first.")
        st.stop()

    df = st.session_state.current_df
    metric = st.session_state.selected_metric
    date_c = st.session_state.selected_date

    # Top Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        st.markdown(f"Investigating Metric: **`{metric}`** | Temporal Axis: **`{date_c if date_c else 'Tabular Index'}`**")
    with ctrl_col2:
        method_choice = st.selectbox("Detection Engine", ["Auto-Detect Optimal", "Rolling Z-Score (Temporal)", "Isolation Forest (Multivariate)"])
    with ctrl_col3:
        run_detect = st.button("⚡ Detect Anomaly Window", use_container_width=True, type="primary")

    if run_detect or st.session_state.anomaly_window is None:
        with st.spinner("Executing statistical changepoint and anomaly detection pass..."):
            clean_df = clean_and_prepare_dataset(df, st.session_state.profile, date_c)
            st.session_state.current_df = clean_df

            if method_choice == "Isolation Forest (Multivariate)":
                from data_detective.anomaly_detector import detect_isolation_forest_anomalies
                pts, window = detect_isolation_forest_anomalies(clean_df, metric_col=metric, date_col=date_c)
            elif method_choice == "Rolling Z-Score (Temporal)" and date_c:
                pts, window = detect_rolling_anomalies(clean_df, metric_col=metric, date_col=date_c)
            else:
                pts, window = detect_optimal_anomaly_window(clean_df, metric_col=metric, date_col=date_c)

            st.session_state.anomaly_points = pts
            st.session_state.anomaly_window = window
            log_audit(
                "Anomaly Detection",
                "Scan Completed",
                f"Identified anomaly window using {window.detection_method}. Baseline: {window.baseline_mean:.4f} -> Anomaly: {window.anomaly_mean:.4f} ({window.relative_pct_change:+.1f}%)"
            )

    window = st.session_state.anomaly_window

    if window:
        # Anomaly Window Stats
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Baseline Mean</div>
                    <div class='metric-value'>{window.baseline_mean:.4f}</div>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>{window.baseline_period_label}</div>
                </div>
            """, unsafe_allow_html=True)
        with w2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Anomaly Mean</div>
                    <div class='metric-value'>{window.anomaly_mean:.4f}</div>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>{window.anomaly_period_label}</div>
                </div>
            """, unsafe_allow_html=True)
        with w3:
            delta_class = "metric-delta-neg" if window.direction == "drop" else "metric-delta-pos"
            delta_sym = "▼" if window.direction == "drop" else "▲"
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Relative Shift</div>
                    <div class='metric-value {delta_class}'>{delta_sym} {abs(window.relative_pct_change):.2f}%</div>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>Absolute delta: {window.absolute_delta:+.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        with w4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Anomaly Severity</div>
                    <div class='metric-value' style='color: #f59e0b;'>HIGH</div>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>Method: {window.detection_method}</div>
                </div>
            """, unsafe_allow_html=True)

        # Time series / Anomaly plot
        if date_c and date_c in df.columns:
            # Aggregate by date for clean visualization
            daily = df.groupby(date_c)[metric].mean().reset_index()
            fig = px.line(
                daily,
                x=date_c,
                y=metric,
                title=f"<b>Time-Series Trajectory of '{metric}' with Anomaly Window</b>",
                labels={date_c: "Date", metric: metric},
                template="plotly_white"
            )
            fig.update_traces(line=dict(color="#2563eb", width=2.8))

            # Add flagged points
            anom_pts = [p for p in st.session_state.anomaly_points if p.is_anomaly]
            if anom_pts:
                pts_df = pd.DataFrame([{date_c: p.index_or_date, metric: p.metric_value, "severity": p.severity} for p in anom_pts])
                fig.add_trace(go.Scatter(
                    x=pts_df[date_c],
                    y=pts_df[metric],
                    mode="markers",
                    name="Flagged Anomalies",
                    marker=dict(color="#dc2626", size=10, symbol="x")
                ))

            # Add baseline and anomaly mean reference lines
            fig.add_hline(y=window.baseline_mean, line_dash="dash", line_color="#16a34a", annotation_text="Baseline Mean", annotation_font_color="#16a34a")
            fig.add_hline(y=window.anomaly_mean, line_dash="dash", line_color="#dc2626", annotation_text="Anomaly Window Mean", annotation_font_color="#dc2626")

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=450,
                font=dict(color="#0f172a"),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Distribution plot for Tabular Isolation Forest
            base_vals = df.loc[window.baseline_indices, metric]
            anom_vals = df.loc[window.anomaly_indices, metric]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=base_vals, name="Baseline Inliers", marker_color="#16a34a", opacity=0.7))
            fig.add_trace(go.Histogram(x=anom_vals, name="Anomaly Cluster", marker_color="#dc2626", opacity=0.7))
            fig.update_layout(
                barmode="overlay",
                title=f"<b>Distribution Comparison: Baseline Cohort vs Anomaly Cluster for '{metric}'</b>",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a"),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Action banner
        st.markdown("---")
        b1, b2 = st.columns([3, 1])
        with b1:
            st.info("💡 Ready to investigate: Data Detective will decompose metric variance across all dimensions, generate testable hypotheses, run rigorous statistical tests, and isolate root causes.")
        with b2:
            if st.button("🔬 Investigate Root Causes", type="primary", use_container_width=True):
                # Trigger full investigation
                with st.spinner("Generating hypotheses and executing statistical tests..."):
                    hyps = generate_and_test_hypotheses(
                        df=df,
                        profile=st.session_state.profile,
                        anomaly_window=window
                    )
                    all_decomp = decompose_all_dimensions(
                        df=df,
                        metric_col=metric,
                        dimension_cols=st.session_state.selected_dimensions,
                        anomaly_window=window
                    )
                    # Flatten top contributions
                    flat_contribs = []
                    for d_contribs in all_decomp.values():
                        flat_contribs.extend(d_contribs)
                    flat_contribs.sort(key=lambda c: abs(c.absolute_contribution), reverse=True)

                    # Top counterfactuals
                    scenarios = generate_top_counterfactual_scenarios(
                        df=df,
                        anomaly_window=window,
                        top_contributions=flat_contribs
                    )

                    # Generate deterministic narrative
                    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    report_text = generate_deterministic_narrative(
                        run_title=f"Investigation of {metric} on {st.session_state.dataset_name}",
                        metric_col=metric,
                        anomaly_window_dict=window.to_dict(),
                        top_hypotheses=hyps,
                        top_contributions=flat_contribs,
                        top_scenarios=scenarios
                    )

                    run_obj = InvestigationRun(
                        id=run_id,
                        title=f"Investigation: {metric} ({window.direction.upper()})",
                        created_at=now_str,
                        dataset_name=st.session_state.dataset_name,
                        row_count=len(df),
                        metric_col=metric,
                        date_col=date_c,
                        anomaly_window=window,
                        hypotheses=hyps,
                        contributions=flat_contribs,
                        counterfactuals=scenarios,
                        audit_trail=st.session_state.audit_trail.copy(),
                        executive_summary=report_text
                    )
                    st.session_state.investigation = run_obj
                    save_investigation(run_obj)
                    log_audit("Investigation Engine", "Analysis Completed", f"Generated {len(hyps)} hypotheses. Saved run {run_id}.")
                    st.success("Investigation complete! Navigate to 'Hypotheses & Evidence' to view the breakdown.")


# ---------------- TAB 3: HYPOTHESES & EVIDENCE ----------------
elif nav_tab == "🔬 Hypotheses & Evidence":
    st.title("🔬 Formulated Hypotheses & Statistical Proof")
    st.markdown("Rigorously tested scientific hypotheses with effect sizes, p-values, sample sizes, and non-causal guardrails.")

    if st.session_state.investigation is None:
        st.warning("Please detect an anomaly and run the investigation first from the 'Anomaly Detection' tab.")
        st.stop()

    inv: InvestigationRun = st.session_state.investigation
    hyps = inv.hypotheses

    # Epistemic Banner
    st.markdown("""
        <div class='epistemic-banner'>
            <span style='color: #1d4ed8; font-weight: 700;'>Epistemic Modesty Rule:</span>
            <span style='color: #1e3a8a; font-size: 0.95rem;'>
                Candidate explanations reflect observational associations and rate-mix contributions. Correlation does not imply direct causation; unmeasured confounding factors must always be considered.
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Filter controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        status_filter = st.selectbox(
            "Filter by Test Status",
            ["All", "Supported (High Confidence)", "Weak / Marginal", "Rejected", "Inconclusive"]
        )

    filtered_hyps = hyps
    if status_filter == "Supported (High Confidence)":
        filtered_hyps = [h for h in hyps if h.status == HypothesisStatus.SUPPORTED]
    elif status_filter == "Weak / Marginal":
        filtered_hyps = [h for h in hyps if h.status == HypothesisStatus.WEAK]
    elif status_filter == "Rejected":
        filtered_hyps = [h for h in hyps if h.status == HypothesisStatus.REJECTED]
    elif status_filter == "Inconclusive":
        filtered_hyps = [h for h in hyps if h.status == HypothesisStatus.INCONCLUSIVE]

    st.write(f"Showing **{len(filtered_hyps)}** hypotheses:")

    for h in filtered_hyps:
        # Determine status badge
        badge_class = f"status-badge-{h.status.value}"
        badge_label = f"● {h.status.value.upper()}"

        with st.container():
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <span style='font-size: 1.15rem; font-weight: 700; color: #0f172a;'>
                            Rank #{h.rank}: {h.title}
                        </span>
                        <div>
                            <span class='{badge_class}'>{badge_label}</span>
                            <span style='margin-left: 10px; font-weight: 700; color: #475569;'>Plausibility: {h.plausibility_score:.1f}%</span>
                        </div>
                    </div>
                    <div style='color: #334155; font-size: 0.95rem; margin-bottom: 8px; line-height: 1.5;'>
                        {h.description}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Evidence Breakdown Accordion
            with st.expander(f"📊 Detailed Statistical Evidence & Test Log for #{h.rank} ({h.id})"):
                t1, t2 = st.columns([3, 2])
                with t1:
                    st.markdown("**Empirical Evidence Tests:**")
                    for ev in h.evidence:
                        st.markdown(f"- **{ev.test_name}**: Statistic = `{ev.test_statistic:.4f}`, **p = {ev.p_value:.4e}**")
                        st.markdown(f"  - Effect Size: **{ev.effect_size_name} = {ev.effect_size_value:+.3f}**")
                        st.markdown(f"  - Sample Sizes: Baseline N={ev.sample_size_baseline}, Anomaly N={ev.sample_size_anomaly}")
                        st.markdown(f"  - *Interpretation:* {ev.interpretation}")

                    st.markdown("**Epistemic Caveats:**")
                    for cav in h.caveats:
                        st.caption(f"⚠️ {cav}")

                with t2:
                    # Segment comparison visualization if applicable
                    if h.dimension and h.dimension in st.session_state.current_df.columns:
                        df_curr = st.session_state.current_df
                        base_idx = inv.anomaly_window.baseline_indices
                        anom_idx = inv.anomaly_window.anomaly_indices

                        df_plot = df_curr.copy()
                        df_plot["Period"] = "Other"
                        df_plot.loc[base_idx, "Period"] = "Baseline"
                        df_plot.loc[anom_idx, "Period"] = "Anomaly Window"
                        df_plot = df_plot[df_plot["Period"] != "Other"]

                        if h.dimension_value:
                            df_plot = df_plot[df_plot[h.dimension] == h.dimension_value]

                        if len(df_plot) > 5:
                            fig_box = px.box(
                                df_plot,
                                x="Period",
                                y=inv.metric_col,
                                color="Period",
                                title=f"<b>Distribution of {inv.metric_col} ({h.dimension}={h.dimension_value or 'All'})</b>",
                                color_discrete_map={"Baseline": "#16a34a", "Anomaly Window": "#dc2626"},
                                template="plotly_white"
                            )
                            fig_box.update_layout(
                                height=280,
                                margin=dict(l=10, r=10, t=40, b=10),
                                paper_bgcolor="#ffffff",
                                plot_bgcolor="#ffffff",
                                font=dict(color="#0f172a")
                            )
                            st.plotly_chart(fig_box, use_container_width=True)


# ---------------- TAB 4: EVIDENCE GRAPH & DECOMPOSITION ----------------
elif nav_tab == "🕸️ Evidence Graph & Decomposition":
    st.title("🕸️ Evidence Graph & Rate-Mix Decomposition")
    st.markdown("Visualizing multi-dimensional causal candidate pathways and additive segment variance decomposition.")

    if st.session_state.investigation is None:
        st.warning("Please run an investigation first.")
        st.stop()

    inv: InvestigationRun = st.session_state.investigation

    # 1. SEGMENT CONTRIBUTION WATERFALL
    st.subheader("📊 Rate-Mix Oaxaca-Blinder Decomposition")
    st.markdown("Partitions total metric delta into within-segment degradation vs cross-segment mix shift.")

    top_c = inv.contributions[:8]
    if top_c:
        # Waterfall Chart
        labels = [f"{c.dimension}={c.segment_value}" for c in top_c]
        values = [c.pct_of_total_delta for c in top_c]

        wf_fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=["#dc2626" if v > 0 and inv.anomaly_window.direction == "spike" or (v < 0 and inv.anomaly_window.direction == "drop") else "#2563eb" for v in values],
            text=[f"{v:+.1f}%" for v in values],
            textposition="auto"
        ))
        wf_fig.update_layout(
            title=f"<b>% Share of Total {inv.metric_col} Shift Explained by Segment Performance Drop</b>",
            xaxis_title="Segment Cohort",
            yaxis_title="% Contribution to Delta",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
            height=380,
            margin=dict(l=20, r=20, t=50, b=30)
        )
        st.plotly_chart(wf_fig, use_container_width=True)

        # Primary Culprit Banner
        primary_culprits = [c for c in top_c if c.is_primary_culprit]
        if primary_culprits:
            p = primary_culprits[0]
            st.markdown(f"""
                <div class='culprit-banner'>
                    <div style='color: #b91c1c; font-weight: 700; font-size: 1.1rem; margin-bottom: 4px;'>⚠️ PRIMARY ANOMALY DRIVER IDENTIFIED:</div>
                    <div style='color: #1e293b; font-size: 1.0rem;'>
                        <b>{p.dimension} = '{p.segment_value}'</b> contributed <b>{p.pct_of_total_delta:+.1f}%</b> of the entire metric degradation.
                        Segment conversion crashed from <b>{p.baseline_rate:.4f}</b> to <b>{p.anomaly_rate:.4f}</b>.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 2. INTERACTIVE NETWORKX EVIDENCE GRAPH
    st.subheader("🕸️ Multi-Layer Evidence Relationship Graph")
    st.markdown("Directed Network linking: **Target Metric ➔ Dimensions ➔ Segments/Covariates ➔ Tested Hypotheses ➔ Statistical Proof**.")

    G = build_evidence_graph(
        metric_name=inv.metric_col,
        hypotheses=inv.hypotheses[:10],
        contributions=inv.contributions
    )
    graph_fig = plot_evidence_graph_plotly(G)
    st.plotly_chart(graph_fig, use_container_width=True)


# ---------------- TAB 5: SCENARIOS & AUDIT TIMELINE ----------------
elif nav_tab == "🔮 Scenarios & Audit Timeline":
    st.title("🔮 Counterfactual Scenarios & Audit Trail")
    st.markdown("Simulating 'What-If' remediation outcomes alongside an audit log of automated investigative decisions.")

    if st.session_state.investigation is None:
        st.warning("Please run an investigation first.")
        st.stop()

    inv: InvestigationRun = st.session_state.investigation

    # 1. COUNTERFACTUAL WHAT-IF SIMULATOR
    st.subheader("🧪 'What-If' Segment Restoration Simulator")
    st.markdown("Calculates expected metric recovery if specific segment issues were resolved to historical baseline rates.")

    scenarios = inv.counterfactuals
    if scenarios:
        selected_sc = st.selectbox(
            "Select Remediation Candidate",
            options=range(len(scenarios)),
            format_func=lambda i: f"{scenarios[i].scenario_name} (Est. Recovery: +{scenarios[i].recovery_percentage:.1f}%)"
        )
        sc = scenarios[selected_sc]

        # Interactive Restoration Slider
        fix_eff = st.slider("Target Remediation Effectiveness (%)", min_value=10, max_value=100, value=100, step=10)

        # Scale restoration
        scaled_delta = sc.restoration_delta * (fix_eff / 100.0)
        scaled_metric = sc.observed_anomaly_metric + scaled_delta
        scaled_recovery = sc.recovery_percentage * (fix_eff / 100.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Observed Anomaly Rate", f"{sc.observed_anomaly_metric:.4f}")
        with c2:
            st.metric(f"Projected Rate ({fix_eff}% Remediation)", f"{scaled_metric:.4f}", delta=f"{scaled_delta:+.4f}")
        with c3:
            st.metric("Estimated Total Recovery", f"{scaled_recovery:.1f}%")

        # Scenario visualization with 95% Confidence Interval error bar
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            x=["Historical Baseline", "Observed Anomaly", f"Restored ({fix_eff}% Fix)"],
            y=[inv.anomaly_window.baseline_mean, sc.observed_anomaly_metric, scaled_metric],
            marker_color=["#16a34a", "#dc2626", "#2563eb"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0, 0, (sc.ci_upper - sc.counterfactual_metric) * (fix_eff / 100.0)],
                arrayminus=[0, 0, (sc.counterfactual_metric - sc.ci_lower) * (fix_eff / 100.0)]
            )
        ))
        fig_sc.update_layout(
            title=f"<b>Counterfactual Impact with 95% CI Error Bounds: {sc.scenario_name}</b>",
            yaxis_title=inv.metric_col,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
            height=350
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        with st.expander("⚠️ Epistemic Assumptions & Causal Caveats"):
            for a in sc.assumptions_and_caveats:
                st.caption(f"• {a}")

    # 2. AUDIT TRAIL TIMELINE
    st.markdown("---")
    st.subheader("📜 Investigation Audit Trail & Timeline")
    st.markdown("Detailed chronological record of analytical tests performed and decisions made.")

    for entry in st.session_state.audit_trail:
        st.markdown(f"""
            <div class='audit-step'>
                <span style='color: #60a5fa; font-weight: 600;'>[{entry.timestamp}] {entry.step}</span>
                <span style='color: #94a3b8;'>➔ {entry.action}</span>
                <div style='color: #cbd5e1; font-size: 0.9rem;'>{entry.details}</div>
            </div>
        """, unsafe_allow_html=True)


# ---------------- TAB 6: INVESTIGATION REPORT & HISTORY ----------------
elif nav_tab == "📑 Investigation Report & History":
    st.title("📑 Investigation Report & Run History")
    st.markdown("Downloadable executive report, optional AI synthesis, and SQLite past investigations store.")

    if st.session_state.investigation is None:
        st.info("No active investigation in memory. You can inspect saved investigations from SQLite below:")
    else:
        inv: InvestigationRun = st.session_state.investigation

        st.subheader("Executive Diagnostic Report")

        # Optional LLM synthesis card
        with st.expander("🤖 Optional Groq AI Executive Synthesis (Strictly Grounded in Statistics)"):
            st.caption("Powered by Groq high-speed inference. The LLM receives ONLY verified statistical findings and is strictly forbidden from hallucinations.")
            
            g_col1, g_col2 = st.columns([2, 1])
            with g_col1:
                env_groq_key = os.environ.get("GROQ_API_KEY", "")
                user_llm_key = st.text_input("Groq API Key (reads from .env by default)", value=env_groq_key, type="password")
            with g_col2:
                selected_groq_model = st.selectbox(
                    "Groq Model",
                    ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"],
                    index=0
                )

            if st.button("⚡ Generate AI Executive Synthesis (Groq)", type="primary"):
                with st.spinner(f"Synthesizing empirical findings with Groq ({selected_groq_model})..."):
                    llm_text = generate_llm_enhanced_summary(
                        inv.to_dict(),
                        api_key=user_llm_key,
                        model=selected_groq_model
                    )
                    if llm_text:
                        inv.llm_explanation = llm_text
                        save_investigation(inv)
                        st.success("Groq AI synthesis completed!")

        if inv.llm_explanation:
            st.markdown("### 🤖 AI Executive Synthesis")
            st.markdown(inv.llm_explanation)
            st.markdown("---")

        # Render deterministic markdown report
        st.markdown(inv.executive_summary)

        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 Download Markdown Report (.md)",
                data=inv.executive_summary,
                file_name=f"DataDetective_Report_{inv.id}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with d_col2:
            html_rep = export_report_html(inv.executive_summary)
            st.download_button(
                label="🌐 Download Standalone HTML Report (.html)",
                data=html_rep,
                file_name=f"DataDetective_Report_{inv.id}.html",
                mime="text/html",
                use_container_width=True
            )

    # SQLite Investigation History
    st.markdown("---")
    st.subheader("🗄️ SQLite Investigation History Store")
    saved_runs = list_investigations()

    if not saved_runs:
        st.caption("No past investigations saved in SQLite database yet.")
    else:
        st.dataframe(pd.DataFrame(saved_runs), use_container_width=True)

        hist_col1, hist_col2 = st.columns([3, 1])
        with hist_col1:
            selected_run_id = st.selectbox("Select Past Investigation to Load", [r["id"] for r in saved_runs])
        with hist_col2:
            st.write("")
            st.write("")
            if st.button("📂 Load Selected Run", use_container_width=True):
                loaded_dict = load_investigation(selected_run_id)
                if loaded_dict:
                    # Restore session state
                    st.session_state.dataset_name = loaded_dict["dataset_name"]
                    st.session_state.selected_metric = loaded_dict["metric_col"]
                    st.session_state.selected_date = loaded_dict["date_col"]

                    # Reconstruct minimal investigation run
                    anom_win_data = loaded_dict["anomaly_window"]
                    anom_win = AnomalyWindow(
                        metric_col=anom_win_data["metric_col"],
                        date_col=anom_win_data["date_col"],
                        baseline_period_label=anom_win_data["baseline_period_label"],
                        anomaly_period_label=anom_win_data["anomaly_period_label"],
                        baseline_indices=list(range(anom_win_data["baseline_count"])),
                        anomaly_indices=list(range(anom_win_data["anomaly_count"])),
                        baseline_mean=anom_win_data["baseline_mean"],
                        anomaly_mean=anom_win_data["anomaly_mean"],
                        absolute_delta=anom_win_data["absolute_delta"],
                        relative_pct_change=anom_win_data["relative_pct_change"],
                        direction=anom_win_data["direction"],
                        detection_method=anom_win_data["detection_method"]
                    )

                    # Reconstruct hypotheses
                    reconstructed_hyps = []
                    for h_data in loaded_dict["hypotheses"]:
                        reconstructed_hyps.append(
                            Hypothesis(
                                id=h_data["id"],
                                title=h_data["title"],
                                hypothesis_type=HypothesisType(h_data["hypothesis_type"]),
                                target_metric=h_data["target_metric"],
                                dimension=h_data["dimension"],
                                dimension_value=h_data["dimension_value"],
                                description=h_data["description"],
                                rationale=h_data["rationale"],
                                test_plan=h_data["test_plan"],
                                status=HypothesisStatus(h_data["status"]),
                                plausibility_score=h_data["plausibility_score"],
                                rank=h_data["rank"],
                                caveats=h_data.get("caveats", [])
                            )
                        )

                    # Reconstruct contributions
                    reconstructed_contribs = []
                    for c_data in loaded_dict["contributions"]:
                        reconstructed_contribs.append(
                            SegmentContribution(
                                dimension=c_data["dimension"],
                                segment_value=c_data["segment_value"],
                                baseline_volume=c_data["baseline_volume"],
                                anomaly_volume=c_data["anomaly_volume"],
                                baseline_share=c_data["baseline_share"],
                                anomaly_share=c_data["anomaly_share"],
                                baseline_rate=c_data["baseline_rate"],
                                anomaly_rate=c_data["anomaly_rate"],
                                rate_change=c_data["rate_change"],
                                absolute_contribution=c_data["absolute_contribution"],
                                pct_of_total_delta=c_data["pct_of_total_delta"],
                                is_primary_culprit=c_data["is_primary_culprit"]
                            )
                        )

                    reconstructed_run = InvestigationRun(
                        id=loaded_dict["id"],
                        title=loaded_dict["title"],
                        created_at=loaded_dict["created_at"],
                        dataset_name=loaded_dict["dataset_name"],
                        row_count=loaded_dict["row_count"],
                        metric_col=loaded_dict["metric_col"],
                        date_col=loaded_dict["date_col"],
                        anomaly_window=anom_win,
                        hypotheses=reconstructed_hyps,
                        contributions=reconstructed_contribs,
                        counterfactuals=[],
                        audit_trail=[],
                        executive_summary=loaded_dict["executive_summary"],
                        llm_explanation=loaded_dict.get("llm_explanation")
                    )
                    st.session_state.investigation = reconstructed_run
                    st.session_state.anomaly_window = anom_win
                    st.success(f"Loaded investigation {selected_run_id}!")
                    st.rerun()
