# 🕵️ Data Detective

> **Evidence-Driven Automated Root-Cause Analysis Platform**  
> *Diagnose metric drops, spikes, and anomalies using rigorous statistical testing, rate-mix decomposition, and counterfactual simulation — without false causal claims or generic AutoEDA dashboards.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-success.svg)](tests/)
[![UI](https://img.shields.io/badge/interface-Streamlit-FF4B4B.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Why Data Detective?

Most automated data tools fall into two traps:
1. **Generic AutoEDA dashboards** (e.g. ydata-profiling) that dump hundreds of histograms without identifying *why* a business metric shifted.
2. **Naive LLM wrappers** that hallucinate explanations or claim causation from raw statistical noise.

**Data Detective** is built on the rigorous investigative methodology used by top data science teams at Google, Meta, and Netflix:
- **Scientific Hypothesis Engine**: Automatically generates falsifiable hypotheses across segment degradations, mix shifts (Simpson's paradox), covariate drift, and multi-dimensional interactions.
- **Parametric & Non-Parametric Proof**: Evaluates hypotheses using Welch's t-tests, Mann-Whitney U, Chi-Square contingency tables, Cohen's d, Cramer's V, and Pearson/Spearman correlation.
- **Oaxaca-Blinder / Kitagawa Decomposition**: Mathematically separates *within-segment performance drops* (rate effect) from *traffic composition shifts* (mix effect).
- **NetworkX Evidence Graph**: Maps the causal candidate topology connecting Target Metrics $\rightarrow$ Dimensions $\rightarrow$ Segments $\rightarrow$ Tested Hypotheses $\rightarrow$ Statistical Proof.
- **Counterfactual Scenario Simulation**: Projects metric recovery under historical baseline conditions with 95% confidence intervals and explicit SUTVA/stationarity caveats.
- **Epistemic Modesty**: Labels findings as *Observational Associations* with confidence scoring, never conflating correlation with causation.
- **Deterministic Statistical NLG**: Writes clear executive investigation memos from empirical findings without requiring an LLM API (with optional LLM enhancement available).

---

## 🏗️ Architecture & Investigation Pipeline

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ 1. Data Ingest  │ ────► │ 2. Profiling &  │ ────► │ 3. Anomaly &    │
│  CSV / Excel /  │       │ Type Inference  │       │ Changepoint Det │
│ Sample Incident │       │ Missing/Outliers│       │ Rolling Z / IsoF│
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ 6. Oaxaca-      │ ◄──── │ 5. Statistical  │ ◄──── │ 4. Hypothesis   │
│ Blinder Decomp  │       │ Evidence Testing│       │ Engine          │
│ Rate-Mix Matrix │       │ Welch, MWU, Chi2│       │ Degrade, Mix    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │
         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ 7. NetworkX     │ ────► │ 8. Baseline     │ ────► │ 9. Executive    │
│ Evidence Graph  │       │ Counterfactual  │       │ Report & SQLite │
│ Interactive Plot│       │ 95% CI Bounds   │       │ Export MD/HTML  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## ⚡ Quickstart: Run in 1 Command

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/DataDet.git
cd DataDet

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
*The Streamlit web interface will open automatically in your browser at `http://localhost:8501`.*

---

## 🧪 Running the Automated Test Suite

Data Detective includes a complete suite of unit and integration tests covering profilers, anomaly detectors, statistical tests, rate-mix decomposition, and end-to-end pipeline execution:

```bash
python -m pytest -v
```

Output:
```text
tests/test_anomaly.py::test_detect_rolling_anomalies PASSED
tests/test_anomaly.py::test_detect_isolation_forest_anomalies PASSED
tests/test_contribution.py::test_calculate_segment_contributions PASSED
tests/test_end_to_end.py::test_end_to_end_investigation PASSED
tests/test_profiler.py::test_infer_column_type PASSED
tests/test_profiler.py::test_outlier_detection_iqr PASSED
tests/test_profiler.py::test_profile_dataset PASSED
tests/test_stats.py::test_compute_cohens_d PASSED
tests/test_stats.py::test_two_sample_difference_calc PASSED
tests/test_stats.py::test_categorical_distribution_shift_calc PASSED

============================= 10 passed in 5.83s ==============================
```

---

## 📦 Built-In Realistic Incident Datasets

Data Detective works immediately out of the box with built-in business incident datasets:

### 1. E-Commerce Checkout Incident (`data/sample_ecommerce_incident.csv`)
- **Context:** An e-commerce platform experienced a sudden conversion drop.
- **Planted Root Cause:** On October 14th, mobile release `v2.4` introduced a payment gateway timeout bug strictly affecting `device == 'iOS'` in `region == 'Europe'`.
- **Observations:** 5,000 sessions with fields `session_id`, `date`, `device`, `region`, `channel`, `app_version`, `checkout_latency_ms`, `converted`, and `order_value`.
- **Diagnostic Result:** Data Detective identifies the changepoint on Oct 14, ranks `device=iOS × region=Europe` as the #1 supported hypothesis ($p < 10^{-10}$), calculates iOS Europe's 82% contribution to the total revenue loss, and models full restoration recovery.

### 2. SaaS Customer Churn Spike (`data/sample_saas_churn.csv`)
- **Context:** Enterprise account churn surged after an infrastructure degradation.
- **Planted Root Cause:** `plan_tier == 'Enterprise'` suffered API latency surges (>380ms) and pricing tier restructuring.
- **Diagnostic Result:** Ranks covariate drift in `api_latency_ms` and enterprise cohort degradation as primary drivers.

---

## 🔬 Core Investigative Methodologies

### 1. Rate-Mix Decomposition (Kitagawa-Oaxaca-Blinder)
Partitions metric difference $\Delta M$ into within-cohort performance drop vs volume mix shift:
$$\Delta M = \sum_{s} \left[ \bar{w}_s \Delta m_s + \bar{m}_s \Delta w_s \right]$$
- $\bar{w}_s \Delta m_s$: **Rate Effect** (did the segment degrade?)
- $\bar{m}_s \Delta w_s$: **Mix Shift Effect** (did traffic shift into lower-performing cohorts?)

### 2. Hypothesis Status Classification
- `SUPPORTED`: Statistically significant ($p < 0.05$), non-negligible effect size ($|d| \ge 0.20$ or $|V| \ge 0.10$), and consistent directional shift.
- `WEAK`: Marginally significant ($0.05 \le p < 0.10$) or small effect size ($|d| < 0.20$).
- `REJECTED`: Null hypothesis fails to reject ($p \ge 0.10$) or contrary effect direction.
- `INCONCLUSIVE`: Sub-sample size $< 10$ or high variance ambiguity.

### 3. Historical-Baseline Counterfactual Simulation
Simulates the expected global metric recovery if segment $s$ is restored to historical pre-incident baseline:
$$M_{\text{counterfactual}} = M_{\text{observed}} + w_s \cdot (\bar{m}_{s,\text{baseline}} - \bar{m}_{s,\text{anomaly}})$$
Accompanied by 95% Confidence Interval error bounds and explicit assumptions:
- **SUTVA (Stable Unit Treatment Value Assumption)**
- **Baseline Stationarity**
- **Partial Equilibrium / Non-Interference**

---

## 📁 Repository Structure

```
DataDet/
├── app.py                      # Streamlit interactive application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Optional LLM API key template
├── README.md                   # System documentation & manual
├── data_detective/             # Core Diagnostic Package
│   ├── __init__.py
│   ├── schemas.py              # Typed dataclasses & Pydantic-style models
│   ├── profiler.py             # Schema inference, missingness, IQR outliers
│   ├── anomaly_detector.py     # Rolling Z-score, changepoint, Isolation Forest
│   ├── stats_tester.py         # Welch's t-test, MWU, Chi-Square, Cohen's d, Cramer's V
│   ├── contribution.py         # Rate-mix Oaxaca-Blinder variance decomposition
│   ├── hypothesis_engine.py    # Automated testable hypothesis generation & scoring
│   ├── evidence_graph.py       # NetworkX DAG & Plotly network visualizer
│   ├── counterfactual.py       # Baseline-adjusted "What-If" simulator & 95% CI
│   ├── report_generator.py     # Deterministic NLG memo & optional LLM client
│   ├── storage.py              # SQLite investigation history store
│   └── sample_data.py          # Realistic incident dataset generators
├── tests/                      # Automated test suite
│   ├── test_profiler.py
│   ├── test_anomaly.py
│   ├── test_stats.py
│   ├── test_contribution.py
│   └── test_end_to_end.py
└── data/                       # Built-in incident CSV datasets
    ├── sample_ecommerce_incident.csv
    └── sample_saas_churn.csv
```

---

## 🛡️ Epistemic Guardrails & Philosophy

1. **Observational Association $\neq$ Causation**: High correlation or segment contribution does not prove physical causality. Confounding variables outside the dataset can mediate observed effects.
2. **Deterministic-First**: All p-values, effect sizes, decomposition percentages, and rankings are calculated using exact mathematics. The optional LLM is strictly used as an explanation layer and never allowed to calculate or invent data.
3. **Auditability**: Every investigation maintains a chronological audit log recording every test executed, degrees of freedom, and statistical decisions.

---

## 📄 License
This project is open-source under the MIT License.
