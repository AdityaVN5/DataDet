"""
Data Detective Investigation Report & AI Explanation Generator.
Produces high-rigor executive reports using deterministic analytical NLG
or an optional LLM synthesizing ONLY structured empirical findings.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Any

from data_detective.schemas import (
    InvestigationRun,
    Hypothesis,
    HypothesisStatus,
    SegmentContribution,
    CounterfactualResult
)


def generate_deterministic_narrative(
    run_title: str,
    metric_col: str,
    anomaly_window_dict: Dict[str, Any],
    top_hypotheses: List[Hypothesis],
    top_contributions: List[SegmentContribution],
    top_scenarios: List[CounterfactualResult]
) -> str:
    """
    Generates a deterministic, statistically precise root-cause analysis report
    directly from empirical evidence without requiring an LLM.
    """
    base_mean = anomaly_window_dict["baseline_mean"]
    anom_mean = anomaly_window_dict["anomaly_mean"]
    pct_change = anomaly_window_dict["relative_pct_change"]
    direction = anomaly_window_dict["direction"]
    method = anomaly_window_dict["detection_method"]

    supported_hyps = [h for h in top_hypotheses if h.status == HypothesisStatus.SUPPORTED]
    weak_hyps = [h for h in top_hypotheses if h.status == HypothesisStatus.WEAK]

    culprits = [c for c in top_contributions if c.is_primary_culprit]
    if not culprits and top_contributions:
        culprits = top_contributions[:2]

    # Executive Overview
    lines = [
        f"# Root-Cause Investigation Report: {run_title}",
        f"**Target Metric:** `{metric_col}` | **Investigation Engine:** Data Detective v1.0",
        "",
        "## 1. Executive Summary",
        f"Anomalous metric shift detected in **`{metric_col}`** using **{method}**. "
        f"The metric shifted from a baseline mean of **{base_mean:.4f}** to **{anom_mean:.4f}** during the anomaly window, "
        f"representing a **{abs(pct_change):.2f}% {direction}**.",
        ""
    ]

    # Candidate Root Causes
    if supported_hyps:
        primary = supported_hyps[0]
        ev_summary = primary.evidence[0].interpretation if primary.evidence else "Empirical difference confirmed."
        lines.extend([
            f"### Primary Candidate Explanation (Observational Association - Rank #1)",
            f"- **Hypothesis:** {primary.title}",
            f"- **Plausibility Score:** {primary.plausibility_score:.1f} / 100",
            f"- **Statistical Evidence:** {ev_summary}",
            f"- **Mechanistic Description:** {primary.description}",
            ""
        ])
    else:
        lines.extend([
            "### Investigation Status: No Single Isolated Driver Confirmed",
            "None of the candidate segment hypotheses met the strict threshold for definitive single-driver status. "
            "The shift appears distributed across cohorts or influenced by unobserved external variance.",
            ""
        ])

    # Contribution Breakdown
    lines.extend([
        "## 2. Segment Contribution & Rate-Mix Decomposition",
        "Using Kitagawa Rate-Mix Decomposition, overall variance was partitioned into within-cohort performance drops vs traffic composition shifts:",
        ""
    ])

    for c in top_contributions[:4]:
        culprit_tag = " ⚠️ **[PRIMARY DRIVER]**" if c.is_primary_culprit else ""
        lines.append(
            f"- **{c.dimension} = '{c.segment_value}'**{culprit_tag}: Accounted for **{c.pct_of_total_delta:+.1f}%** of total metric change. "
            f"(Baseline rate: {c.baseline_rate:.4f} → Anomaly rate: {c.anomaly_rate:.4f}, Delta: {c.rate_change:+.4f})."
        )
    lines.append("")

    # Statistical Hypotheses Audit
    lines.extend([
        "## 3. Evaluated Hypotheses & Evidence Matrix",
        "| Rank | Status | Hypothesis | Dimension / Factor | Plausibility | Key Test & Metric |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for h in top_hypotheses[:8]:
        status_icon = "✅ SUPPORTED" if h.status == HypothesisStatus.SUPPORTED else (
            "⚠️ WEAK" if h.status == HypothesisStatus.WEAK else (
                "❌ REJECTED" if h.status == HypothesisStatus.REJECTED else "❓ INCONCLUSIVE"
            )
        )
        test_info = h.evidence[0].test_name if h.evidence else "N/A"
        eff_info = f"{h.evidence[0].effect_size_name}={h.evidence[0].effect_size_value:+.2f}" if h.evidence else ""
        lines.append(
            f"| #{h.rank} | {status_icon} | {h.title} | `{h.dimension}` | {h.plausibility_score:.1f}% | {test_info} ({eff_info}) |"
        )
    lines.append("")

    # Counterfactual Scenarios
    if top_scenarios:
        lines.extend([
            "## 4. Counterfactual Baseline-Restoration Scenarios",
            "Simulating potential recovery if targeted remediation restored segment performance to baseline levels:",
            ""
        ])
        for sc in top_scenarios:
            lines.append(
                f"- **Scenario: {sc.scenario_name}**"
                f"\n  - Projected Metric: **{sc.counterfactual_metric:.4f}** (vs observed {sc.observed_anomaly_metric:.4f})"
                f"\n  - Potential Recovery: **+{sc.recovery_percentage:.1f}%** of total anomaly loss"
                f"\n  - 95% Confidence Interval: [{sc.ci_lower:.4f}, {sc.ci_upper:.4f}]"
            )
        lines.append("")

    # Epistemic Modesty & Caveats
    lines.extend([
        "## 5. Epistemic Guardrails & Non-Causal Disclaimers",
        "> [!IMPORTANT]",
        "> 1. **Observational vs Causal:** These findings document high-magnitude statistical associations and rate-mix contributions. They do not constitute formal DAG-proven causality or randomized trial proof.",
        "> 2. **Omitted Variable Bias:** Unobserved factors (infrastructure outages, third-party vendor downtime, seasonality) may mediate the observed patterns.",
        "> 3. **Validation Recommended:** Controlled rollback, Canary patch, or A/B holdout testing is advised before high-cost structural interventions.",
        ""
    ])

    return "\n".join(lines)


def generate_llm_enhanced_summary(
    investigation_dict: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "openai/gpt-oss-120b"
) -> Optional[str]:
    """
    Calls Groq LLM ONLY if an API key is available, sending ONLY strictly typed
    analytical findings and forbidding fabricated numbers or causal overreach.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=key)

        prompt_payload = {
            "metric": investigation_dict.get("metric_col"),
            "anomaly_window": investigation_dict.get("anomaly_window"),
            "top_hypotheses": investigation_dict.get("hypotheses", [])[:4],
            "top_contributions": investigation_dict.get("contributions", [])[:4],
            "counterfactuals": investigation_dict.get("counterfactuals", [])[:2],
        }

        user_content = (
            f"Analyze this empirical evidence payload:\n{json.dumps(prompt_payload, indent=2)}\n\n"
            f"RULES:\n"
            f"1. ONLY discuss facts present in the payload. DO NOT invent numbers or outside causes.\n"
            f"2. NEVER claim correlation is causation. Explicitly state observational limitations.\n"
            f"3. Write a concise, executive 3-paragraph diagnostic memo covering: "
            f"(a) Anomaly magnitude and baseline delta, "
            f"(b) Top supported hypothesis with specific statistical proof (p-values, effect size, rate drop), "
            f"(c) Recommended engineering/business triage actions based on the primary contributing segment."
        )

        completion_params = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 1,
            "max_completion_tokens": 2048,
            "top_p": 1,
            "stream": True,
            "stop": None
        }

        # Include reasoning_effort if using a reasoning model
        if "gpt-oss" in model or "r1" in model:
            try:
                completion_params["reasoning_effort"] = "medium"
            except Exception:
                pass

        try:
            completion = client.chat.completions.create(**completion_params)
            collected_chunks = []
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                collected_chunks.append(delta)
            return "".join(collected_chunks)
        except Exception:
            # Fallback without reasoning_effort if not supported by model variant
            completion_params.pop("reasoning_effort", None)
            completion = client.chat.completions.create(**completion_params)
            collected_chunks = []
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                collected_chunks.append(delta)
            return "".join(collected_chunks)

    except Exception as e:
        return f"*(Optional Groq LLM explanation skipped: {str(e)})*"


def export_report_html(markdown_content: str) -> str:
    """
    Converts markdown report to a standalone, professionally styled HTML document.
    """
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Detective - Root Cause Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            line-height: 1.6;
            padding: 40px 20px;
            margin: 0;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }}
        h1 {{ color: #1e40af; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        h2 {{ color: #1d4ed8; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        h3 {{ color: #334155; }}
        code {{ background: #f1f5f9; color: #0284c7; padding: 2px 6px; border-radius: 4px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px 14px; text-align: left; }}
        th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        blockquote {{
            border-left: 4px solid #2563eb;
            margin: 20px 0;
            padding: 12px 20px;
            background: #eff6ff;
            border-radius: 0 8px 8px 0;
            color: #1e3a8a;
        }}
    </style>
</head>
<body>
    <div class="container">
        <pre style="white-space: pre-wrap; font-family: inherit;">{markdown_content}</pre>
    </div>
</body>
</html>
"""
    return html_template
