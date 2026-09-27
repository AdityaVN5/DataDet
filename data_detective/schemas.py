"""
Data Detective Data Structures and Schemas.
Provides typed definitions for profiling, anomaly detection, hypotheses,
statistical evidence, contribution decomposition, and counterfactuals.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DataType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT_ID = "text_id"


class HypothesisStatus(str, Enum):
    SUPPORTED = "supported"        # Statistically significant + meaningful effect size
    WEAK = "weak"                  # Marginally significant or small effect size
    REJECTED = "rejected"          # Null hypothesis cannot be rejected (or contrary effect)
    INCONCLUSIVE = "inconclusive"  # Insufficient sample size or excessive variance


class HypothesisType(str, Enum):
    SEGMENT_PERFORMANCE = "segment_performance"  # A segment's conversion/mean degraded
    MIX_SHIFT = "mix_shift"                      # Shift in traffic/volume toward a lower-performing bucket
    COVARIATE_DRIFT = "covariate_drift"          # Correlation or drift in a continuous driver (e.g. latency)
    MULTIVARIATE_ANOMALY = "multivariate_anomaly"# Isolation Forest cluster of multidimensional anomalies
    TEMPORAL_CHANGEPOINT = "temporal_changepoint"# Abrupt structural step-change vs natural variance


@dataclass
class ColumnProfile:
    name: str
    inferred_type: DataType
    raw_dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    is_constant: bool
    is_id_candidate: bool
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    iqr: Optional[float] = None
    outlier_count: int = 0
    outlier_pct: float = 0.0
    top_categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inferred_type"] = self.inferred_type.value
        return d


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    duplicate_pct: float
    columns: Dict[str, ColumnProfile]
    candidate_date_cols: List[str] = field(default_factory=list)
    candidate_metric_cols: List[str] = field(default_factory=list)
    candidate_segment_cols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": self.duplicate_pct,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "candidate_date_cols": self.candidate_date_cols,
            "candidate_metric_cols": self.candidate_metric_cols,
            "candidate_segment_cols": self.candidate_segment_cols,
        }


@dataclass
class AnomalyPoint:
    index_or_date: str
    metric_value: float
    expected_value: float
    deviation_z: float
    is_anomaly: bool
    anomaly_score: float  # e.g., Isolation Forest score or Z-score normalized
    severity: str         # low, medium, high, critical


@dataclass
class AnomalyWindow:
    metric_col: str
    date_col: Optional[str]
    baseline_period_label: str
    anomaly_period_label: str
    baseline_indices: List[Any]
    anomaly_indices: List[Any]
    baseline_mean: float
    anomaly_mean: float
    absolute_delta: float
    relative_pct_change: float
    direction: str  # "drop" or "spike"
    detection_method: str  # "rolling_zscore", "isolation_forest", "iqr", "user_specified"
    flagged_points: List[AnomalyPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_col": self.metric_col,
            "date_col": self.date_col,
            "baseline_period_label": self.baseline_period_label,
            "anomaly_period_label": self.anomaly_period_label,
            "baseline_count": len(self.baseline_indices),
            "anomaly_count": len(self.anomaly_indices),
            "baseline_mean": round(self.baseline_mean, 4),
            "anomaly_mean": round(self.anomaly_mean, 4),
            "absolute_delta": round(self.absolute_delta, 4),
            "relative_pct_change": round(self.relative_pct_change, 2),
            "direction": self.direction,
            "detection_method": self.detection_method,
            "flagged_points_count": len(self.flagged_points),
        }


@dataclass
class StatisticalEvidence:
    test_name: str                     # e.g., "Two-Sample Mann-Whitney U", "Student's t-test", "Chi-Square Independence"
    test_statistic: float
    p_value: float
    effect_size_name: str              # e.g., "Cohen's d", "Cramer's V", "Cliff's Delta", "Pearson r"
    effect_size_value: float
    sample_size_baseline: int
    sample_size_anomaly: int
    degrees_of_freedom: Optional[int] = None
    confidence_interval_95: Optional[Tuple[float, float]] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_statistic": round(self.test_statistic, 4),
            "p_value": self.p_value,
            "p_value_formatted": f"< 0.0001" if self.p_value < 0.0001 else f"{self.p_value:.4f}",
            "effect_size_name": self.effect_size_name,
            "effect_size_value": round(self.effect_size_value, 4),
            "sample_size_baseline": self.sample_size_baseline,
            "sample_size_anomaly": self.sample_size_anomaly,
            "degrees_of_freedom": self.degrees_of_freedom,
            "confidence_interval_95": [round(x, 4) for x in self.confidence_interval_95] if self.confidence_interval_95 else None,
            "interpretation": self.interpretation,
        }


@dataclass
class Hypothesis:
    id: str
    title: str
    hypothesis_type: HypothesisType
    target_metric: str
    dimension: str
    dimension_value: Optional[str]
    description: str
    rationale: str
    test_plan: str
    evidence: List[StatisticalEvidence] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.INCONCLUSIVE
    plausibility_score: float = 0.0  # 0.0 to 100.0 score based on effect size, p-val & contribution
    rank: int = 999
    caveats: List[str] = field(default_factory=list)
    segment_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "hypothesis_type": self.hypothesis_type.value,
            "target_metric": self.target_metric,
            "dimension": self.dimension,
            "dimension_value": self.dimension_value,
            "description": self.description,
            "rationale": self.rationale,
            "test_plan": self.test_plan,
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status.value,
            "plausibility_score": round(self.plausibility_score, 1),
            "rank": self.rank,
            "caveats": self.caveats,
            "segment_stats": self.segment_stats,
        }


@dataclass
class SegmentContribution:
    dimension: str
    segment_value: str
    baseline_volume: int
    anomaly_volume: int
    baseline_share: float
    anomaly_share: float
    baseline_rate: float
    anomaly_rate: float
    rate_change: float
    absolute_contribution: float
    pct_of_total_delta: float
    is_primary_culprit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "segment_value": str(self.segment_value),
            "baseline_volume": self.baseline_volume,
            "anomaly_volume": self.anomaly_volume,
            "baseline_share": round(self.baseline_share, 4),
            "anomaly_share": round(self.anomaly_share, 4),
            "baseline_rate": round(self.baseline_rate, 4),
            "anomaly_rate": round(self.anomaly_rate, 4),
            "rate_change": round(self.rate_change, 4),
            "absolute_contribution": round(self.absolute_contribution, 4),
            "pct_of_total_delta": round(self.pct_of_total_delta, 2),
            "is_primary_culprit": self.is_primary_culprit,
        }


@dataclass
class CounterfactualResult:
    scenario_name: str
    dimension: str
    segment_value: str
    target_metric: str
    observed_anomaly_metric: float
    counterfactual_metric: float
    restoration_delta: float
    recovery_percentage: float
    ci_lower: float
    ci_upper: float
    assumptions_and_caveats: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "dimension": self.dimension,
            "segment_value": self.segment_value,
            "target_metric": self.target_metric,
            "observed_anomaly_metric": round(self.observed_anomaly_metric, 4),
            "counterfactual_metric": round(self.counterfactual_metric, 4),
            "restoration_delta": round(self.restoration_delta, 4),
            "recovery_percentage": round(self.recovery_percentage, 2),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "assumptions_and_caveats": self.assumptions_and_caveats,
        }


@dataclass
class AuditLogEntry:
    timestamp: str
    step: str
    action: str
    details: str
    status: str = "COMPLETED"


@dataclass
class InvestigationRun:
    id: str
    title: str
    created_at: str
    dataset_name: str
    row_count: int
    metric_col: str
    date_col: Optional[str]
    anomaly_window: AnomalyWindow
    hypotheses: List[Hypothesis]
    contributions: List[SegmentContribution]
    counterfactuals: List[CounterfactualResult]
    audit_trail: List[AuditLogEntry]
    executive_summary: str = ""
    llm_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "metric_col": self.metric_col,
            "date_col": self.date_col,
            "anomaly_window": self.anomaly_window.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "contributions": [c.to_dict() for c in self.contributions],
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "audit_trail": [asdict(a) for a in self.audit_trail],
            "executive_summary": self.executive_summary,
            "llm_explanation": self.llm_explanation,
        }
