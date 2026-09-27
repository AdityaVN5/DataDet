"""
Data Detective Evidence Graph Engine.
Constructs a NetworkX directed evidence graph linking Target Metrics,
Dimensions, Segment Cohorts, Covariates, and Statistical Evidence nodes.
Renders interactive Plotly network visualizations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import plotly.graph_objects as go

from data_detective.schemas import Hypothesis, HypothesisStatus, SegmentContribution


def build_evidence_graph(
    metric_name: str,
    hypotheses: List[Hypothesis],
    contributions: List[SegmentContribution]
) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph depicting evidence pathways:
    Metric -> Dimension -> Segment / Factor -> Statistical Hypothesis / Evidence
    """
    G = nx.DiGraph()

    # Metric Root Node
    G.add_node(
        metric_name,
        node_type="metric",
        label=f"Target: {metric_name}",
        title=f"Target Metric: {metric_name}",
        size=35,
        color="#3b82f6",  # Bright Blue
        info="Root investigative metric"
    )

    # Add Dimensions and Top Segments
    seen_dimensions = set()

    for h in hypotheses:
        dim = h.dimension
        if dim not in seen_dimensions:
            seen_dimensions.add(dim)
            G.add_node(
                dim,
                node_type="dimension",
                label=f"Dim: {dim}",
                title=f"Dimension: {dim}",
                size=25,
                color="#8b5cf6",  # Purple
                info=f"Categorical dimension: {dim}"
            )
            # Edge from Metric to Dimension
            G.add_edge(metric_name, dim, weight=1.0, relation="decomposed_by")

        # Segment or Factor Node
        seg_node_id = f"{dim}={h.dimension_value}" if h.dimension_value else f"{dim}_drift"
        if not G.has_node(seg_node_id):
            G.add_node(
                seg_node_id,
                node_type="segment",
                label=str(h.dimension_value) if h.dimension_value else dim,
                title=f"Factor: {seg_node_id}",
                size=22,
                color="#06b6d4",  # Cyan
                info=h.description
            )
            G.add_edge(dim, seg_node_id, weight=1.5, relation="contains")

        # Hypothesis Node
        hyp_color = "#ef4444" if h.status == HypothesisStatus.SUPPORTED else (
            "#f59e0b" if h.status == HypothesisStatus.WEAK else "#6b7280"
        )
        hyp_node_id = h.id
        G.add_node(
            hyp_node_id,
            node_type="hypothesis",
            label=f"Rank #{h.rank}: {h.status.value.upper()}",
            title=f"{h.id} ({h.status.value.upper()})",
            size=18,
            color=hyp_color,
            info=f"{h.title} | Plausibility: {h.plausibility_score:.1f}%"
        )
        G.add_edge(seg_node_id, hyp_node_id, weight=h.plausibility_score / 20.0, relation="tested_by")

        # Statistical Evidence Node (if primary evidence exists)
        if h.evidence:
            ev = h.evidence[0]
            ev_node_id = f"EV_{h.id}"
            ev_label = f"{ev.effect_size_name}={ev.effect_size_value:+.2f} (p={ev.p_value:.3f})"
            G.add_node(
                ev_node_id,
                node_type="evidence",
                label=ev_label,
                title=f"Evidence: {ev.test_name}",
                size=14,
                color="#10b981",  # Emerald Green
                info=ev.interpretation
            )
            G.add_edge(hyp_node_id, ev_node_id, weight=2.0, relation="verified_by")

    return G


def plot_evidence_graph_plotly(G: nx.DiGraph) -> go.Figure:
    """
    Renders the NetworkX evidence graph into a high-aesthetic, interactive Plotly visualization.
    """
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No evidence graph data available", showarrow=False)
        return fig

    # Compute node positions using spring layout with good separation
    pos = nx.spring_layout(G, k=1.8, iterations=70, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="rgba(148, 163, 184, 0.45)"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces categorized by node_type for clean legend
    node_types = ["metric", "dimension", "segment", "hypothesis", "evidence"]
    type_display_names = {
        "metric": "Target Metric",
        "dimension": "Dimension",
        "segment": "Segment / Factor",
        "hypothesis": "Hypothesis (Ranked)",
        "evidence": "Statistical Proof"
    }

    traces = [edge_trace]

    for ntype in node_types:
        nx_coords = []
        ny_coords = []
        n_text = []
        n_hover = []
        n_colors = []
        n_sizes = []

        for node in G.nodes():
            data = G.nodes[node]
            if data.get("node_type") == ntype:
                x, y = pos[node]
                nx_coords.append(x)
                ny_coords.append(y)
                n_text.append(data.get("label", node))
                n_hover.append(
                    f"<b>{data.get('title', node)}</b><br>"
                    f"Type: {ntype.upper()}<br>"
                    f"{data.get('info', '')}"
                )
                n_colors.append(data.get("color", "#64748b"))
                n_sizes.append(data.get("size", 20))

        if nx_coords:
            node_trace = go.Scatter(
                x=nx_coords,
                y=ny_coords,
                mode="markers+text",
                name=type_display_names.get(ntype, ntype),
                text=n_text,
                textposition="bottom center",
                textfont=dict(size=11, color="#1e293b", family="Arial, sans-serif"),
                hoverinfo="text",
                hovertext=n_hover,
                marker=dict(
                    color=n_colors,
                    size=n_sizes,
                    line=dict(width=2, color="#ffffff")
                )
            )
            traces.append(node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="<b>Evidence & Root-Cause Relationship Graph</b>",
                font=dict(size=17, color="#0f172a")
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color="#334155", size=11),
                bgcolor="rgba(255, 255, 255, 0.9)"
            ),
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=580
        )
    )
    return fig
