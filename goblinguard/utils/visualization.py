"""Plotly chart builders for the GoblinGuard dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

DARK_LAYOUT: dict = dict(
    paper_bgcolor="#0A0C0F",
    plot_bgcolor="#111418",
    font=dict(family="IBM Plex Mono, monospace", color="#E8EDF2", size=12),
    margin=dict(l=16, r=16, t=32, b=16),
    showlegend=False,
)

_SEVERITY_COLORS: dict[str, str] = {
    "watch": "#F5C542",
    "alert": "#F0833A",
    "critical": "#E8394D",
}


def build_ngram_bar_chart(findings: list[dict]) -> go.Figure:
    """Horizontal bar chart of top n-grams by Z-score.

    Bars colored by severity: watch=amber, alert=orange, critical=red.
    Dark background to match app theme.

    Args:
        findings: List of NgramFinding dicts with ngram, zscore, and severity fields.

    Returns:
        Plotly Figure object.
    """
    if not findings:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="N-gram Anomalies")
        fig.add_annotation(text="No anomalous n-grams detected", showarrow=False)
        return fig

    ngrams = [f["ngram"] if isinstance(f, dict) else f.ngram for f in findings]
    zscores = [f["zscore"] if isinstance(f, dict) else f.zscore for f in findings]
    severities = [f["severity"] if isinstance(f, dict) else f.severity for f in findings]

    # Reverse for horizontal bar chart (top item at top)
    ngrams = ngrams[::-1]
    zscores = zscores[::-1]
    severities = severities[::-1]

    colors = [_SEVERITY_COLORS.get(str(s), "#F5C542") for s in severities]

    fig = go.Figure(
        go.Bar(
            x=zscores,
            y=ngrams,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"z={z:.2f}" for z in zscores],
            textposition="outside",
            textfont=dict(size=11, color="#E8EDF2"),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Anomalous N-grams by Z-score", font=dict(size=14)),
        xaxis=dict(title="Z-score", gridcolor="#1A1E24"),
        yaxis=dict(gridcolor="#1A1E24"),
        height=max(300, len(ngrams) * 35 + 80),
    )
    return fig


def build_score_trend_chart(sentence_scores: list[dict]) -> go.Figure:
    """Line chart of classifier_prob across sentences.

    Shows where tic intensity peaks within a document.
    Reference line at 0.5 threshold.

    Args:
        sentence_scores: List of SentenceScore dicts with classifier_prob field.

    Returns:
        Plotly Figure object.
    """
    if not sentence_scores:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="Sentence Tic Probability")
        fig.add_annotation(text="No sentences to display", showarrow=False)
        return fig

    probs = [
        s["classifier_prob"] if isinstance(s, dict) else s.classifier_prob for s in sentence_scores
    ]
    indices = list(range(1, len(probs) + 1))
    hover_texts = [
        (s["sentence"] if isinstance(s, dict) else s.sentence)[:80] + "..." for s in sentence_scores
    ]

    fig = go.Figure()

    # Main probability line
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=probs,
            mode="lines+markers",
            line=dict(color="#39D98A", width=2),
            marker=dict(size=6, color="#39D98A"),
            hovertext=hover_texts,
            hoverinfo="text+y",
            name="P(tic)",
        )
    )

    # Threshold reference line at 0.5
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#E8394D",
        opacity=0.6,
        annotation_text="threshold",
        annotation_font_color="#E8394D",
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Classifier Probability per Sentence", font=dict(size=14)),
        xaxis=dict(title="Sentence Index", gridcolor="#1A1E24"),
        yaxis=dict(title="P(tic)", range=[0, 1.05], gridcolor="#1A1E24"),
        height=350,
    )
    return fig


def build_detector_breakdown_chart(breakdown: dict) -> go.Figure:
    """Horizontal grouped bar comparing ngram/ae/classifier sub-scores.

    Three bars, color-coded to match severity palette.

    Args:
        breakdown: Dict with ngram_score, autoencoder_score, classifier_score fields.

    Returns:
        Plotly Figure object.
    """
    labels = ["N-gram Scanner", "Autoencoder", "Classifier"]
    values = [
        breakdown.get("ngram_score", 0.0) if isinstance(breakdown, dict) else breakdown.ngram_score,
        (
            breakdown.get("autoencoder_score", 0.0)
            if isinstance(breakdown, dict)
            else breakdown.autoencoder_score
        ),
        (
            breakdown.get("classifier_score", 0.0)
            if isinstance(breakdown, dict)
            else breakdown.classifier_score
        ),
    ]
    colors = ["#F5C542", "#F0833A", "#E8394D"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=12, color="#E8EDF2"),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Detector Breakdown", font=dict(size=14)),
        xaxis=dict(title="Signal Strength", range=[0, 1.15], gridcolor="#1A1E24"),
        yaxis=dict(gridcolor="#1A1E24"),
        height=250,
    )
    return fig
