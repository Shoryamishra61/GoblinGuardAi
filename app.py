"""Streamlit dashboard for GoblinGuard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="GoblinGuard — LLM Tic Auditor",
    page_icon="👺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Design System CSS ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --color-bg:            #0A0C0F;
  --color-surface:       #111418;
  --color-surface-2:     #1A1E24;
  --color-border:        #252A32;
  --color-border-focus:  #3D8B6B;
  --color-text-primary:  #E8EDF2;
  --color-text-secondary:#7A8694;
  --color-text-muted:    #3D4550;
  --color-primary:       #39D98A;
  --color-primary-dim:   #1A6B43;
  --color-primary-glow:  rgba(57, 217, 138, 0.15);
  --color-safe:          #39D98A;
  --color-watch:         #F5C542;
  --color-alert:         #F0833A;
  --color-critical:      #E8394D;
  --glow-safe:    0 0 16px rgba(57,217,138,0.35);
  --glow-watch:   0 0 16px rgba(245,197,66,0.35);
  --glow-alert:   0 0 16px rgba(240,131,58,0.35);
  --glow-critical:0 0 16px rgba(232,57,77,0.35);
  --font-display: 'Syne', sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;
  --radius-sm: 4px; --radius-md: 8px; --radius-lg: 12px; --radius-xl: 16px;
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px;
  --space-6:24px; --space-8:32px; --space-12:48px; --space-16:64px;
}

.stApp, .main, [data-testid="stAppViewContainer"] {
  background-color: var(--color-bg) !important;
}
[data-testid="stSidebar"] {
  background-color: var(--color-surface) !important;
  border-right: 1px solid var(--color-border);
}

html, body, .stMarkdown, .stText, p, span, label {
  color: var(--color-text-primary);
  font-family: var(--font-display);
}

.score-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  text-align: center;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.score-value {
  font-family: var(--font-display);
  font-size: 72px;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
}
.score-label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-text-secondary);
  margin-top: 8px;
}

.risk-badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 12px;
}
.risk-safe     { background:rgba(57,217,138,0.12); color:var(--color-safe); }
.risk-watch    { background:rgba(245,197,66,0.12);  color:var(--color-watch); }
.risk-alert    { background:rgba(240,131,58,0.12);  color:var(--color-alert); }
.risk-critical {
  background:rgba(232,57,77,0.12);
  color:var(--color-critical);
  animation: pulse-critical 2s ease-in-out infinite;
}
@keyframes pulse-critical {
  0%,100% { box-shadow: none; }
  50%      { box-shadow: var(--glow-critical); }
}

.ngram-row {
  display:flex; align-items:center; gap:12px;
  padding: 10px 16px;
  border-radius: var(--radius-md);
  margin-bottom: 4px;
  border: 1px solid var(--color-border);
  background: var(--color-surface-2);
  transition: background 100ms;
}
.ngram-row:hover { background: var(--color-surface); }
.ngram-text {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--color-text-primary);
  flex: 1;
}
.ngram-zscore {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--color-text-secondary);
  min-width: 80px;
  text-align: right;
}

.sent-clean    { border-radius:3px; padding:1px 2px; }
.sent-watch    { background:rgba(245,197,66,0.10); border-bottom:1px solid rgba(245,197,66,0.4); border-radius:3px; padding:1px 2px; }
.sent-alert    { background:rgba(240,131,58,0.12); border-bottom:1px solid rgba(240,131,58,0.5); border-radius:3px; padding:1px 2px; }
.sent-critical { background:rgba(232,57,77,0.14);  border-bottom:2px solid var(--color-critical); border-radius:3px; padding:1px 2px; }

.section-header {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 10px;
  margin-bottom: 16px;
}

.fix-card {
  background: var(--color-surface-2);
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-primary);
  border-radius: var(--radius-md);
  padding: 16px;
  margin-bottom: 12px;
}
.fix-card .fix-title {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing:0.08em;
  text-transform:uppercase;
  color: var(--color-primary);
  margin-bottom: 6px;
}
.fix-card .fix-body {
  font-size: 14px;
  color: var(--color-text-secondary);
  line-height: 1.6;
}
.fix-card code {
  font-family: var(--font-mono);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 12px;
  color: var(--color-primary);
}

.stTextArea textarea, .stTextInput input {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text-primary) !important;
  font-family: var(--font-mono) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--color-border-focus) !important;
  box-shadow: 0 0 0 3px var(--color-primary-glow) !important;
}
.stButton > button {
  background: var(--color-primary) !important;
  color: #0A0C0F !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  height: 44px !important;
  transition: transform 80ms ease, box-shadow 120ms ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: var(--glow-safe) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Model Loading ---
@st.cache_resource
def load_engine():  # type: ignore[no-untyped-def]
    """Load AuditEngine with cached resource."""
    from goblinguard.model.audit_engine import AuditEngine
    from goblinguard.schemas.config import GoblinGuardConfig

    config = GoblinGuardConfig()
    model_dir = Path("model/")
    return AuditEngine(config=config, model_dir=model_dir)


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
    <div style="padding: 8px 0 24px 0;">
      <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                  letter-spacing:-0.02em; color:#E8EDF2;">
        👺 GoblinGuard
      </div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                  letter-spacing:0.12em; text-transform:uppercase;
                  color:#7A8694; margin-top:4px;">
        LLM Output Auditor v1.0.0
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Audit Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Select Audit Mode",
        ["Single Audit", "Batch Audit (.json)", "Compare: Base vs Fine-tuned"],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="section-header" style="margin-top:24px;">Model Metadata</div>',
        unsafe_allow_html=True,
    )
    model_name = st.text_input("Model name", placeholder="e.g. gpt-5.5")
    model_version = st.text_input("Version", placeholder="e.g. 2026-04-30")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.markdown("---")
    st.markdown(
        """
    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3D4550; line-height:1.6;">
      Inspired by OpenAI's<br>
      <a href="https://openai.com/index/where-the-goblins-came-from/"
         style="color:#3D8B6B;">Where the Goblins Came From</a><br>
      April 30, 2026
    </div>
    """,
        unsafe_allow_html=True,
    )

# --- Page Header ---
st.markdown(
    """
<div style="margin-bottom:32px;">
  <h1 style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800;
              letter-spacing:-0.02em; color:#E8EDF2; margin:0;">
    LLM Output Auditor
  </h1>
  <p style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#7A8694; margin:8px 0 0 0;">
    Detect reward-hacking tics and style leakage before they reach production.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Try loading engine ---
try:
    engine = load_engine()
    engine_loaded = True
except Exception:
    engine_loaded = False
    st.warning(
        "⚠️ Model weights not found. Train models first:\n\n"
        "`python -m goblinguard train --data data/goblin_examples.json --output model/`"
    )

# --- Input Area ---
import json

if mode == "Single Audit":
    text_input = st.text_area(
        "Paste LLM output", height=200, placeholder="Paste model output here..."
    )
    texts_to_audit = [text_input] if text_input else []

elif mode == "Batch Audit (.json)":
    uploaded = st.file_uploader("Upload JSON file of outputs", type=["json"])
    texts_to_audit = []
    if uploaded:
        data = json.loads(uploaded.read().decode("utf-8"))
        if isinstance(data, list):
            texts_to_audit = [
                (item.get("text", str(item)) if isinstance(item, dict) else str(item))
                for item in data
            ]
        elif isinstance(data, dict):
            for key in ("examples", "outputs", "texts"):
                if key in data:
                    texts_to_audit = [
                        (item.get("text", str(item)) if isinstance(item, dict) else str(item))
                        for item in data[key]
                    ]
                    break

else:  # Compare mode
    col_base, col_ft = st.columns(2)
    with col_base:
        base_text = st.text_area("Base Output", height=200, placeholder="Paste base model output...")
    with col_ft:
        ft_text = st.text_area("Fine-tuned Output", height=200, placeholder="Paste fine-tuned output...")
    texts_to_audit = []

# --- Run Audit ---
if mode != "Compare: Base vs Fine-tuned":
    run_clicked = st.button("Run Audit →")
    if run_clicked and texts_to_audit and engine_loaded:
        with st.spinner("Running audit..."):
            report = engine.audit(
                texts_to_audit,
                model_name=model_name or "unknown",
                version=model_version or "unknown",
                temperature=temperature,
            )
        st.session_state["last_report"] = report
else:
    run_clicked = st.button("Run Comparison →")
    if run_clicked and base_text and ft_text and engine_loaded:
        with st.spinner("Running audit..."):
            base_report = engine.audit([base_text], model_name="base", version=model_version or "unknown")
            ft_report = engine.audit([ft_text], model_name="finetuned", version=model_version or "unknown")
        st.session_state["last_report"] = ft_report
        st.session_state["base_report"] = base_report

# --- Results ---
if "last_report" in st.session_state:
    from goblinguard.utils.text_utils import split_sentences
    from goblinguard.utils.visualization import (
        build_ngram_bar_chart,
        build_score_trend_chart,
    )

    report = st.session_state["last_report"]
    risk = report.risk_level.value if hasattr(report.risk_level, "value") else str(report.risk_level)

    score_color = {
        "safe": "#39D98A",
        "watch": "#F5C542",
        "alert": "#F0833A",
        "critical": "#E8394D",
    }.get(risk, "#39D98A")

    # D1 — Score row
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
        <div class="score-card">
          <div class="score-value" style="color:{score_color};">{report.tic_score:.1f}</div>
          <div class="score-label">TicScore</div>
          <div class="risk-badge risk-{risk}">{risk.upper()}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("N-gram Signal", f"{report.detector_breakdown.ngram_score:.2f}")
        m2.metric("Autoencoder Signal", f"{report.detector_breakdown.autoencoder_score:.2f}")
        m3.metric("Classifier Signal", f"{report.detector_breakdown.classifier_score:.2f}")

    # D2 — Highlighted text
    if report.sentence_scores:
        html_parts = []
        for ss in report.sentence_scores:
            lbl = ss.label.value if hasattr(ss.label, "value") else str(ss.label)
            html_parts.append(f'<span class="sent-{lbl}">{ss.sentence}</span>')
        st.markdown(" ".join(html_parts), unsafe_allow_html=True)

    # D3 — N-gram leaderboard
    if report.ngram_findings:
        ngram_html = ""
        for nf in report.ngram_findings:
            sev = nf.severity.value if hasattr(nf.severity, "value") else str(nf.severity)
            ngram_html += f"""
            <div class="ngram-row">
              <span class="ngram-text">{nf.ngram}</span>
              <span class="risk-badge risk-{sev}">{sev}</span>
              <span class="ngram-zscore">z={nf.zscore:.2f}</span>
            </div>"""
        st.markdown(ngram_html, unsafe_allow_html=True)

    # D4 — Trend chart
    if report.sentence_scores:
        scores_data = [
            {"sentence": ss.sentence, "classifier_prob": ss.classifier_prob}
            for ss in report.sentence_scores
        ]
        st.plotly_chart(build_score_trend_chart(scores_data), use_container_width=True)

    # D5 — N-gram bar chart
    if report.ngram_findings:
        findings_data = [
            {"ngram": nf.ngram, "zscore": nf.zscore, "severity": nf.severity.value if hasattr(nf.severity, "value") else str(nf.severity)}
            for nf in report.ngram_findings
        ]
        st.plotly_chart(build_ngram_bar_chart(findings_data), use_container_width=True)

    # D6 — Fix suggestions
    if report.fix_suggestions:
        for fs in report.fix_suggestions:
            st.markdown(
                f"""
            <div class="fix-card">
              <div class="fix-title">{fs.tic_class}</div>
              <div class="fix-body">{fs.description}<br><br><code>{fs.suggested_override}</code></div>
            </div>""",
                unsafe_allow_html=True,
            )

    # D7 — Export
    st.download_button(
        label="Export JSON Report",
        data=report.to_json(),
        file_name=f"goblinguard_report_{report.audit_id[:8]}.json",
        mime="application/json",
    )
