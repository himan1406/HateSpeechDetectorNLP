"""Shared visual system for the local Streamlit interface.

Small, dependency-free helpers: one page setup, a few typographic rules, and
compact HTML for the score and interval bars. Keeping them here means the three
screens stay consistent without repeating CSS in every page.
"""

from __future__ import annotations

from html import escape

import streamlit as st

_CSS = """
<style>
:root {
  --ui-border: rgba(17, 24, 39, 0.10);
  --ui-muted: #6B7280;
  --ui-accent: #2563EB;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stMainBlockContainer"] {
  padding-top: 2.6rem;
  padding-bottom: 5rem;
  max-width: 1120px;
}
h1 { letter-spacing: -0.03em; font-weight: 650; }
h2, h3, h4 { letter-spacing: -0.015em; font-weight: 600; }
[data-testid="stMetricValue"] {
  font-size: 1.75rem;
  font-weight: 600;
  letter-spacing: -0.025em;
}
[data-testid="stMetricLabel"] { color: var(--ui-muted); }
[data-testid="stExpander"] details { border-radius: 10px; border-color: var(--ui-border); }
[data-testid="stDataFrame"] { border-radius: 10px; }
.stButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 550; }

.ui-eyebrow {
  color: var(--ui-muted);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin-bottom: -0.35rem;
}
.ui-post {
  border: 1px solid var(--ui-border);
  border-radius: 10px;
  background: #FCFCFD;
  padding: 1rem 1.1rem;
  font-size: 0.95rem;
  line-height: 1.9;
  overflow-wrap: anywhere;
}
.ui-track {
  position: relative;
  height: 8px;
  margin: 0.6rem 0 0.35rem;
  border-radius: 999px;
  background: #EDF0F4;
}
.ui-fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; }
.ui-band {
  position: absolute;
  top: 0;
  bottom: 0;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.20);
}
.ui-dot {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid #FFFFFF;
  background: var(--ui-accent);
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.35);
  transform: translate(-50%, -50%);
}
.ui-ref {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 2px;
  border-radius: 2px;
  background: #98A2B3;
}
.ui-axis {
  display: flex;
  justify-content: space-between;
  color: var(--ui-muted);
  font-size: 0.72rem;
}
.ui-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.75rem 0 0.5rem; }
.ui-chip {
  border: 1px solid var(--ui-border);
  border-radius: 999px;
  background: #FFFFFF;
  color: #344054;
  font-size: 0.8rem;
  padding: 0.2rem 0.55rem;
}
.ui-chip b { margin-left: 0.3rem; font-weight: 600; }
.ui-chip b.pos { color: #C2410C; }
.ui-chip b.neg { color: #1D4ED8; }
.ui-empty {
  border: 1px dashed var(--ui-border);
  border-radius: 12px;
  padding: 1.6rem 1.4rem;
  text-align: center;
  margin-top: 0.4rem;
}
.ui-empty p { margin: 0; }
.ui-empty-title { color: #344054; font-size: 0.95rem; font-weight: 600; }
.ui-empty-body { color: var(--ui-muted); font-size: 0.86rem; margin-top: 0.2rem !important; }
</style>
"""


def page(title: str, *, layout: str = "centered") -> None:
    """Configure the page and install the shared styles. Call this first."""
    st.set_page_config(page_title=f"{title} · XAI-HSD", layout=layout)
    st.markdown(_CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="ui-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def score_bar(score: float, threshold: float) -> None:
    """Horizontal score with a tick at the operating threshold."""
    filled = max(0.0, min(1.0, float(score))) * 100
    marker = max(0.0, min(1.0, float(threshold))) * 100
    color = "#DC2626" if score >= threshold else "var(--ui-accent)"
    st.markdown(
        f'<div class="ui-track" role="img" aria-label="Score {score:.0%}, '
        f'threshold {threshold:.0%}">'
        f'<div class="ui-fill" style="width:{filled:.2f}%;background:{color}"></div>'
        f'<div class="ui-ref" style="left:{marker:.2f}%"></div>'
        "</div>"
        f'<div class="ui-axis"><span>0%</span><span>threshold {threshold:.0%}</span>'
        "<span>100%</span></div>",
        unsafe_allow_html=True,
    )


def interval_bar(
    share: float, low: float, high: float, *, reference: float | None = None
) -> None:
    """Flagged share with its confidence band and, optionally, a reference tick."""
    def position(value: float) -> float:
        return max(0.0, min(1.0, float(value))) * 100

    left, right = position(low), position(high)
    tick = ""
    if reference is not None:
        tick = (
            f'<div class="ui-ref" style="left:{position(reference):.2f}%" '
            f'title="Reference rate {reference:.1%}"></div>'
        )
    st.markdown(
        '<div class="ui-track" role="img" aria-label="'
        f'Flagged share {share:.1%}, interval {low:.1%} to {high:.1%}">'
        f'<div class="ui-band" style="left:{left:.2f}%;'
        f'width:{max(right - left, 0.4):.2f}%"></div>'
        f"{tick}"
        f'<div class="ui-dot" style="left:{position(share):.2f}%"></div>'
        "</div>"
        '<div class="ui-axis"><span>0%</span><span>50%</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )
    detail = f"{share:.1%} flagged · interval {low:.1%}–{high:.1%}"
    if reference is not None:
        detail += f" · reference {reference:.1%}"
    st.caption(detail)


def chips(items) -> None:
    """Signed word attributions as compact pills, strongest first."""
    parts = []
    for token, value in items:
        tone = "pos" if value > 0 else "neg"
        parts.append(
            f'<span class="ui-chip">{escape(str(token))}'
            f'<b class="{tone}">{value:+.3f}</b></span>'
        )
    st.markdown('<div class="ui-chips">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def post_block(body: str, *, html: bool = False) -> None:
    content = body if html else escape(body)
    st.markdown(f'<div class="ui-post">{content}</div>', unsafe_allow_html=True)


def empty(title: str, body: str) -> None:
    st.markdown(
        '<div class="ui-empty">'
        f'<p class="ui-empty-title">{escape(title)}</p>'
        f'<p class="ui-empty-body">{escape(body)}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
