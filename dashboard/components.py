"""Small reusable Streamlit presentation components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px; padding-top: 2rem; }
                [data-testid="stMetric"] { background: #172622 !important; border: 1px solid #2d5149;
                    border-radius: 8px; padding: 1rem; min-height: 118px; }
                [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
                [data-testid="stMetricValue"], [data-testid="stMetricValue"] div,
                [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div {
                    color: #f4fbf8 !important; opacity: 1 !important;
                }
                [data-testid="stMetricLabel"] { color: #a9c7bd !important; }
                [data-testid="stMetricLabel"] p { font-size: .82rem !important; line-height: 1.2;
                    white-space: normal; }
                [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
                    font-weight: 700; font-size: 1.08rem !important; line-height: 1.2;
                    white-space: nowrap !important; overflow: visible !important;
                    text-overflow: clip !important; width: 100%; }
                [data-testid="stMetric"] svg { color: #8ed6c0 !important; fill: #8ed6c0 !important; }
        .dashboard-kicker { color: #27756d; font-size: .8rem; font-weight: 700;
          letter-spacing: .08em; text-transform: uppercase; }
        .dashboard-note { color: #52615e; font-size: .9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="dashboard-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="dashboard-note">{description}</div>', unsafe_allow_html=True)


def metric_row(metrics: list[tuple[str, Any, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def missing_artifact(name: str, command_hint: str = "Run the required pipeline stage first.") -> None:
    st.warning(f"{name} is unavailable. {command_hint}")