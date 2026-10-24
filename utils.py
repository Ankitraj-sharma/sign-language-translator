"""
Visualization, Analytics & Metrics Helpers for SignEase AI.
"""

from collections import Counter
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import numpy as np


THEME_COLORS = ["#00D2BA", "#00A8FF", "#9C88FF", "#FBC531", "#4CD137", "#487EB0", "#E84118"]


def sign_frequency_chart(signs_history: list[str]) -> go.Figure:
    """Horizontal bar chart showing frequencies of recognized signs."""
    if not signs_history:
        fig = go.Figure()
        fig.add_annotation(
            text="No signs recorded in this session yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#888")
        )
        fig.update_layout(
            height=250,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        return fig

    counts = Counter(signs_history)
    df = pd.DataFrame(counts.items(), columns=["Sign", "Count"]).sort_values("Count", ascending=True)

    fig = px.bar(
        df,
        x="Count",
        y="Sign",
        orientation="h",
        color="Sign",
        color_discrete_sequence=THEME_COLORS,
        title="Session Sign Translation Frequency",
    )
    fig.update_layout(
        showlegend=False,
        height=max(260, len(df) * 36 + 60),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Recognized Count",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
    )
    fig.update_xaxes(dtick=1, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def confidence_gauge_chart(confidence: float) -> go.Figure:
    """Circular gauge chart showing current detection confidence."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            title={"text": "Gesture Confidence (%)", "font": {"size": 16, "color": "#00D2BA"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
                "bar": {"color": "#00D2BA"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(235, 77, 75, 0.25)"},
                    {"range": [50, 75], "color": "rgba(240, 147, 43, 0.25)"},
                    {"range": [75, 100], "color": "rgba(106, 176, 76, 0.25)"},
                ],
