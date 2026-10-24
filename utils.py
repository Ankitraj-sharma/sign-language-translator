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
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
    )
    return fig


def session_metrics(history: list[dict]) -> dict:
    """Calculate aggregate statistics from translation events."""
    if not history:
        return {
            "total_translations": 0,
            "unique_signs": 0,
            "avg_confidence": 0.0,
            "top_sign": "None",
        }

    signs = [h["sign"] for h in history if h.get("sign")]
    confs = [h["confidence"] for h in history if "confidence" in h]

    top_sign = Counter(signs).most_common(1)[0][0] if signs else "None"
    avg_conf = round(sum(confs) / len(confs) * 100, 1) if confs else 0.0

    return {
        "total_translations": len(signs),
        "unique_signs": len(set(signs)),
        "avg_confidence": avg_conf,
        "top_sign": top_sign,
    }


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
