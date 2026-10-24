"""
SignEase AI — Real-Time Sign Language to Speech Translator
===========================================================
A production-grade Computer Vision and Human-Computer Interaction system
that translates American & Indian Sign Language gestures into text
and spoken audio in real time directly through your webcam.

Author : Vision & Accessibility AI Team
Stack  : Streamlit · MediaPipe · OpenCV · WebRTC · Web Speech API
"""

import time
from collections import Counter
import streamlit as st
from PIL import Image

from dictionary_data import SIGN_DICTIONARY, get_categories, get_signs_by_category
from gesture_classifier import GestureClassifier
from hand_tracker import HandTracker
from live_stream import RTC_CONFIGURATION, LiveSignProcessor
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tts_engine import SentenceBuilder, generate_audio_bytes, speak_in_browser
from utils import (
    confidence_gauge_chart,
    cv2_to_pil,
    pil_to_cv2,
    session_metrics,
    sign_frequency_chart,
)

# ═══════════════════════════════════════════════════════════════════════
#  Page Configuration & Styling
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SignEase AI · Real-Time Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Dark CSS
st.markdown(
    """
    <style>
    /* ── Main Hero Banner ──────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #09131F 0%, #102A43 50%, #061B2E 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(0, 210, 186, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-banner h1 {
        color: #00D2BA;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #9FB3C8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Telemetry & KPI Cards ─────────────────────────────────────── */
    .telemetry-card {
        background: linear-gradient(145deg, #141C2E 0%, #0E1626 100%);
        border: 1px solid rgba(0, 210, 186, 0.2);
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 186, 0.5);
    }
    .telemetry-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #00D2BA;
    }
    .telemetry-label {
        font-size: 0.82rem;
        color: #829AB1;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Sentence Display Box ──────────────────────────────────────── */
    .sentence-box {
        background: rgba(0, 210, 186, 0.06);
        border: 2px dashed #00D2BA;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin: 1.2rem 0;
        text-align: center;
    }
    .sentence-text {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E6EDF3;
        letter-spacing: 0.5px;
    }
    .sentence-label {
        color: #829AB1;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ── Dictionary Sign Card ──────────────────────────────────────── */
    .dict-card {
        background: #141C2E;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .dict-card:hover {
        border-color: #00D2BA;
    }
    .dict-emoji {
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
