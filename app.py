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
    .dict-name {
        color: #00D2BA;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .dict-desc {
        color: #9FB3C8;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ═══════════════════════════════════════════════════════════════════════
if "accumulated_words" not in st.session_state:
    st.session_state.accumulated_words = []
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""
if "tracker" not in st.session_state:
    st.session_state.tracker = HandTracker(max_num_hands=1)
if "classifier" not in st.session_state:
    st.session_state.classifier = GestureClassifier()

# ═══════════════════════════════════════════════════════════════════════
#  Hero Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <h1>🤟 SignEase AI — Real-Time Sign Language Translator</h1>
        <p>Bridging communication barriers with real-time hand kinematics, visual translation,
        and continuous voice synthesis directly through your browser webcam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Translation Controls")
    st.markdown("---")

    voice_lang = st.selectbox(
        "🗣️ Speech Voice Accent",
        options=["en-US (US English)", "en-IN (Indian English)", "en-GB (British English)"],
        index=0,
        help="Select the accent for in-browser speech synthesis.",
    )
    lang_code = voice_lang.split()[0]

    speech_rate = st.slider(
        "⚡ Speech Speed",
        min_value=0.7,
        max_value=1.3,
        value=1.0,
        step=0.1,
        help="Controls the speed of the spoken voice.",
    )

    auto_speak = st.checkbox(
        "🔊 Auto-Speak Committed Signs",
        value=True,
        help="Automatically speaks each sign aloud as soon as it is held and recognized.",
    )

    hold_frames = st.slider(
        "⏱️ Sign Hold Threshold",
        min_value=8,
        max_value=25,
        value=14,
        help="Frames a sign must be held continuously to commit into the sentence.",
    )

    st.markdown("---")
    st.markdown("### 📈 Session Counters")
    st.metric("Total Translated Signs", len(st.session_state.session_history))
    st.metric("Words in Current Sentence", len(st.session_state.accumulated_words))

    st.markdown("---")
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.accumulated_words = []
        st.session_state.session_history = []
        st.session_state.last_spoken_text = ""
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  Application Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_live, tab_snap, tab_dict, tab_analytics, tab_about = st.tabs(
    [
        "🔴 Live Webcam Stream",
        "📸 Snapshot Translation",
        "📖 Sign Dictionary",
        "📊 Analytics Dashboard",
        "ℹ️ System Architecture",
    ]
)

# ─────────────────────────────────────────────────────────────────────
#  TAB 1 : Real-Time Live Webcam Stream (No Upload Needed!)
# ─────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("### 🔴 Real-Time Continuous Sign Translator")
    st.markdown(
        """
        <div style="background: rgba(0, 210, 186, 0.08); border-left: 4px solid #00D2BA; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #00D2BA; font-size: 1.05rem;">✨ Continuous Live Video — Zero File Uploads Needed!</b><br>
            Click <b>START</b> below to open your camera. Make any sign in front of the lens.
            Hold the sign steady for ~1 second to automatically add it to your sentence and speak it aloud!
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_video, col_hud = st.columns([1.5, 1.0])

    with col_video:
        ctx = webrtc_streamer(
            key="signease-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=LiveSignProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.hold_threshold = hold_frames

    with col_hud:
        st.markdown("#### ⚡ Live Recognition Telemetry")

        if ctx.state.playing and ctx.video_processor:
            telemetry = ctx.video_processor.get_telemetry()
            current_sign = telemetry["sign"]
            conf = telemetry["confidence"]
            fps = telemetry["fps"]
            committed = telemetry["committed_sign"]

            # Telemetry display
            entry = SIGN_DICTIONARY.get(current_sign)
            emoji_icon = entry["emoji"] if entry else "✋"

            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div style="font-size: 2.5rem;">{emoji_icon}</div>
                    <div class="telemetry-val">{current_sign}</div>
                    <div class="telemetry-label">Active Detected Gesture</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(confidence_gauge_chart(conf), use_container_width=True, key="live_gauge")

            st.caption(f"⚡ Processing Stream: **{fps} FPS**")

            # Handle newly committed sign from live stream
            if committed and committed not in ("NO HAND DETECTED", "UNKNOWN"):
                clean_word = committed.split("(")[0].strip()
                if not st.session_state.accumulated_words or st.session_state.accumulated_words[-1] != clean_word:
                    st.session_state.accumulated_words.append(clean_word)
                    st.session_state.session_history.append(
                        {
                            "sign": clean_word,
                            "confidence": conf,
                            "timestamp": time.time(),
                        }
                    )
                    if auto_speak:
                        speak_in_browser(clean_word, lang=lang_code, rate=speech_rate)

        else:
            st.markdown(
                """
                <div class="telemetry-card">
                    <div style="font-size: 2.5rem;">📷</div>
                    <div class="telemetry-val" style="color: #829AB1;">Camera Idle</div>
                    <div class="telemetry-label">Click START on the left to begin streaming</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(confidence_gauge_chart(0.0), use_container_width=True, key="idle_gauge")

    # ── Sentence Construction Bar ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Real-Time Sentence Builder")

    current_sentence = " ".join(st.session_state.accumulated_words)
    display_sentence = current_sentence if current_sentence else "Awaiting your first sign..."

    st.markdown(
        f"""
        <div class="sentence-box">
            <div class="sentence-label">Translated Spoken Sentence</div>
            <div class="sentence-text">{display_sentence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio Action Buttons
    c_speak, c_clear, c_undo = st.columns([1, 1, 1])
    with c_speak:
        if st.button("🔊 Speak Full Sentence", use_container_width=True, disabled=not current_sentence):
            speak_in_browser(current_sentence, lang=lang_code, rate=speech_rate)
    with c_clear:
        if st.button("🧹 Clear Sentence", use_container_width=True, disabled=not current_sentence):
            st.session_state.accumulated_words = []
            st.rerun()
    with c_undo:
        if st.button("⬅️ Undo Last Word", use_container_width=True, disabled=not current_sentence):
            if st.session_state.accumulated_words:
                st.session_state.accumulated_words.pop()
                st.rerun()

# ─────────────────────────────────────────────────────────────────────
#  TAB 2 : Snapshot & Single Image Mode
# ─────────────────────────────────────────────────────────────────────
with tab_snap:
    st.markdown("### 📸 Snapshot & Image Translation")
    st.caption("Capture a still photo or upload an image to inspect 21 3D hand joints and semantic translation.")

    col_cam, col_upload = st.columns(2)
    with col_cam:
        snap_file = st.camera_input("Take a photo", key="snap_camera")
    with col_upload:
        up_file = st.file_uploader(
            "Or upload an image file",
            type=["jpg", "jpeg", "png", "webp"],
            key="snap_upload",
        )

    target_image = snap_file or up_file

    if target_image is not None:
        pil_image = Image.open(target_image).convert("RGB")
        bgr_frame = pil_to_cv2(pil_image)

        tracker = st.session_state.tracker
        classifier = st.session_state.classifier

