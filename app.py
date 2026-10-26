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
            media_stream_constraints={
                "video": {
                    "facingMode": "user",
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                },
                "audio": False,
            },
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

        with st.spinner("Analyzing hand landmarks & gesture kinematics..."):
            annotated_bgr, hands = tracker.process_frame(bgr_frame, draw=True)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### 🎯 Landmark Skeleton")
            st.image(annotated_bgr, channels="BGR", use_container_width=True)

        with res_col2:
            st.markdown("#### 📋 Kinematic Analysis")
            if hands and len(hands) > 0:
                h_data = hands[0]
                sign, conf = classifier.classify(h_data["landmarks"], handedness=h_data.get("handedness", "Right"))
                f_states = classifier.get_finger_states(h_data["landmarks"])
                dict_entry = SIGN_DICTIONARY.get(sign)

                st.success(f"### Recognized: **{sign}** ({int(conf * 100)}% Conf)")
                if dict_entry:
                    st.markdown(f"**Meaning:** {dict_entry['meaning']}")
                    st.markdown(f"**How to perform:** {dict_entry['how_to_perform']}")

                st.markdown("##### 🖐️ Finger Extension States")
                f_cols = st.columns(5)
                for i, (fname, is_ext) in enumerate(f_states.items()):
                    with f_cols[i]:
                        status_str = "🟢 UP" if is_ext else "🔴 DOWN"
                        st.metric(fname.capitalize(), status_str)

                # Audio synthesis button
                if st.button("🔊 Speak this Sign", key="snap_speak"):
                    speak_in_browser(sign, lang=lang_code, rate=speech_rate)

                # Log into history
                st.session_state.session_history.append(
                    {"sign": sign, "confidence": conf, "timestamp": time.time()}
                )
            else:
                st.warning("No hand detected in the image. Please ensure the hand is clearly visible.")

# ─────────────────────────────────────────────────────────────────────
#  TAB 3 : Interactive Sign Dictionary
# ─────────────────────────────────────────────────────────────────────
with tab_dict:
    st.markdown("### 📖 Supported Sign Language Dictionary")
    st.caption("Browse through the vocabulary of gestures supported by SignEase AI with performance guidelines.")

    categories = get_categories()
    selected_cat = st.radio("Filter by Category", ["All Categories"] + categories, horizontal=True)

    entries = (
        list(SIGN_DICTIONARY.values())
        if selected_cat == "All Categories"
        else get_signs_by_category(selected_cat)
    )

    grid_cols = st.columns(3)
    for idx, entry in enumerate(entries):
        col = grid_cols[idx % 3]
        with col:
            st.markdown(
                f"""
                <div class="dict-card">
                    <div class="dict-emoji">{entry['emoji']}</div>
                    <div class="dict-name">{entry['label']}</div>
                    <div style="color: #00D2BA; font-size: 0.8rem; margin-top: 2px;">Category: {entry['category']}</div>
                    <div class="dict-desc"><b>Meaning:</b> {entry['meaning']}</div>
                    <div class="dict-desc" style="margin-top: 6px;"><b>How to do:</b> {entry['how_to_perform']}</div>
                    <div class="dict-desc" style="color: #829AB1; font-size: 0.8rem; margin-top: 4px;"><b>Finger State:</b> {entry['finger_status']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────
#  TAB 4 : Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Translation Session Analytics")
    st.caption("Aggregated frequency and confidence telemetry across this interactive session.")

    history = st.session_state.session_history
    if not history:
        st.info("No translation events recorded yet. Start the live webcam or analyze photos to view session charts!")
    else:
        stats = session_metrics(history)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signs Recognized", stats["total_translations"])
        m2.metric("Unique Signs Used", stats["unique_signs"])
        m3.metric("Average Confidence", f"{stats['avg_confidence']}%")
        m4.metric("Top Sign", stats["top_sign"])

        st.markdown("---")

        c_chart1, c_chart2 = st.columns([1.5, 1.0])
        with c_chart1:
            signs_list = [h["sign"] for h in history]
            st.plotly_chart(sign_frequency_chart(signs_list), use_container_width=True)

        with c_chart2:
            st.markdown("#### 💾 Export Translation Transcript")
            full_sentence = " ".join(st.session_state.accumulated_words)
            if full_sentence:
                st.text_area("Full Transcript", value=full_sentence, height=120)
                audio_bytes = generate_audio_bytes(full_sentence)
                if audio_bytes:
                    st.download_button(
                        label="⬇️ Download Spoken Audio (MP3)",
                        data=audio_bytes,
                        file_name="signease_speech.mp3",
                        mime="audio/mp3",
                        use_container_width=True,
                    )
            else:
                st.caption("Create a sentence in Tab 1 to download transcript and audio.")

# ─────────────────────────────────────────────────────────────────────
#  TAB 5 : System Architecture & Technical Research
# ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ### 🏗️ Technical Architecture & Kinematic Principles

        **SignEase AI** operates on a modular, end-to-end Computer Vision & Natural Language Speech synthesis pipeline:

        ```
        ┌────────────────────────────────────────────────────────┐
        │            User Browser / Mobile Webcam                │
        │      Live MediaStream via WebRTC (P2P Audio/Video)     │
        └──────────────────────────┬─────────────────────────────┘
                                   │  Continuous 30 FPS Stream
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │            HandTracker (MediaPipe / CV Hybrid)         │
        │    Extracts 21 3D Spatial Landmarks (x, y, z)          │
        │    Skin Tone Invariant · Distance & Scale Normalized   │
        └──────────────────────────┬─────────────────────────────┘
                                   │  Normalized Kinematic Coords
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │             GestureClassifier Kinematic Engine         │
        │    Vector Geometry · Finger Extension Angle Analysis   │
        │    Temporal Sliding Window Debouncing & Noise Filter   │
        └──────────────────────────┬─────────────────────────────┘
                                   │  Recognized Semantic Token
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │          SentenceBuilder & Speech Synthesis            │
        │    Accumulates Words · Client-side Web Speech API      │
        │    Instant Zero-Latency Audio Output on Device         │
        └────────────────────────────────────────────────────────┘
        ```

        ---

        #### 🌟 Key Technical Differentiators for Judges

        1. **Zero-Upload Continuous WebRTC Streaming**: Users do not have to capture or upload files; the stream runs live with augmented HUD overlays.
        2. **Robust Multi-Platform Fallback**: Seamlessly works with MediaPipe 21-joint skeleton when present, with automatic OpenCV contour/defects fallback.
        3. **Dual In-Browser & Server Audio Pipeline**: In-browser instant voice synthesis via Web Speech API combined with offline MP3 export via gTTS.
        4. **Temporal Stability Filtering**: Sliding-window debouncing eliminates single-frame wobbles and ensures high precision.

        ---

        <div style="text-align: center; color: #829AB1; padding: 1rem;">
            Built with ❤️ for Accessibility & Inclusive Communication · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
