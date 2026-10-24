# 🤟 SignEase AI — Real-Time Sign Language to Speech Translator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Computer Vision](https://img.shields.io/badge/CV-MediaPipe%20%7C%20OpenCV-brightgreen.svg)](https://mediapipe.dev)
[![WebRTC](https://img.shields.io/badge/Streaming-WebRTC-orange.svg)](https://webrtc.org)
[![TTS](https://img.shields.io/badge/Speech-Web%20Speech%20API-teal.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade Computer Vision and Assistive Technology** web application that translates American & Indian Sign Language gestures into text and spoken voice in real time directly through your webcam.

---

## 🌟 Key Highlights for Competitions & Hackathons

- 🔴 **Zero Upload Continuous Live Stream**: Built on WebRTC — just click START and make signs. The app translates live video at 30 FPS.
- 🖐️ **21 3D Hand Landmark Tracking**: Tracks every finger joint, knuckle, and fingertip using MediaPipe with automatic OpenCV fallback.
- 🗣️ **Instant Voice Synthesis (TTS)**: Translates signs into audible voice using the browser's Web Speech API and generates downloadable MP3 summaries via gTTS.
- 💬 **Sentence Builder with Debouncing**: Accumulates words over time into meaningful sentences with intelligent noise filtering and undo support.
- 📖 **Interactive Sign Dictionary**: Complete visual reference library of signs across Greetings, Needs, Expressions, and Numbers.
- 📊 **Telemetry & Analytics Dashboard**: Real-time confidence gauges, gesture frequency charts, and translation transcripts.

---

## 🏗️ Technical Architecture

```
┌────────────────────────────────────────────────────────┐
│            User Browser / Mobile Webcam                │
│      Live MediaStream via WebRTC (P2P Audio/Video)     │
└──────────────────────────┬─────────────────────────────┘
                           │  Continuous 30 FPS Video
                           ▼
┌────────────────────────────────────────────────────────┐
│            HandTracker (MediaPipe / CV Hybrid)         │
│    Extracts 21 3D Spatial Landmarks (x, y, z)          │
│    Skin Tone Invariant · Distance & Scale Normalized   │
└──────────────────────────┬─────────────────────────────┘
                           │  Normalized Kinematic Coordinates
                           ▼
┌────────────────────────────────────────────────────────┐
│             GestureClassifier Kinematic Engine         │
│    Vector Geometry · Finger Extension Angle Analysis   │
│    Temporal Sliding Window Debouncing & Noise Filter   │
└──────────────────────────┬─────────────────────────────┘
                           │  Recognized Semantic Sign Token
                           ▼
┌────────────────────────────────────────────────────────┐
│          SentenceBuilder & Speech Synthesis            │
│    Accumulates Words · Client-side Web Speech API      │
│    Instant Zero-Latency Audio Output on Device         │
└────────────────────────────────────────────────────────┘
```

---

## 📖 Supported Vocabulary & Gestures

| Category | Gestures / Words | Description |
|----------|-----------------|-------------|
| **Greetings** | `HELLO`, `THANK YOU`, `PLEASE` | Open palm waves, flat-hand gratitude gestures |
| **Responses** | `YES`, `NO`, `OKAY` | Thumbs up (YES), Thumbs down (NO), OK circle |
| **Needs & Emergency** | `HELP`, `WATER`, `STOP` | Fist alert, 'W' shape (WATER), Flat palm (STOP) |
| **Expressions** | `I LOVE YOU`, `PEACE / VICTORY`, `CALL ME` | ASL ILY symbol, V-sign, Shaka sign |
| **Numbers** | `0`, `1`, `2`, `3`, `4`, `5` | Single and multi-finger count standards |

---

## 📂 Project Structure

```
sign-language-translator/
├── app.py                  # Main Streamlit application with live WebRTC HUD
├── hand_tracker.py         # 21 3D hand landmark detection & skeleton renderer
├── gesture_classifier.py   # Spatial kinematics & temporal debouncing engine
├── tts_engine.py           # In-browser Web Speech API & gTTS audio generation
├── live_stream.py          # Asynchronous WebRTC video stream processor
├── dictionary_data.py      # Structured sign metadata, categories & tips
├── utils.py                # Plotly analytics charts, telemetry, and formatting
├── requirements.txt        # Python library dependencies
├── packages.txt            # Debian system dependencies for Cloud deploy
├── .streamlit/
│   └── config.toml         # Cyberpunk dark theme configuration
├── .gitignore
├── LICENSE                 # MIT License
└── README.md               # Project documentation & benchmarks
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.10+
- Webcam / Camera

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ankitraj-sharma/sign-language-translator.git
cd sign-language-translator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository: `Ankitraj-sharma/sign-language-translator`.
4. Set main file path to: `app.py`.
5. Click **Deploy!**

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Built with ❤️ for Assistive Technology & Accessibility</b>
</div>
