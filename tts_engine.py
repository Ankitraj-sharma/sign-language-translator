"""
Text-to-Speech (TTS) Engine for SignEase AI.

Provides instant in-browser speech synthesis via the Web Speech API
and high-fidelity server-side MP3 generation using Google Text-to-Speech (gTTS).
"""

import io
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components


def speak_in_browser(text: str, lang: str = "en-US", rate: float = 1.0) -> None:
    """
    Triggers client-side browser speech synthesis using Web Speech API.
    Zero latency, runs directly on user's device audio output.
    """
    if not text or not text.strip():
        return

    # Sanitize text for JS string injection
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")

    js_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            // Cancel any pending speech
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.lang = "{lang}";
            utterance.rate = {rate};
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def generate_audio_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Synthesizes MP3 audio bytes using gTTS for playback/download.
