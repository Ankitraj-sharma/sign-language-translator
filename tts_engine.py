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
    """
    if not text or not text.strip():
        return None

    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        return None


class SentenceBuilder:
    """
    Accumulates recognized gestures into words and full sentences,
    filtering consecutive duplicates and handling spaces.
    """

    def __init__(self, debounce_frames: int = 15) -> None:
        self.debounce_frames = debounce_frames
        self.current_candidate: str = ""
        self.candidate_count: int = 0
        self.words: list[str] = []
        self.last_spoken: str = ""

    def update(self, detected_gesture: str) -> Optional[str]:
        """
        Process a new frame gesture.
        Returns the committed word/phrase when stability threshold is reached, else None.
        """
        if not detected_gesture or detected_gesture == "UNKNOWN":
            self.current_candidate = ""
            self.candidate_count = 0
            return None

        if detected_gesture == self.current_candidate:
            self.candidate_count += 1
            if self.candidate_count == self.debounce_frames:
                # Commit sign to words list if not duplicate of last word
                clean_sign = detected_gesture.split("(")[0].strip()
                if not self.words or self.words[-1] != clean_sign:
                    self.words.append(clean_sign)
                    return clean_sign
        else:
            self.current_candidate = detected_gesture
            self.candidate_count = 1

        return None

    def get_sentence(self) -> str:
        """Return the accumulated sentence."""
        return " ".join(self.words)

    def clear(self) -> None:
        """Clear accumulated sentence and candidate state."""
        self.words = []
        self.current_candidate = ""
        self.candidate_count = 0
        self.last_spoken = ""
