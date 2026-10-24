"""
Real-Time WebRTC Video Stream Processor for Sign Language Translation.

Processes webcam frames at high FPS, executes hand tracking & gesture kinematics,
and overlays an augmented HUD onto the live stream.
"""

import threading
import time
from typing import Optional

import av
import cv2
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

from gesture_classifier import GestureClassifier
from hand_tracker import HandTracker

# Reliable STUN infrastructure for WebRTC NAT traversal on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    }
)


class LiveSignProcessor(VideoProcessorBase):
    """
    Asynchronous WebRTC Video Processor for continuous real-time sign recognition.
    """

    def __init__(self) -> None:
        self.tracker = HandTracker(max_num_hands=1)
        self.classifier = GestureClassifier(buffer_size=8)

        # Thread synchronization
        self._lock = threading.Lock()
        self.latest_sign: str = "Awaiting hand..."
        self.latest_confidence: float = 0.0
        self.finger_states: dict[str, bool] = {}
        self.fps: float = 0.0

        # Frame timing
        self._last_time = time.time()
        self._frame_count = 0

        # Sign hold debounce counter for sentence builder
        self._current_hold_sign = ""
        self._hold_counter = 0
        self.newly_committed_sign: Optional[str] = None
        self.hold_threshold = 15  # frames required to commit a sign

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame and return annotated frame."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Calculate FPS
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_time = now

        # Run hand tracking
        annotated, hands = self.tracker.process_frame(img, draw=True)

        current_sign = "NO HAND DETECTED"
        current_conf = 0.0
        f_states = {}

        if hands and len(hands) > 0:
            primary_hand = hands[0]
            lms = primary_hand["landmarks"]
            handedness = primary_hand.get("handedness", "Right")

            f_states = self.classifier.get_finger_states(lms)
            sign, conf = self.classifier.classify(lms, handedness=handedness)
            current_sign = sign
            current_conf = conf

            # Debouncing logic to commit sign
            if current_sign != "UNKNOWN":
                if current_sign == self._current_hold_sign:
                    self._hold_counter += 1
                    if self._hold_counter == self.hold_threshold:
                        with self._lock:
                            self.newly_committed_sign = current_sign
                else:
                    self._current_hold_sign = current_sign
                    self._hold_counter = 1
            else:
                self._current_hold_sign = ""
                self._hold_counter = 0

        # Draw futuristic HUD overlay on the video frame
        self._draw_hud(annotated, current_sign, current_conf, f_states, w, h)

        # Thread-safe update
        with self._lock:
            self.latest_sign = current_sign
            self.latest_confidence = current_conf
            self.finger_states = f_states

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _draw_hud(
        self,
        img,
        sign: str,
        conf: float,
        states: dict[str, bool],
        w: int,
        h: int,
    ) -> None:
        """Render high-tech augmented reality overlay on the video feed."""
        # Top translucent status bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (10, 14, 23), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Neon divider line
        cv2.line(img, (0, 75), (w, 75), (0, 210, 186), 2)

        # Active Sign Label
        color = (0, 210, 186) if sign not in ("NO HAND DETECTED", "UNKNOWN") else (150, 150, 150)
        display_text = f"SIGN: {sign}"
        if conf > 0:
            display_text += f" ({int(conf * 100)}%)"

        cv2.putText(
            img,
            display_text,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        # FPS meter on top right
        fps_text = f"{self.fps} FPS"
        cv2.putText(
            img,
            fps_text,
            (w - 130, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        # Progress bar for sign hold / debouncing (bottom of top bar)
        if self._hold_counter > 0 and self._hold_counter <= self.hold_threshold:
            progress = min(1.0, self._hold_counter / self.hold_threshold)
            bar_w = int(w * progress)
            cv2.rectangle(img, (0, 72), (bar_w, 75), (0, 255, 128), -1)

    def get_telemetry(self) -> dict:
        """Thread-safe retrieval of latest pipeline metrics."""
        with self._lock:
            committed = self.newly_committed_sign
            self.newly_committed_sign = None  # consume single-shot event
            return {
                "sign": self.latest_sign,
                "confidence": self.latest_confidence,
                "fps": self.fps,
                "finger_states": dict(self.finger_states),
                "committed_sign": committed,
            }
