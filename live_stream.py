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

