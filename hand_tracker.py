"""
Hand Landmark Tracking Module.

Provides robust 21-point hand landmark detection using Google MediaPipe
with an automatic OpenCV contour & convexity-defects fallback so that
the pipeline runs reliably across all Python versions and platforms.
"""

from typing import Optional
import cv2
import numpy as np

# Try importing MediaPipe
_MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False


# Hand skeleton connection pairs (MediaPipe standard)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]


class HandTracker:
    """
    Detects hands and extracts normalized 21-landmark coordinates.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.55,
