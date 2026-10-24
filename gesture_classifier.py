"""
Gesture Recognition & Sign Classification Engine.

Transforms 21 normalized 3D hand landmarks into semantic sign language gestures,
words, and numbers using invariant geometric feature vectors and rule-based
spatial kinematics.
"""

import math
from collections import deque
from typing import Optional


class GestureClassifier:
    """
    Classifies 21 hand landmarks into standardized signs and numbers.
    Maintains a temporal deque buffer for noise reduction and debouncing.
    """

    def __init__(self, buffer_size: int = 7) -> None:
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)

    @staticmethod
    def _dist(p1: tuple[float, ...], p2: tuple[float, ...]) -> float:
        """Euclidean distance in normalized 2D space."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_finger_states(self, lms: list[tuple[float, ...]]) -> dict[str, bool]:
        """
        Determines whether each finger is extended (True) or curled (False).
        Indices:
          Thumb:  4 (tip), 2 (mcp)
          Index:  8 (tip), 6 (pip), 5 (mcp)
          Middle: 12 (tip), 10 (pip), 9 (mcp)
          Ring:   16 (tip), 14 (pip), 13 (mcp)
          Pinky:  20 (tip), 18 (pip), 17 (mcp)
        """
        wrist = lms[0]

        # Fingers: tip compared to pip relative to wrist
        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            # Tip must be further from wrist than PIP and MCP
            d_tip_wrist = self._dist(lms[tip_idx], wrist)
            d_pip_wrist = self._dist(lms[pip_idx], wrist)
            # In typical upright view, tip y is higher (smaller value) than pip
            upward = lms[tip_idx][1] < lms[pip_idx][1]
            return (d_tip_wrist > d_pip_wrist * 1.15) or upward

        # Thumb: compare tip to MCP and IP
        d_thumb_wrist = self._dist(lms[4], wrist)
        d_mcp_wrist = self._dist(lms[2], wrist)
        thumb_extended = d_thumb_wrist > d_mcp_wrist * 1.25

        return {
            "thumb": thumb_extended,
            "index": is_extended(8, 6, 5),
            "middle": is_extended(12, 10, 9),
            "ring": is_extended(16, 14, 13),
            "pinky": is_extended(20, 18, 17),
        }

    def classify(self, lms: list[tuple[float, ...]], handedness: str = "Right") -> tuple[str, float]:
        """
        Evaluate normalized landmarks and return (gesture_name, confidence).
        """
        if len(lms) < 21:
            return "UNKNOWN", 0.0

        states = self.get_finger_states(lms)
        thumb = states["thumb"]
        index = states["index"]
        middle = states["middle"]
        ring = states["ring"]
        pinky = states["pinky"]

        wrist = lms[0]
        thumb_tip = lms[4]
        index_tip = lms[8]
        middle_tip = lms[12]
        ring_tip = lms[16]
        pinky_tip = lms[20]

        # Key spatial metrics
        thumb_index_dist = self._dist(thumb_tip, index_tip)
        index_middle_dist = self._dist(index_tip, middle_tip)
        palm_size = self._dist(wrist, lms[9]) + 1e-6  # normalize against hand size

        norm_thumb_index_dist = thumb_index_dist / palm_size
        norm_index_middle_dist = index_middle_dist / palm_size

        raw_gesture = "UNKNOWN"
        conf = 0.85

        # 1. OKAY Sign: Thumb and Index tips touch, Middle, Ring, Pinky extended
        if norm_thumb_index_dist < 0.35 and middle and ring and pinky:
            raw_gesture = "OKAY"
            conf = 0.94

        # 2. I LOVE YOU: Thumb, Index, Pinky extended; Middle, Ring curled
        elif thumb and index and not middle and not ring and pinky:
            raw_gesture = "I LOVE YOU"
            conf = 0.96

        # 3. SUPER / CALL ME: Thumb and Pinky extended; Index, Middle, Ring curled
        elif thumb and not index and not middle and not ring and pinky:
            raw_gesture = "SUPER / CALL ME"
            conf = 0.92

        # 4. PEACE / VICTORY: Index and Middle extended in V; Thumb, Ring, Pinky curled
        elif index and middle and not ring and not pinky:
            if norm_index_middle_dist > 0.25:
                raw_gesture = "PEACE / VICTORY"
                conf = 0.95
            else:
                raw_gesture = "TWO (2)"
                conf = 0.91

        # 5. WATER / 'W' sign: Index, Middle, Ring extended; Thumb and Pinky curled
        elif not thumb and index and middle and ring and not pinky:
            raw_gesture = "WATER"
            conf = 0.91

        # 6. THUMBS UP (YES) or THUMBS DOWN (NO)
        elif thumb and not index and not middle and not ring and not pinky:
            # Check thumb direction relative to wrist
            if thumb_tip[1] < wrist[1] - 0.05:  # thumb pointing upwards
                raw_gesture = "YES"
                conf = 0.93
            elif thumb_tip[1] > wrist[1] + 0.05:  # thumb pointing downwards
