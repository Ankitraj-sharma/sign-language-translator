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
