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
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_mediapipe = _MEDIAPIPE_AVAILABLE

        if self.use_mediapipe:
            self.detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self.detector = None

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> tuple[np.ndarray, Optional[list[dict]]]:
        """
        Process an image/video frame, draw skeletons, and return landmarks.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV / WebRTC.
        draw : bool
            Whether to overlay skeleton and keypoints.

        Returns
        -------
        tuple[np.ndarray, Optional[list[dict]]]:
            Annotated frame and list of hands, each containing:
            - "landmarks": list of 21 (x, y, z) tuples normalized (0..1)
            - "pixel_landmarks": list of 21 (px, py) integer tuples
            - "handedness": 'Left' or 'Right'
            - "bbox": (x_min, y_min, x_max, y_max)
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        hands_data: list[dict] = []

        if self.use_mediapipe and self.detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)

            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    handedness_label = "Right"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness_label = results.multi_handedness[idx].classification[0].label

                    norm_coords: list[tuple[float, float, float]] = []
                    pixel_coords: list[tuple[int, int]] = []
                    xs, ys = [], []

                    for lm in hand_lms.landmark:
                        norm_coords.append((lm.x, lm.y, lm.z))
                        px, py = int(lm.x * w), int(lm.y * h)
                        pixel_coords.append((px, py))
                        xs.append(px)
                        ys.append(py)

                    bbox = (
                        max(0, min(xs) - 20),
                        max(0, min(ys) - 20),
                        min(w, max(xs) + 20),
                        min(h, max(ys) + 20),
                    )

                    hands_data.append(
                        {
                            "landmarks": norm_coords,
                            "pixel_landmarks": pixel_coords,
                            "handedness": handedness_label,
                            "bbox": bbox,
                        }
                    )

                    if draw:
                        self._draw_custom_skeleton(annotated, pixel_coords, bbox, handedness_label)

                return annotated, hands_data

        # Fallback: Color-space contour & convexity defects detector
        hands_data = self._fallback_contour_detector(frame, annotated, draw)
        return annotated, hands_data if hands_data else None

    def _draw_custom_skeleton(
        self,
        img: np.ndarray,
        pts: list[tuple[int, int]],
        bbox: tuple[int, int, int, int],
        handedness: str,
    ) -> None:
        """Draw a high-tech cyan/emerald hand skeleton with glowing joints."""
        # Draw connections
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(img, pts[p1], pts[p2], (0, 210, 186), 3, cv2.LINE_AA)
            cv2.line(img, pts[p1], pts[p2], (255, 255, 255), 1, cv2.LINE_AA)

        # Draw landmark points
        for i, (px, py) in enumerate(pts):
            # Fingertips (4, 8, 12, 16, 20) highlighted in vivid amber
            if i in (4, 8, 12, 16, 20):
                cv2.circle(img, (px, py), 7, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (px, py), 4, (255, 200, 0), -1, cv2.LINE_AA)

        # Bounding box with rounded look
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 210, 186), 2, cv2.LINE_AA)

        # Hand tag
