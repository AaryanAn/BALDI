import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


class Gestures:
    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.prev_point = None
        self.smoothed_point = None
        self.paths = []            # Stores ALL completed paths
        self.current_path = None   # The path currently being drawn
        self.drawing = False
        self.still_start_time = None

        self.SMOOTHING_ALPHA = 0.35     # EMA blend factor (closer to 0 = smoother)
        self.SMOOTHING_DEADZONE = 4      # px — ignore movements smaller than this

        # Pinch detection
        # Threshold as a fraction of hand size (wrist→middle-MCP distance).
        # 0.35 = pinch when thumb/index tips are within 35% of hand length.
        self.PINCH_THRESHOLD  = 0.2

        self.PINCH_RELEASE    = 0.50    # hysteresis — must open past this to re-arm
        self._pinch_active    = False   # True while fingers are currently touching

    def detect_index_fingertip(self, frame_bgr):
        """
        Returns:
            annotated_frame (BGR),
            (x, y) pixel coords OR None
        """

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            self.prev_point   = None
            self.smoothed_point = None
            self._pinch_active  = False
            return frame_bgr, None

        hand_landmarks = result.hand_landmarks[0]
        height, width, _ = frame_bgr.shape

        def px(lm):
            return (int(lm.x * width), int(lm.y * height))

        index_tip = px(hand_landmarks[8])
        thumb_tip = px(hand_landmarks[4])
        wrist     = px(hand_landmarks[0])
        mid_mcp   = px(hand_landmarks[9])   # middle finger MCP — hand size reference

        # ── Pinch detection ───────────────────────────────────────────────────
        hand_size   = math.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
        pinch_dist  = math.hypot(thumb_tip[0] - index_tip[0],
                                 thumb_tip[1] - index_tip[1])
        norm_dist   = pinch_dist / hand_size if hand_size > 0 else 1.0

        if norm_dist < self.PINCH_THRESHOLD and not self._pinch_active:
            # Fingers just came together — toggle writing mode
            self._pinch_active = True
            self.drawing = not self.drawing
            if self.drawing:
                self.current_path = []
                self.paths.append(self.current_path)
        elif norm_dist > self.PINCH_RELEASE:
            self._pinch_active = False   # re-arm for next pinch

        # ── EMA smoothing on index fingertip ──────────────────────────────────
        raw_point = index_tip
        if self.smoothed_point is None:
            self.smoothed_point = raw_point
        else:
            ax, ay = self.smoothed_point
            bx, by = raw_point
            dx, dy = bx - ax, by - ay
            if math.hypot(dx, dy) >= self.SMOOTHING_DEADZONE:
                a = self.SMOOTHING_ALPHA
                self.smoothed_point = (int(ax + a * dx), int(ay + a * dy))

        point = self.smoothed_point
        self.update_path(point)

        # ── Draw paths ────────────────────────────────────────────────────────
        for path in self.paths:
            for i in range(1, len(path)):
                cv2.line(frame_bgr, path[i - 1], path[i], (0, 255, 255), 3)

        # Fingertip dot: green = idle, red = writing
        color = (0, 0, 255) if self.drawing else (0, 255, 0)
        cv2.circle(frame_bgr, point, 6, color, -1)

        # Pinch indicator line between thumb and index
        pinch_color = (0, 0, 255) if self._pinch_active else (200, 200, 200)
        cv2.line(frame_bgr, thumb_tip, index_tip, pinch_color, 2)

        return frame_bgr, point

    def update_path(self, point):
        if self.prev_point is None:
            self.prev_point = point
            return

        if self.drawing and self.current_path is not None:
            self.current_path.append(point)

        self.prev_point = point

    def clear_path(self):
        if self.paths:
            save_data = np.array(self.paths[0])
            # np.savetxt('../samples/sample1.csv', save_data, delimiter=',')

        self.paths        = []
        self.current_path = None
        self.drawing      = False
        self.prev_point   = None
        self._pinch_active = False
