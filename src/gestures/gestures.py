import cv2
import mediapipe as mp

from gestures.pinch_hysteresis import next_pinch_state
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import threading


class Gestures:
    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Camera runs in a worker thread; UI reads paths on the main thread.
        self._lock = threading.Lock()

        self.prev_point = None
        self.smoothed_point = None
        self.paths = []            # Stores ALL completed paths
        self.current_path = None   # The path currently being drawn
        self.drawing = False
        self._was_pinching = False

        # Min distance between stroke samples (pixels) to reduce jitter
        self.STROKE_SAMPLE_MIN_PX = 10
        # Thumb tip (4) vs index tip (8); normalized by min(w,h) — same as ML_testing
        self.PINCH_ENTER_NORM = 0.042
        self.PINCH_EXIT_NORM = 0.062
        self.SMOOTHING_ALPHA = 0.25
        self.SMOOTHING_DEADZONE = 4

    def detect_index_fingertip(self, frame_bgr):
        """
        Returns:
            annotated_frame (BGR),
            (x, y) pixel coords OR None
        """

        with self._lock:
            return self._detect_index_fingertip_locked(frame_bgr)

    def _detect_index_fingertip_locked(self, frame_bgr):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            self.prev_point = None
            self.smoothed_point = None
            if self._was_pinching:
                self.drawing = False
                self.current_path = None
            self._was_pinching = False
            return frame_bgr, None

        hand_landmarks = result.hand_landmarks[0]

        height, width, _ = frame_bgr.shape
        m = float(min(width, height))
        thumb = hand_landmarks[4]
        fingertip = hand_landmarks[8]
        tx, ty = thumb.x * width, thumb.y * height
        ix, iy = fingertip.x * width, fingertip.y * height
        pinch_dist = math.hypot(tx - ix, ty - iy) / m
        pinch_active = next_pinch_state(
            self._was_pinching,
            pinch_dist,
            self.PINCH_ENTER_NORM,
            self.PINCH_EXIT_NORM,
        )

        x_px = int(fingertip.x * width)
        y_px = int(fingertip.y * height)

        raw_point = (x_px, y_px)

        if self.smoothed_point is None:
            self.smoothed_point = raw_point
        else:
            ax, ay = self.smoothed_point
            bx, by = raw_point
            dx = bx - ax
            dy = by - ay
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.SMOOTHING_DEADZONE:
                pass
            else:
                alpha = self.SMOOTHING_ALPHA
                sx = int(ax + alpha * dx)
                sy = int(ay + alpha * dy)
                self.smoothed_point = (sx, sy)

        point = self.smoothed_point

        self.update_path(point, pinch_active)

        # Draw ALL stored paths
        for path in self.paths:
            for i in range(1, len(path)):
                cv2.line(frame_bgr,
                         path[i - 1],
                         path[i],
                         (0, 255, 255),
                         3)

        # Draw fingertip dot
        color = (0, 0, 255) if self.drawing else (0, 255, 0)
        cv2.circle(frame_bgr, point, 6, color, -1)

        return frame_bgr, point

    def update_path(self, point, pinch_active):
        """Pinch thumb + index to start a stroke; release to finish. Points recorded while pinched."""

        if pinch_active and not self._was_pinching:
            self.drawing = True
            self.current_path = []
            self.paths.append(self.current_path)
            self.current_path.append(point)
        elif not pinch_active and self._was_pinching:
            self.drawing = False
            self.current_path = None

        self._was_pinching = pinch_active

        if self.prev_point is None:
            self.prev_point = point
            return

        if self.drawing and self.current_path is not None and pinch_active:
            dx = point[0] - self.prev_point[0]
            dy = point[1] - self.prev_point[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance >= self.STROKE_SAMPLE_MIN_PX:
                self.current_path.append(point)

        self.prev_point = point

    def clear_path(self):
        with self._lock:
            self.paths = []
            self.current_path = None
            self.drawing = False
            self._was_pinching = False

    def snapshot_paths(self):
        with self._lock:
            out = []
            for stroke in self.paths:
                out.append(list(stroke))
            return out

