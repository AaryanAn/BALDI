import cv2
import numpy as np
import math
import time


class IRTracker:
    """
    Tracks an IR reflector tip from an ELP infrared camera.

    Pipeline per frame (optimised for throughput and robustness):
        1. Convert BGR → grayscale.
        2. Adaptive-max threshold: threshold = max(gray) - THRESH_OFFSET.
           The IR reflector is always the absolute brightest object,
           so anchoring the threshold to the frame maximum works in any
           lighting condition and avoids per-frame histogram computation.
        3. One morphological opening pass (pre-built kernel) to remove
           single-pixel salt noise.
        4. Find external contours; pick the best blob using cv2.moments
           (no per-pixel mask allocation).
        5. EMA smooth the center; annotate; return.

    Cursor color:
        Green  — tracker active, idle (not writing)
        Red    — writing mode engaged
    """

    # ------------------------------------------------------------------ #
    # Camera settings                                                      #
    # ------------------------------------------------------------------ #
    _AUTO_EXPOSURE = 1    # manual mode (disables auto-exposure)
    _EXPOSURE      = -8  # ELP range: -1 (bright) → -13 (dark)
    _GAIN          = 100    # digital gain

    # ------------------------------------------------------------------ #
    # Detection hyper-parameters                                           #
    # ------------------------------------------------------------------ #
    # Adaptive threshold: captures pixels within THRESH_OFFSET gray levels
    # of the brightest pixel.  The IR reflector dot is always the max;
    # THRESH_OFFSET=40 gives a comfortable margin while still rejecting
    # everything else.  THRESH_FLOOR prevents false detections when the
    # entire frame is dark (e.g., lens cap on).
    THRESH_OFFSET  = 40   # gray levels below the frame maximum to include
    THRESH_FLOOR   = 130  # absolute minimum threshold (no detection below)

    MIN_RADIUS     = 2    # px  — reject single-pixel noise
    MAX_RADIUS     = 80   # px  — reject huge false positives
    MIN_AREA       = 5    # px² — small reflector tip can be ~8px²
    MIN_CIRCULARITY = 0.4  # 4π·area/perimeter² — pen tip ≈ 0.8+

    # ------------------------------------------------------------------ #
    # Path / drawing state                                                 #
    # ------------------------------------------------------------------ #
    STILL_THRESHOLD    = 10   # px   — movement below this = "still"
    STILL_TIME_REQUIRED = 1 # sec  — hold still to toggle drawing
    SMOOTHING_ALPHA    = 0.3  # EMA blend (higher = more responsive)
    SMOOTHING_DEADZONE = 3    # px   — sub-pixel jitter suppression

    # Pre-built morphology kernel — avoids allocation every frame
    _MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __init__(self, camera_index: int = 1):
        self.camera_index = camera_index

        # Path state — identical public API to Gestures
        self.paths:        list[list[tuple[int, int]]] = []
        self.current_path: list[tuple[int, int]] | None = None
        self.drawing:      bool = False
        self.prev_point:   tuple[int, int] | None = None
        self.still_start_time: float | None = None
        self.smoothed_point:   tuple[int, int] | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect_ir_tip(self, frame_bgr: np.ndarray):
        """
        Detect the IR reflector tip in a BGR frame.

        Returns
        -------
        frame_bgr : np.ndarray
            Input frame annotated in-place (no copy made).
        center : tuple[int, int] | None
            EMA-smoothed (x, y) tip coordinates, or None if not detected.
        """
        # ---- 1. Grayscale -----------------------------------------------
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- 2. Adaptive-max threshold ----------------------------------
        #  The reflector is always the brightest object.  Anchoring the
        #  threshold to (max - THRESH_OFFSET) automatically works whether
        #  the room is bright or dim, without any histogram computation.
        max_val = int(gray.max())
        thresh  = max(max_val - self.THRESH_OFFSET, self.THRESH_FLOOR)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # ---- 3. Morphological opening — removes salt noise --------------
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, self._MORPH_KERNEL, iterations=1
        )

        # ---- 4. Find contours and pick best blob ------------------------
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_center, best_radius = self._best_contour(contours)

        # ---- 5. EMA smoothing -------------------------------------------
        if raw_center is None:
            self.smoothed_point = None
            center = None
        else:
            if self.smoothed_point is None:
                self.smoothed_point = raw_center
            else:
                ax, ay = self.smoothed_point
                bx, by = raw_center
                dx, dy = bx - ax, by - ay
                if math.hypot(dx, dy) >= self.SMOOTHING_DEADZONE:
                    a = self.SMOOTHING_ALPHA
                    self.smoothed_point = (
                        int(ax + a * dx),
                        int(ay + a * dy),
                    )

            center = self.smoothed_point
            # Pass raw_center for still-detection (true physical movement)
            # and the smoothed center for actual path recording.
            self._update_path(raw_center, center)

        # ---- 6. Annotate on the smoothed center -------------------------
        if center is not None:
            self._draw_overlay(frame_bgr, center, best_radius)

        return frame_bgr, center

    def clear_path(self):
        """Reset all stored paths and drawing state."""
        self.paths         = []
        self.current_path  = None
        self.drawing       = False
        self.prev_point    = None
        self.still_start_time = None
        self.smoothed_point   = None

    def snapshot_paths(self) -> list[list[tuple[int, int]]]:
        """Return a deep copy of all stored stroke paths (mirrors Gestures)."""
        return [list(stroke) for stroke in self.paths]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _best_contour(
        self, contours: list
    ) -> tuple[tuple[int, int] | None, int]:
        """
        Pick the contour most likely to be the IR reflector tip.

        Scoring uses circularity × radius (no per-pixel mask needed).
        Returns (center, radius) or (None, 0).
        """
        best_center    = None
        best_score     = -1.0
        best_radius    = 0

        for cnt in contours:
            # Fast area + centroid via moments
            M    = cv2.moments(cnt)
            area = M["m00"]
            if area < self.MIN_AREA:
                continue

            cx = M["m10"] / area
            cy = M["m01"] / area

            # Radius gate (cheap)
            _, radius = cv2.minEnclosingCircle(cnt)
            if not (self.MIN_RADIUS <= radius <= self.MAX_RADIUS):
                continue

            # Circularity gate
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue

            # Score: prefer the roundest + largest blob
            score = circularity * radius
            if score > best_score:
                best_score  = score
                best_center = (int(cx), int(cy))
                best_radius = int(radius)

        return best_center, best_radius

    def _draw_overlay(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
    ) -> None:
        """
        Draw detection overlay in-place.

        Green = idle  |  Red = writing mode active
        """
        cx, cy = center
        color  = (0, 0, 255) if self.drawing else (0, 255, 0)
        ring_r = max(radius, 8)

        cv2.circle(frame, center, ring_r, color, 2)          # outer ring
        cv2.circle(frame, center, 5, color, -1)              # center dot
        arm = ring_r + 10
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 1)  # H arm
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 1)  # V arm

    def _update_path(
        self,
        raw_point: tuple[int, int],
        smoothed_point: tuple[int, int],
    ) -> None:
        """
        Toggle drawing on/off and record path points.

        Parameters
        ----------
        raw_point : tuple
            The raw detected blob center (NOT EMA-filtered).  Used for
            still-vs-moving detection so EMA lag does not mask real motion.
        smoothed_point : tuple
            The EMA-filtered center.  Recorded into the path for smooth
            strokes — only used for storage, not for motion decisions.
        """
        now = time.time()

        if self.prev_point is None:
            self.prev_point = raw_point
            return

        # Use raw movement to judge whether the pen is physically still.
        # EMA shrinks apparent movement, so comparing smoothed→smoothed
        # would falsely trigger the "still" branch while the pen is moving.
        distance = math.hypot(
            raw_point[0] - self.prev_point[0],
            raw_point[1] - self.prev_point[1],
        )

        if distance < self.STILL_THRESHOLD:
            if self.still_start_time is None:
                self.still_start_time = now
            elif now - self.still_start_time > self.STILL_TIME_REQUIRED:
                self.drawing = not self.drawing
                self.still_start_time = None
                if self.drawing:
                    self.current_path = []
                    self.paths.append(self.current_path)
        else:
            # Pen is moving — reset still timer and record smoothed position.
            self.still_start_time = None
            if self.drawing and self.current_path is not None:
                self.current_path.append(smoothed_point)

        self.prev_point = raw_point

    # ------------------------------------------------------------------ #
    # Camera factory helper                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def open_camera(cls, camera_index: int = 1) -> cv2.VideoCapture:
        """
        Open and configure the ELP IR camera.

        Raises RuntimeError if the camera cannot be opened.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"IRTracker: could not open camera at index {camera_index}."
            )
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cls._AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE,      cls._EXPOSURE)
        cap.set(cv2.CAP_PROP_GAIN,          cls._GAIN)
        return cap
