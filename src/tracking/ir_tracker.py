import cv2
import numpy as np
import math
import time


class IRTracker:
    """
    Tracks an IR reflector tip from an ELP infrared camera.

    Pipeline per frame:
        1. Convert BGR → grayscale.
        2. Background subtraction — a slow EMA background model absorbs
           static IR sources (room lights, reflections).  Only new / moving
           bright spots survive in the difference image.
        3. Fixed threshold on the difference image.
        4. Morphological opening (pre-built kernel).
        5. Contour selection with a proximity gate — candidates must be
           within MAX_JUMP px of the last known position to prevent
           noise blobs from hijacking the cursor.
        6. EMA smooth the accepted center; annotate; return.

    Cursor color:
        Green — idle (not writing)
        Red   — writing mode active
    """

    # ------------------------------------------------------------------ #
    # Camera settings                                                      #
    # ------------------------------------------------------------------ #
    _AUTO_EXPOSURE = 1    # manual mode (disables auto-exposure)
    _EXPOSURE      = -9   # ELP range: -1 (bright) → -13 (dark)
    _GAIN          = 100  # digital gain

    # ------------------------------------------------------------------ #
    # Background model                                                     #
    # ------------------------------------------------------------------ #
    # The background is a slow EMA of grayscale frames.
    #   bg ← BG_ALPHA * bg + (1 - BG_ALPHA) * current
    # At 0.97 the background absorbs ~63% of any static scene in ~1 second
    # (at 30 fps).  Static IR light sources disappear; the moving reflector
    # stands out in the difference image.
    BG_ALPHA = 0.97

    # ------------------------------------------------------------------ #
    # Detection hyper-parameters (applied to the diff image)              #
    # ------------------------------------------------------------------ #
    THRESH_OFFSET  = 25   # include pixels within this many levels of the diff max
    THRESH_FLOOR   = 15   # minimum diff value to trigger any detection at all

    MIN_RADIUS      = 2   # px  — reject single-pixel noise
    MAX_RADIUS      = 80  # px  — reject huge false positives
    MIN_AREA        = 5   # px² — small reflector tip can be ~8 px²
    MIN_CIRCULARITY = 0.4 # 4π·area/perimeter²

    # ------------------------------------------------------------------ #
    # Proximity gate                                                       #
    # ------------------------------------------------------------------ #
    # Once tracking is established, only accept blobs within MAX_JUMP px
    # of the last smoothed position.  Prevents noise blobs from hijacking
    # the cursor.  When tracking is lost (smoothed_point is None) the gate
    # opens fully to allow re-acquisition anywhere in the frame.
    MAX_JUMP = 120  # px

    # ------------------------------------------------------------------ #
    # Path / drawing state                                                 #
    # ------------------------------------------------------------------ #
    STILL_THRESHOLD     = 10  # px   — movement below this = "still"
    STILL_TIME_REQUIRED = 1   # sec  — hold still to toggle drawing
    SMOOTHING_ALPHA     = 0.3 # EMA blend (higher = more responsive)
    SMOOTHING_DEADZONE  = 3   # px   — sub-pixel jitter suppression

    # Pre-built morphology kernel — avoids allocation every frame
    _MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __init__(self, camera_index: int = 1):
        self.camera_index = camera_index

        # Background model (float32, same size as camera frame, built lazily)
        self._bg_model: np.ndarray | None = None

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
            Input frame annotated in-place.
        center : tuple[int, int] | None
            EMA-smoothed (x, y) tip coordinates, or None if not detected.
        """
        # ---- 1. Grayscale -----------------------------------------------
        gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        # ---- 2. Background update & subtraction -------------------------
        #  cv2.accumulateWeighted: bg ← (1-alpha)*bg + alpha*src
        #  Using alpha = 1 - BG_ALPHA so the update weight is small.
        if self._bg_model is None:
            self._bg_model = gray_f.copy()
        else:
            cv2.accumulateWeighted(gray_f, self._bg_model, 1.0 - self.BG_ALPHA)

        bg_u8 = np.clip(self._bg_model, 0, 255).astype(np.uint8)
        diff  = cv2.subtract(gray, bg_u8)   # saturates at 0 — no negatives

        # ---- 3. Threshold on the difference image -----------------------
        #  THRESH_FLOOR guards against detecting when max_diff is just noise.
        max_diff = int(diff.max())
        if max_diff < self.THRESH_FLOOR:
            # Scene has no bright anomaly — nothing to detect
            self.smoothed_point = None
            return frame_bgr, None

        thresh = max_diff - self.THRESH_OFFSET
        _, binary = cv2.threshold(diff, max(thresh, 1), 255, cv2.THRESH_BINARY)

        # ---- 4. Morphological opening -----------------------------------
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, self._MORPH_KERNEL, iterations=1
        )

        # ---- 5. Contours + proximity-gated selection --------------------
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_center, best_radius = self._best_contour(
            contours, last_pos=self.smoothed_point
        )

        # ---- 6. EMA smoothing -------------------------------------------
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
            # raw_center → still detection (true physical motion)
            # center     → stored in path (EMA-smoothed stroke)
            self._update_path(raw_center, center)

        # ---- 7. Annotate ------------------------------------------------
        if center is not None:
            self._draw_overlay(frame_bgr, center, best_radius)

        return frame_bgr, center

    def reset_background(self):
        """Force the background model to re-learn from scratch."""
        self._bg_model = None

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
        self,
        contours: list,
        last_pos: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int] | None, int]:
        """
        Select the contour most likely to be the IR reflector tip.

        Proximity gate
        --------------
        If ``last_pos`` is given (tracking is active), candidates farther
        than ``MAX_JUMP`` pixels from that position are rejected outright.
        This prevents noise blobs elsewhere in the frame from stealing the
        cursor.  When ``last_pos`` is None (tracking just started / was
        lost), any candidate is accepted so the pen can be re-acquired.

        Scoring
        -------
        circularity × radius — prefers the roundest, largest passing blob.

        Returns
        -------
        (center, radius)  or  (None, 0)
        """
        best_center = None
        best_score  = -1.0
        best_radius = 0

        for cnt in contours:
            M    = cv2.moments(cnt)
            area = M["m00"]
            if area < self.MIN_AREA:
                continue

            cx = M["m10"] / area
            cy = M["m01"] / area

            # ── Proximity gate ──────────────────────────────────────────
            if last_pos is not None:
                dist_from_last = math.hypot(cx - last_pos[0], cy - last_pos[1])
                if dist_from_last > self.MAX_JUMP:
                    continue   # too far from known position — skip

            # ── Radius gate ─────────────────────────────────────────────
            _, radius = cv2.minEnclosingCircle(cnt)
            if not (self.MIN_RADIUS <= radius <= self.MAX_RADIUS):
                continue

            # ── Circularity gate ────────────────────────────────────────
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue

            # ── Score ───────────────────────────────────────────────────
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
        """Green = idle  |  Red = writing mode active."""
        cx, cy = center
        color  = (0, 0, 255) if self.drawing else (0, 255, 0)
        ring_r = max(radius, 8)

        cv2.circle(frame, center, ring_r, color, 2)
        cv2.circle(frame, center, 5, color, -1)
        arm = ring_r + 10
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 1)
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 1)

    def _update_path(
        self,
        raw_point:     tuple[int, int],
        smoothed_point: tuple[int, int],
    ) -> None:
        """
        Toggle drawing on/off (hold still) and record path points.

        raw_point     — used for still-vs-moving detection (true motion).
        smoothed_point — stored in the path (smooth strokes).
        """
        now = time.time()

        if self.prev_point is None:
            self.prev_point = raw_point
            return

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
            self.still_start_time = None
            if self.drawing and self.current_path is not None:
                self.current_path.append(smoothed_point)

        self.prev_point = raw_point

    # ------------------------------------------------------------------ #
    # Camera factory helper                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def open_camera(cls, camera_index: int = 1) -> cv2.VideoCapture:
        """Open and configure the ELP IR camera. Raises RuntimeError on failure."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"IRTracker: could not open camera at index {camera_index}."
            )
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cls._AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE,      cls._EXPOSURE)
        cap.set(cv2.CAP_PROP_GAIN,          cls._GAIN)
        return cap
