import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
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
        self.path = []
        self.drawing = False
        self.still_start_time = None

        self.STILL_THRESHOLD = 10        # pixels of movement
        self.STILL_TIME_REQUIRED = 0.75  # seconds


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
            return frame_bgr, None

        hand_landmarks = result.hand_landmarks[0]

        height, width, _ = frame_bgr.shape
        fingertip = hand_landmarks[8]

        x_px = int(fingertip.x * width)
        y_px = int(fingertip.y * height)
        
        point = (x_px, y_px)

        self.update_path(point)

        # Draw lines
        for i in range(1, len(self.path)):
            cv2.line(rgb_frame,
                    self.path[i - 1],
                    self.path[i],
                    (0, 255, 255),
                    3)

        # Draw dot
        color = (0, 0, 255) if self.drawing else (0, 255, 0)
        cv2.circle(frame_bgr, (x_px, y_px), 15, color, -1)


        return frame_bgr, (x_px, y_px)

    def update_path(self, point):
        now = time.time()

        if self.prev_point is None:
            self.prev_point = point
            return

        dx = point[0] - self.prev_point[0]
        dy = point[1] - self.prev_point[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # If finger is mostly still
        if distance < self.STILL_THRESHOLD:
            if self.still_start_time is None:
                self.still_start_time = now
            elif now - self.still_start_time > self.STILL_TIME_REQUIRED:
                # Toggle drawing mode
                self.path = []
                self.drawing = not self.drawing
                self.still_start_time = None
        else:
            # Reset still timer
            self.still_start_time = None

            if self.drawing:
                self.path.append(point)

        self.prev_point = point
