"""
Light monitoring module for the Face Attendance System.
Detects darkness, enhances low-light frames, and manages
auto-pause/resume during attendance sessions.
"""

import numpy as np
import cv2
import time
import threading
from core.config import Config


class LightMonitor:
    """Monitors ambient brightness and enhances low-light frames."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def check_brightness(self, frame: np.ndarray) -> dict:
        """
        Assess frame brightness and return a status dict.

        Args:
            frame: BGR image.

        Returns:
            Dict with brightness, is_dark, is_too_bright, is_usable,
            quality_label, and message.
        """
        if frame is None or frame.size == 0:
            return {
                "brightness": 0.0,
                "uniformity": 0.0,
                "is_dark": True,
                "is_too_bright": False,
                "is_usable": False,
                "is_recoverable": False,
                "quality_label": "DARK",
                "message": "No frame available",
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        uniformity = float(np.std(gray))

        b_min = Config.BRIGHTNESS_MIN
        b_max = Config.BRIGHTNESS_MAX
        b_recover = Config.BRIGHTNESS_RECOVERABLE

        is_dark = brightness < b_min
        is_too_bright = brightness > b_max
        is_usable = not is_dark and not is_too_bright
        is_recoverable = brightness >= b_recover and brightness < b_min

        if brightness < b_recover:
            quality_label = "DARK"
            message = f"Too dark to recover (brightness {brightness:.0f})"
        elif brightness < b_min:
            quality_label = "LOW_LIGHT"
            message = f"Low light — enhancing (brightness {brightness:.0f})"
        elif brightness > b_max:
            quality_label = "TOO_BRIGHT"
            message = f"Too bright (brightness {brightness:.0f})"
        else:
            quality_label = "GOOD"
            message = f"Good lighting (brightness {brightness:.0f})"

        return {
            "brightness": round(brightness, 1),
            "uniformity": round(uniformity, 1),
            "is_dark": is_dark,
            "is_too_bright": is_too_bright,
            "is_usable": is_usable,
            "is_recoverable": is_recoverable,
            "quality_label": quality_label,
            "message": message,
        }

    def enhance_low_light(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Enhance a low-light frame to make face detection viable.

        If brightness is between BRIGHTNESS_RECOVERABLE and BRIGHTNESS_MIN:
          - Apply CLAHE on LAB L-channel (clipLimit=3.0, 8x8 grid)
          - Apply gamma correction (gamma=1.5)
          - Apply denoising
          - Return enhanced frame

        If brightness < BRIGHTNESS_RECOVERABLE: return None (unusable).

        Args:
            frame: BGR image.

        Returns:
            Enhanced BGR frame, or None if too dark to recover.
        """
        if frame is None or frame.size == 0:
            return None

        status = self.check_brightness(frame)

        if status["quality_label"] == "DARK":
            return None  # Too dark to recover

        if status["is_usable"]:
            return frame  # Already OK, no enhancement needed

        # --- Enhancement pipeline ---

        # 1. CLAHE on LAB L-channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))
        channels[0] = self.clahe.apply(channels[0])
        lab = cv2.merge(channels)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Gamma correction (gamma=1.5 brightens)
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        # 3. Denoise (low-light images are noisy)
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced, None, h=6, hForColoredComponents=6,
            templateWindowSize=7, searchWindowSize=21,
        )

        return enhanced

    def monitor_continuous(self, camera, callback, interval=5, stop_event=None):
        """
        Continuously monitor brightness, calling callback with status.

        Args:
            camera: CameraService instance (must be opened).
            callback: Function called with brightness status dict.
            interval: Seconds between checks.
            stop_event: threading.Event to signal stop.
        """
        if stop_event is None:
            stop_event = threading.Event()

        while not stop_event.is_set():
            frame = camera.read_frame()
            if frame is not None:
                status = self.check_brightness(frame)
                callback(status)
            stop_event.wait(timeout=interval)
