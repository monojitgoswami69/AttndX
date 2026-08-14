"""
Camera service module.
Provides a context-manager wrapper around OpenCV's VideoCapture
for safe camera access, frame reading, and multi-frame capture.
"""

import time
import numpy as np
import cv2
from core.config import Config


class CameraService:
    """Manages webcam access via OpenCV VideoCapture."""

    def __init__(self, camera_index: int | None = None):
        """
        Initialize the camera service.

        Args:
            camera_index: Camera device index. Defaults to Config.CAMERA_INDEX.
        """
        self.camera_index = camera_index if camera_index is not None else Config.CAMERA_INDEX
        self.cap: cv2.VideoCapture | None = None

    # ── Context Manager ──────────────────────────────

    def __enter__(self):
        """Open the camera when entering the context."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the camera when exiting the context."""
        self.release()
        return False

    # ── Core Methods ─────────────────────────────────

    def open(self, camera_index: int | None = None) -> bool:
        """
        Open the camera for capture.

        Args:
            camera_index: Optional override for camera device index.

        Returns:
            True if camera opened successfully.
        """
        # Release any existing capture
        if self.cap is not None:
            self.release()

        candidates = []
        if camera_index is not None:
            candidates.append(camera_index)
        else:
            candidates.append(self.camera_index)

        if candidates[0] != 0:
            candidates.append(0)

        last_error = None
        for idx in candidates:
            try:
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(idx)

                if not self.cap.isOpened():
                    last_error = f"camera index {idx}"
                    continue

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                for _ in range(3):
                    ret, _ = self.cap.read()
                    if ret:
                        break

                self.camera_index = idx
                print(f"[Camera] Opened camera at index {idx}.")
                return True
            except Exception as exc:
                last_error = str(exc)
                self.cap = None

        print(f"[Camera] Failed to open camera. Tried: {candidates}. Last error: {last_error}")
        self.cap = None
        return False

    def read_frame(self) -> np.ndarray | None:
        """
        Read a single frame from the camera.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None

        return frame

    def capture_frames(
        self,
        count: int,
        interval: float,
        callback=None,
    ) -> list[np.ndarray]:
        """
        Capture multiple frames with a delay between each.

        Args:
            count: Number of frames to capture.
            interval: Seconds to wait between frames.
            callback: Optional function called after each capture with
                      (frame_index, frame) as args. Useful for progress.

        Returns:
            List of captured BGR frames.
        """
        frames = []

        for i in range(count):
            frame = self.read_frame()
            if frame is not None:
                frames.append(frame)
                if callback is not None:
                    callback(i, frame)
            else:
                print(f"[Camera] Failed to read frame {i + 1}/{count}.")

            # Wait between captures (skip wait after last frame)
            if i < count - 1:
                time.sleep(interval)

        print(f"[Camera] Captured {len(frames)}/{count} frames.")
        return frames

    def is_opened(self) -> bool:
        """Check if the camera is currently open and ready."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Camera] Camera released.")

    def get_frame_size(self) -> tuple[int, int] | None:
        """
        Get the current frame dimensions.

        Returns:
            (width, height) tuple or None if camera not open.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
