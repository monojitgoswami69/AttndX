"""
Camera source abstraction.

Handles camera initialization, frame acquisition, error recovery,
and clean resource shutdown. Decoupled from recognition logic.

Supports:
  - cv2.VideoCapture (local camera)
  - External frame buffer (for WebRTC — frames pushed by VideoProcessor)

Privacy invariant:
  - The cv2 camera is ONLY opened when no WebRTC source is available.
  - It MUST be released when the session ends, the app shuts down,
    or the object is garbage-collected.
  - atexit and __del__ serve as last-resort safety nets.
"""
import atexit
import logging
import threading
import time

import cv2
import numpy as np

from gen2.config import Config

logger = logging.getLogger(__name__)

# Registry of all CameraSource instances for atexit cleanup
_active_cameras: list["CameraSource"] = []
_atexit_registered = False


def _atexit_release_all():
    """Release all camera devices on process exit."""
    for cam in _active_cameras:
        try:
            cam.release()
        except Exception:
            pass
    _active_cameras.clear()


class CameraSource:
    """OpenCV VideoCapture wrapper with fallback and error handling.

    Guarantees camera release via:
      1. Explicit .release() call (primary — engine/UI responsibility)
      2. __del__ (secondary — GC fallback)
      3. atexit handler (tertiary — process exit fallback)
    """

    def __init__(self, index: int | None = None):
        self.index = index if index is not None else Config.get("camera", "index")
        self.width = Config.get("camera", "width")
        self.height = Config.get("camera", "height")
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

        # Register for atexit cleanup
        _active_cameras.append(self)
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_atexit_release_all)
            _atexit_registered = True

    def __del__(self):
        """GC fallback: release camera if caller forgot."""
        try:
            self.release()
        except Exception:
            pass

    def open(self, index: int | None = None) -> bool:
        """Open the camera. Returns True on success."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            idx = index if index is not None else self.index
            # Try the requested index, then fall back to 0
            candidates = [idx]
            if idx != 0:
                candidates.append(0)

            backend_cfg = Config.get("camera", "backend", default="default")
            api_preference = cv2.CAP_DSHOW if backend_cfg == "dshow" else cv2.CAP_ANY

            for try_idx in candidates:
                try:
                    if api_preference == cv2.CAP_DSHOW:
                        self._cap = cv2.VideoCapture(try_idx, cv2.CAP_DSHOW)
                    else:
                        self._cap = cv2.VideoCapture(try_idx)

                    if not self._cap.isOpened():
                        continue
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    # Warm up: read a few frames
                    for _ in range(3):
                        ret, _ = self._cap.read()
                        if ret:
                            break
                    self.index = try_idx
                    logger.info(f"Camera opened at index {try_idx} (backend: {backend_cfg})")
                    return True
                except Exception as e:
                    logger.warning(f"Camera index {try_idx} failed: {e}")
                    self._cap = None
            return False

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame. Returns None on failure."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info("Camera released")


class ExternalFrameBuffer:
    """Latest-frame-wins buffer for WebRTC.
    No queue growth. If inference is slower than capture, stale frames are
    silently dropped — only the newest is kept."""

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        """Push a new frame (replaces any existing)."""
        with self._lock:
            self._frame = frame.copy() if frame is not None else None

    def pop(self) -> np.ndarray | None:
        """Get the latest frame. Returns None if no frame available."""
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def peek(self) -> np.ndarray | None:
        """Peek at the latest frame without consuming it."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
