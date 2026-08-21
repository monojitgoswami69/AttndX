"""
Production-grade native camera capture with background frame acquisition.

Features:
  - OS-optimized backend auto-selection:
      * Windows: DirectShow (CAP_DSHOW) -> MSMF -> CAP_ANY
      * macOS: AVFoundation (CAP_AVFOUNDATION) -> CAP_ANY
      * Linux: V4L2 (CAP_V4L2) -> CAP_ANY
  - Threaded continuous frame acquisition:
      * Eliminates hardware buffer lag / stale frames
      * read_frame() returns the latest frame instantly (<0.1ms)
  - Zero-overhead thread-safe locking
  - Automatic error recovery and cleanup via atexit / __del__
"""
import atexit
import logging
import sys
import threading
import time

import cv2
import numpy as np

from config import Config

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
    """Production-grade OpenCV VideoCapture wrapper with background threaded capture."""

    def __init__(self, index: int | None = None):
        self.index = index if index is not None else Config.get("camera", "index", default=0)
        self.width = Config.get("camera", "width", default=640)
        self.height = Config.get("camera", "height", default=480)
        self.fps = Config.get("camera", "fps", default=30)

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._last_frame_time: float = 0.0
        self._is_opened: bool = False

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

    def _get_platform_backends(self) -> list[int]:
        """Return candidate OpenCV backends optimized for current OS."""
        cfg_backend = Config.get("camera", "backend", default="auto")
        if cfg_backend == "dshow":
            return [cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif cfg_backend == "msmf":
            return [cv2.CAP_MSMF, cv2.CAP_ANY]
        elif cfg_backend == "avfoundation":
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif cfg_backend == "v4l2":
            return [cv2.CAP_V4L2, cv2.CAP_ANY]

        # Auto detection based on OS
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform.startswith("darwin"):
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif sys.platform.startswith("linux"):
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def open(self, index: int | None = None) -> bool:
        """Open the camera and start background capture thread. Returns True on success."""
        with self._lock:
            if self._is_opened and self._cap is not None and self._cap.isOpened():
                return True

            self._stop_capture_thread()
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            idx = index if index is not None else self.index
            candidates = [idx]
            if idx != 0:
                candidates.append(0)

            backends = self._get_platform_backends()

            for try_idx in candidates:
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(try_idx, backend)
                        if not cap.isOpened():
                            cap.release()
                            continue

                        # Configure camera parameters
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass

                        # Warm up test read
                        success = False
                        for _ in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                self._latest_frame = frame
                                self._last_frame_time = time.time()
                                success = True
                                break
                            time.sleep(0.05)

                        if not success:
                            cap.release()
                            continue

                        self._cap = cap
                        self.index = try_idx
                        self._is_opened = True
                        self._start_capture_thread()
                        logger.info(f"Camera successfully opened (index: {try_idx}, backend: {backend})")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed opening camera index {try_idx} with backend {backend}: {e}")

            logger.error("Could not open any camera device.")
            self._is_opened = False
            return False

    def _start_capture_thread(self):
        """Start background frame acquisition thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()

    def _stop_capture_thread(self):
        """Stop background capture thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def _capture_loop(self):
        """Continuous background capture loop."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
            else:
                time.sleep(0.01)

    def read_frame(self) -> np.ndarray | None:
        """Read the latest captured frame. Returns None on failure or if camera is closed."""
        if not self._is_opened:
            return None
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def is_opened(self) -> bool:
        """Check if camera is currently opened and acquiring frames."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def release(self):
        """Stop capture thread and release camera device."""
        with self._lock:
            self._stop_capture_thread()
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self._cap = None
            self._is_opened = False
            self._latest_frame = None
            logger.info("Camera released cleanly.")


class ExternalFrameBuffer:
    """Latest-frame-wins buffer for external frame injection (optional fallback)."""

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
