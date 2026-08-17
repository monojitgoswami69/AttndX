import unittest
from unittest.mock import patch

import numpy as np

from services.camera_service import CameraService


class FakeCapture:
    def __init__(self, opened, frame=None):
        self._opened = opened
        self._frame = frame

    def isOpened(self):
        return self._opened

    def set(self, *args, **kwargs):
        return True

    def read(self):
        if self._frame is None:
            return False, None
        return True, self._frame

    def release(self):
        return None


class CameraServiceTests(unittest.TestCase):
    @patch("services.camera_service.cv2.VideoCapture")
    def test_open_falls_back_to_default_camera_index(self, video_capture):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        video_capture.side_effect = [
            FakeCapture(False),
            FakeCapture(False),
            FakeCapture(False),
            FakeCapture(True, frame=frame),
        ]

        service = CameraService(camera_index=2)
        opened = service.open(2)

        self.assertTrue(opened)
        self.assertEqual(service.camera_index, 0)


if __name__ == "__main__":
    unittest.main()
