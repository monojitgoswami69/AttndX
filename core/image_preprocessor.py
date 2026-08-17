"""
Image preprocessing and quality assessment module.
Checks blur, brightness, and size of face crops, and prepares
normalized face images for the embedding model.
"""

import numpy as np
import cv2
from core.config import Config


class ImagePreprocessor:
    """Assess image quality and preprocess face crops for embedding."""

    def __init__(self):
        """Initialize with config-based thresholds."""
        self.face_size = Config.FACE_INPUT_SIZE          # (112, 112)
        self.min_face_size = Config.MIN_FACE_SIZE        # 40 px
        self.blur_threshold = Config.BLUR_THRESHOLD      # 50.0
        self.brightness_low = Config.BRIGHTNESS_LOW      # 40.0
        self.brightness_high = Config.BRIGHTNESS_HIGH    # 220.0

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _compute_blur_score(self, gray_image: np.ndarray) -> float:
        """
        Compute blur score using the Laplacian variance method.
        Higher values = sharper image.

        Args:
            gray_image: Grayscale image.

        Returns:
            Laplacian variance (float).
        """
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def _compute_brightness(self, gray_image: np.ndarray) -> float:
        """
        Compute mean brightness of a grayscale image.

        Args:
            gray_image: Grayscale image.

        Returns:
            Mean pixel intensity (0-255).
        """
        return float(np.mean(gray_image))

    def assess_quality(self, face_image: np.ndarray) -> dict:
        """
        Assess the quality of a face crop.

        Args:
            face_image: BGR face crop (numpy array).

        Returns:
            Dict with:
                - score (float 0-1): Overall quality score.
                - blur (float): Laplacian variance (higher = sharper).
                - brightness (float): Mean pixel intensity.
                - size_ok (bool): Whether dimensions meet minimum.
                - issues (list[str]): Human-readable list of problems.
        """
        if face_image is None or face_image.size == 0:
            return {
                "score": 0.0,
                "blur": 0.0,
                "brightness": 0.0,
                "size_ok": False,
                "issues": ["Empty or invalid image"],
            }

        h, w = face_image.shape[:2]
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        issues = []
        sub_scores = []

        # ── Size check ──
        size_ok = h >= self.min_face_size and w >= self.min_face_size
        if not size_ok:
            issues.append(f"Face too small ({w}×{h}px, min {self.min_face_size}px)")
            sub_scores.append(0.0)
        else:
            # Score based on how far above minimum size the face is
            size_ratio = min((h * w) / (self.min_face_size ** 2), 4.0) / 4.0
            sub_scores.append(size_ratio)

        # ── Blur check ──
        blur_score = self._compute_blur_score(gray)
        if blur_score < self.blur_threshold:
            issues.append(f"Image too blurry (score: {blur_score:.1f}, need ≥{self.blur_threshold})")
            sub_scores.append(max(0.0, blur_score / self.blur_threshold))
        else:
            # Cap at 1.0 for "perfectly sharp"
            sub_scores.append(min(1.0, blur_score / (self.blur_threshold * 4)))

        # ── Brightness check ──
        brightness = self._compute_brightness(gray)
        if brightness < self.brightness_low:
            issues.append(f"Image too dark (brightness: {brightness:.1f})")
            sub_scores.append(max(0.0, brightness / self.brightness_low))
        elif brightness > self.brightness_high:
            issues.append(f"Image too bright (brightness: {brightness:.1f})")
            sub_scores.append(max(0.0, 1.0 - (brightness - self.brightness_high) / (255 - self.brightness_high)))
        else:
            # Normalize brightness score within acceptable range
            range_mid = (self.brightness_low + self.brightness_high) / 2
            range_half = (self.brightness_high - self.brightness_low) / 2
            dist = abs(brightness - range_mid) / range_half
            sub_scores.append(1.0 - dist * 0.3)  # slight penalty for being far from center

        # ── Overall score (average of sub-scores) ──
        overall = float(np.mean(sub_scores)) if sub_scores else 0.0
        overall = max(0.0, min(1.0, overall))

        return {
            "score": round(overall, 3),
            "blur": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "size_ok": size_ok,
            "issues": issues,
        }

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Preprocess a face crop for the embedding model:
        1. Resize to 112×112.
        2. Apply CLAHE to the luminance channel for contrast normalization.

        Args:
            face_image: BGR face crop.

        Returns:
            Preprocessed 112×112 BGR image, or None if input is invalid.
        """
        if face_image is None or face_image.size == 0:
            return None

        # Resize to target dimensions
        resized = cv2.resize(
            face_image,
            self.face_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert to LAB color space for CLAHE on luminance
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))
        channels[0] = self.clahe.apply(channels[0])
        lab = cv2.merge(channels)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result
