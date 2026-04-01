"""
Anti-spoofing / liveness detection module.
Uses three complementary methods to detect printed photos, screen photos,
and video replays presented to the camera.

Methods:
  1. Texture analysis via Local Binary Patterns (LBP)
  2. Blink detection via Eye Aspect Ratio (EAR) over consecutive frames
  3. Frequency analysis via 2D FFT energy distribution
"""

import cv2
import numpy as np
from core.config import Config

# Try importing scikit-image LBP; fall back to manual implementation
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[Liveness] scikit-image not found — using manual LBP implementation")


class LivenessDetector:
    """Detects spoofing attempts using texture, blink, and frequency analysis."""

    def __init__(self):
        self.texture_threshold = Config.LIVENESS_TEXTURE_THRESHOLD
        self.ear_threshold = Config.LIVENESS_BLINK_EAR_THRESHOLD
        self.min_blinks = Config.LIVENESS_MIN_BLINKS
        self.frames_to_capture = Config.LIVENESS_FRAMES_TO_CAPTURE

        # Eye landmark indices for 68-point models (dlib/insightface 2D)
        # Left eye: points 36-41, Right eye: points 42-47
        self.LEFT_EYE = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE = [42, 43, 44, 45, 46, 47]

        # Face cascade for eye region extraction (fallback)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    # ──────────────────────────────────────────────
    # METHOD 1: Texture Analysis (LBP)
    # ──────────────────────────────────────────────

    def analyze_texture(self, face_image):
        """
        Analyze face texture using Local Binary Patterns.
        Real faces have diverse, rich textures.
        Printed/screen photos have uniform, repetitive patterns.

        Returns:
            {"is_live": bool, "texture_score": float, "method": "texture_analysis"}
        """
        if face_image is None or face_image.size == 0:
            return {"is_live": False, "texture_score": 0.0, "method": "texture_analysis"}

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image

        # Resize to consistent size for comparable scores
        gray = cv2.resize(gray, (128, 128))

        if HAS_SKIMAGE:
            # Use scikit-image LBP
            lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        else:
            # Manual LBP implementation
            lbp = self._manual_lbp(gray, radius=3, n_points=24)

        # Compute histogram of LBP values
        n_bins = 26  # For P=24 uniform LBP: P+2 bins
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # Texture diversity = variance of the histogram
        # Real faces: high variance (diverse patterns)
        # Fake: low variance (uniform/repetitive)
        texture_score = float(np.var(hist) * 1000)  # Scale up for readability

        # Additional: compute Laplacian variance (blur/sharpness)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Combine: texture diversity + sharpness
        # Screen photos often show moiré patterns (very high freq) or blur
        combined_score = texture_score + (laplacian_var / 500)

        is_live = combined_score >= self.texture_threshold

        return {
            "is_live": is_live,
            "texture_score": round(combined_score, 3),
            "texture_variance": round(texture_score, 3),
            "sharpness": round(laplacian_var, 1),
            "method": "texture_analysis",
        }

    def _manual_lbp(self, gray, radius=3, n_points=24):
        """Manual LBP when scikit-image is not available."""
        rows, cols = gray.shape
        result = np.zeros_like(gray, dtype=np.float64)

        for n in range(n_points):
            angle = 2.0 * np.pi * n / n_points
            dx = radius * np.cos(angle)
            dy = -radius * np.sin(angle)

            # Bilinear interpolation coordinates
            x1, y1 = int(np.floor(dx)), int(np.floor(dy))
            x2, y2 = x1 + 1, y1 + 1
            fx, fy = dx - x1, dy - y1

            # Compute for valid region only
            r_start = max(0, -min(y1, y2))
            r_end = min(rows, rows - max(y1, y2))
            c_start = max(0, -min(x1, x2))
            c_end = min(cols, cols - max(x1, x2))

            if r_start >= r_end or c_start >= c_end:
                continue

            # Bilinear interpolation
            region = gray[r_start:r_end, c_start:c_end].astype(np.float64)
            center = gray[r_start:r_end, c_start:c_end].astype(np.float64)

            try:
                interp = (
                    (1 - fx) * (1 - fy) * gray[r_start + y1:r_end + y1, c_start + x1:c_end + x1]
                    + fx * (1 - fy) * gray[r_start + y1:r_end + y1, c_start + x2:c_end + x2]
                    + (1 - fx) * fy * gray[r_start + y2:r_end + y2, c_start + x1:c_end + x1]
                    + fx * fy * gray[r_start + y2:r_end + y2, c_start + x2:c_end + x2]
                )
                bit = (interp >= center).astype(np.float64)
                result[r_start:r_end, c_start:c_end] += bit * (2 ** n)
            except (IndexError, ValueError):
                continue

        return result

    # ──────────────────────────────────────────────
    # METHOD 2: Blink Detection (EAR)
    # ──────────────────────────────────────────────

    def detect_blink(self, frames, face_detector=None):
        """
        Track Eye Aspect Ratio across consecutive frames to detect blinks.
        Real people blink naturally; photos/screens do not.

        Uses OpenCV eye detection as a practical approach:
        - If eyes are detected → open → EAR ~ HIGH
        - If eyes NOT detected → closed/blinking → EAR ~ LOW
        - If EAR fluctuates (open→closed→open) → BLINK → LIVE

        Args:
            frames: List of BGR face crop images (30-60 frames).
            face_detector: Optional face detector for landmarks.

        Returns:
            {"is_live": bool, "blinks_detected": int,
             "ear_values": list, "method": "blink_detection"}
        """
        if not frames or len(frames) < 5:
            return {
                "is_live": False, "blinks_detected": 0,
                "ear_values": [], "method": "blink_detection",
            }

        ear_values = []
        for frame in frames:
            ear = self._compute_ear_from_frame(frame)
            ear_values.append(ear)

        # Detect blinks: EAR drops below threshold then rises back
        blinks = 0
        in_blink = False
        for ear in ear_values:
            if ear < self.ear_threshold and not in_blink:
                in_blink = True
            elif ear >= self.ear_threshold and in_blink:
                blinks += 1
                in_blink = False

        is_live = blinks >= self.min_blinks

        return {
            "is_live": is_live,
            "blinks_detected": blinks,
            "ear_values": [round(v, 3) for v in ear_values],
            "method": "blink_detection",
        }

    def _compute_ear_from_frame(self, face_image):
        """
        Compute an Eye Aspect Ratio proxy using Haar cascade eye detection.

        Returns:
            float: EAR value (0.0 = eyes closed/not found, 0.3 = eyes open)
        """
        if face_image is None or face_image.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        h, w = gray.shape

        # Focus on upper half of face (eye region)
        eye_region = gray[int(h * 0.15):int(h * 0.55), :]

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(w * 0.08), int(h * 0.04)),
            maxSize=(int(w * 0.45), int(h * 0.25)),
        )

        if len(eyes) >= 2:
            # Both eyes detected → compute EAR from aspect ratios
            ears = []
            for (ex, ey, ew, eh) in eyes[:2]:
                # Aspect ratio of eye region
                ear = eh / max(ew, 1)
                ears.append(ear)
            avg_ear = float(np.mean(ears))
            # Normalize to 0.0-0.4 range
            return min(avg_ear * 0.8, 0.4)
        elif len(eyes) == 1:
            # One eye → partially visible
            ew, eh = eyes[0][2], eyes[0][3]
            return float(eh / max(ew, 1)) * 0.5
        else:
            # No eyes detected → might be blinking (or bad frame)
            return 0.05

    # ──────────────────────────────────────────────
    # METHOD 3: Frequency Analysis (FFT)
    # ──────────────────────────────────────────────

    def analyze_frequency(self, face_image):
        """
        Analyze frequency spectrum via 2D FFT.
        Real faces have natural frequency distribution.
        Screen/printed photos have unnatural high-frequency peaks.

        Returns:
            {"is_live": bool, "freq_score": float,
             "high_freq_ratio": float, "method": "frequency_analysis"}
        """
        if face_image is None or face_image.size == 0:
            return {"is_live": False, "freq_score": 0.0, "method": "frequency_analysis"}

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        gray = cv2.resize(gray, (128, 128))

        # Apply 2D FFT
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Define frequency bands
        # Low freq: center region (natural face structure)
        # Mid freq: middle ring (fine features like skin texture)
        # High freq: outer ring (noise, moiré, pixel grid)

        total_energy = float(np.sum(magnitude))
        if total_energy == 0:
            return {"is_live": False, "freq_score": 0.0, "method": "frequency_analysis"}

        # Create masks for frequency bands
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        low_mask = dist <= (min(h, w) * 0.15)
        mid_mask = (dist > (min(h, w) * 0.15)) & (dist <= (min(h, w) * 0.4))
        high_mask = dist > (min(h, w) * 0.4)

        low_energy = float(np.sum(magnitude[low_mask]))
        mid_energy = float(np.sum(magnitude[mid_mask]))
        high_energy = float(np.sum(magnitude[high_mask]))

        high_freq_ratio = high_energy / total_energy if total_energy > 0 else 0

        # Natural face: balanced energy distribution
        # Screen photo: excess high-freq energy (pixel grid/moiré)
        # Printed photo: specific mid-freq pattern (print dots)
        # Very smooth/blurred: very low high-freq → also suspicious

        # Score: how "natural" the frequency distribution looks
        # Penalize both extremes (too much or too little high-freq)
        ideal_high_ratio = 0.35  # Empirical value for real faces
        deviation = abs(high_freq_ratio - ideal_high_ratio)

        # Compute frequency score (higher = more natural/live)
        freq_score = max(0, 1.0 - deviation * 3.5)

        # Also check for periodic peaks (screen moiré)
        moire_score = self._detect_moire(magnitude, cx, cy)

        # Combine
        if moire_score > 0.5:
            freq_score *= 0.5  # Heavy penalty for moiré

        is_live = freq_score > 0.45 and moire_score < 0.5

        return {
            "is_live": is_live,
            "freq_score": round(freq_score, 3),
            "high_freq_ratio": round(high_freq_ratio, 3),
            "moire_detected": moire_score > 0.5,
            "method": "frequency_analysis",
        }

    def _detect_moire(self, magnitude, cx, cy):
        """Check for moiré pattern (periodic peaks in frequency domain)."""
        h, w = magnitude.shape

        # Look for isolated peaks in high-frequency region
        high_region = magnitude.copy()
        # Zero out low-frequency center
        r = int(min(h, w) * 0.2)
        high_region[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 0

        if high_region.max() == 0:
            return 0.0

        # Threshold to find peaks
        threshold = high_region.mean() + 2.5 * high_region.std()
        peaks = high_region > threshold
        peak_count = int(np.sum(peaks))
        total_pixels = h * w - (2 * r) ** 2

        # Moiré typically produces 2-8 sharp symmetric peaks
        peak_ratio = peak_count / max(total_pixels, 1)

        if 0.001 < peak_ratio < 0.02:
            return 0.7  # Likely moiré
        elif peak_ratio >= 0.02:
            return 0.3  # Too many peaks = noise, not moiré
        return 0.1  # Normal

    # ──────────────────────────────────────────────
    # Combined Liveness Check
    # ──────────────────────────────────────────────

    def check_liveness(self, face_crops, multi_frame_crops=None):
        """
        Run all 3 liveness methods and use voting system.

        Args:
            face_crops: Single face crop (for texture + frequency) or list.
            multi_frame_crops: List of face crops over time (for blink detection).
                             If None, blink detection is skipped.

        Returns:
            {
                "is_live": bool,
                "confidence": float (0-1),
                "methods": {
                    "texture": {"is_live": bool, "score": float},
                    "blink": {"is_live": bool, "blinks": int},
                    "frequency": {"is_live": bool, "score": float}
                },
                "spoofing_type": None | "printed_photo" | "screen_photo" | "video_replay"
            }
        """
        if not Config.LIVENESS_ENABLED:
            return {
                "is_live": True, "confidence": 1.0,
                "methods": {}, "spoofing_type": None,
            }

        # Get the primary face crop
        if isinstance(face_crops, list):
            primary = face_crops[0] if face_crops else None
        else:
            primary = face_crops

        if primary is None:
            return {
                "is_live": False, "confidence": 0.0,
                "methods": {}, "spoofing_type": "unknown",
            }

        # Run methods
        texture_result = self.analyze_texture(primary)
        frequency_result = self.analyze_frequency(primary)

        # Blink detection (only if we have multi-frame data)
        if multi_frame_crops and len(multi_frame_crops) >= 10:
            blink_result = self.detect_blink(multi_frame_crops)
        else:
            # Not enough frames for blink — mark as inconclusive (assume OK)
            blink_result = {
                "is_live": True, "blinks_detected": -1,
                "ear_values": [], "method": "blink_detection",
            }

        # Voting: 2 out of 3 say LIVE → LIVE
        votes = [
            texture_result["is_live"],
            blink_result["is_live"],
            frequency_result["is_live"],
        ]
        live_votes = sum(votes)
        is_live = live_votes >= 2

        # Confidence based on votes
        confidence = live_votes / 3.0

        # Determine spoofing type
        spoofing_type = None
        if not is_live:
            if frequency_result.get("moire_detected"):
                spoofing_type = "screen_photo"
            elif not texture_result["is_live"] and texture_result.get("sharpness", 0) < 50:
                spoofing_type = "printed_photo"
            elif not blink_result["is_live"] and blink_result["blinks_detected"] == 0:
                spoofing_type = "video_replay"
            else:
                spoofing_type = "printed_photo"  # Default

        return {
            "is_live": is_live,
            "confidence": round(confidence, 2),
            "methods": {
                "texture": {
                    "is_live": texture_result["is_live"],
                    "score": texture_result["texture_score"],
                },
                "blink": {
                    "is_live": blink_result["is_live"],
                    "blinks": blink_result["blinks_detected"],
                },
                "frequency": {
                    "is_live": frequency_result["is_live"],
                    "score": frequency_result["freq_score"],
                },
            },
            "spoofing_type": spoofing_type,
        }

    def quick_liveness_check(self, face_crop):
        """
        Fast single-frame liveness check (texture + frequency only).
        Used during live preview for responsive UI feedback.

        Returns:
            {"is_live": bool, "confidence": float, "spoofing_type": str|None}
        """
        if not Config.LIVENESS_ENABLED:
            return {"is_live": True, "confidence": 1.0, "spoofing_type": None}

        if face_crop is None or face_crop.size == 0:
            return {"is_live": False, "confidence": 0.0, "spoofing_type": "unknown"}

        texture = self.analyze_texture(face_crop)
        frequency = self.analyze_frequency(face_crop)

        votes = [texture["is_live"], frequency["is_live"]]
        is_live = sum(votes) >= 1  # At least 1 of 2 for quick check

        spoofing_type = None
        if not is_live:
            if frequency.get("moire_detected"):
                spoofing_type = "screen_photo"
            else:
                spoofing_type = "printed_photo"

        return {
            "is_live": is_live,
            "confidence": round(sum(votes) / 2.0, 2),
            "spoofing_type": spoofing_type,
        }
