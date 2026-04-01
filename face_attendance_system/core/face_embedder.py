"""
Face embedding module.
Generates 512-dimensional L2-normalized embeddings for face images.

Strategy:
  1. Try InsightFace (if installed) — full-featured, best accuracy.
  2. Fallback to ONNX Runtime with a direct ArcFace model download
     — no C compilation needed, works on any Python version.
"""

import numpy as np
import cv2
import os
import urllib.request
from pathlib import Path
from core.config import Config

# Model download URLs (tried in order) and paths
_MODEL_DIR = Config.BASE_DIR / "models"
_ARCFACE_FILE = "arcfaceresnet100-11-int8.onnx"
_ARCFACE_URLS = [
    # Official ONNX Model Zoo — verified working, 65MB, 512-d output
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx",
]


def _download_with_progress(url: str, dest: str) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    try:
        print(f"[FaceEmbedder] Trying: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB chunks
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        print(f"\r[FaceEmbedder] Downloaded {mb:.1f}MB ({pct}%)", end="", flush=True)
            print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n[FaceEmbedder] Download failed from {url}: {e}")
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        return False


class FaceEmbedder:
    """Generate face embeddings using InsightFace or ONNX ArcFace fallback."""

    def __init__(self):
        self.embedding_dim = Config.EMBEDDING_DIM
        self._backend = None  # "insightface" or "onnx"
        self._app = None      # InsightFace app
        self._ort_session = None  # ONNX Runtime session

        # Try InsightFace first, then ONNX fallback
        if self._try_insightface():
            self._backend = "insightface"
            print("[FaceEmbedder] Using InsightFace backend.")
        elif self._try_onnx_arcface():
            self._backend = "onnx"
            print("[FaceEmbedder] Using ONNX ArcFace backend (fallback).")
        else:
            raise RuntimeError(
                "Could not load any face embedding model. "
                "Install insightface or ensure onnxruntime is available."
            )

    # ──────────────────────────────────────────────
    # Backend Initialization
    # ──────────────────────────────────────────────

    def _try_insightface(self) -> bool:
        """Try loading InsightFace. Returns True on success."""
        try:
            from insightface.app import FaceAnalysis

            try:
                app = FaceAnalysis(
                    name=Config.INSIGHTFACE_MODEL,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                app = FaceAnalysis(
                    name=Config.INSIGHTFACE_MODEL,
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=-1, det_size=(640, 640))

            self._app = app
            return True
        except Exception as e:
            print(f"[FaceEmbedder] InsightFace not available: {e}")
            return False

    def _try_onnx_arcface(self) -> bool:
        """Try loading ArcFace via ONNX Runtime. Downloads model if needed."""
        try:
            import onnxruntime as ort

            model_path = _MODEL_DIR / _ARCFACE_FILE

            # Download model if not present
            if not model_path.exists():
                print(f"[FaceEmbedder] ArcFace model not found. Downloading (~120MB)...")
                _MODEL_DIR.mkdir(parents=True, exist_ok=True)

                downloaded = False
                for url in _ARCFACE_URLS:
                    if _download_with_progress(url, str(model_path)):
                        downloaded = True
                        break

                if not downloaded:
                    print(
                        "[FaceEmbedder] Could not download ArcFace model from any source.\n"
                        "Please manually download w600k_r50.onnx and place it in:\n"
                        f"  {model_path}"
                    )
                    return False

                print(f"[FaceEmbedder] Model saved to {model_path}")

            # Load ONNX session
            providers = ort.get_available_providers()
            self._ort_session = ort.InferenceSession(
                str(model_path), providers=providers
            )
            print(f"[FaceEmbedder] ONNX ArcFace loaded. Providers: {providers}")
            return True
        except Exception as e:
            print(f"[FaceEmbedder] ONNX fallback failed: {e}")
            return False

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def _normalize_l2(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding
        return embedding / norm

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Extract a 512-d L2-normalized embedding from a face crop.

        Args:
            face_image: BGR numpy array containing a face crop.

        Returns:
            512-d L2-normalized numpy array, or None if failed.
        """
        if face_image is None or face_image.size == 0:
            return None

        if self._backend == "insightface":
            return self._get_embedding_insightface(face_image)
        elif self._backend == "onnx":
            return self._get_embedding_onnx(face_image)
        return None

    def get_embeddings_batch(self, images: list[np.ndarray]) -> list[np.ndarray | None]:
        """Extract embeddings for a batch of face images."""
        return [self.get_embedding(img) for img in images]

    # ──────────────────────────────────────────────
    # InsightFace Backend
    # ──────────────────────────────────────────────

    def _get_embedding_insightface(self, face_image: np.ndarray) -> np.ndarray | None:
        """Extract embedding using InsightFace."""
        try:
            h, w = face_image.shape[:2]
            if h < 20 or w < 20:
                return None

            # Upscale small crops
            min_dim = min(h, w)
            if min_dim < 112:
                scale = 112 / min_dim
                face_image = cv2.resize(face_image, None, fx=scale, fy=scale)

            faces = self._app.get(face_image)
            if not faces:
                return None

            largest = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            embedding = largest.embedding
            if embedding is None:
                return None

            embedding = self._normalize_l2(embedding.flatten())
            if embedding.shape[0] != self.embedding_dim:
                return None

            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[FaceEmbedder] InsightFace error: {e}")
            return None

    # ──────────────────────────────────────────────
    # ONNX ArcFace Backend
    # ──────────────────────────────────────────────

    def _preprocess_arcface(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face crop for ArcFace ONNX model.
        Resizes to 112x112, converts BGR→RGB, normalizes, transposes to NCHW.
        """
        # Resize to 112x112
        img = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5

        # HWC → CHW → NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def _get_embedding_onnx(self, face_image: np.ndarray) -> np.ndarray | None:
        """Extract embedding using ONNX Runtime ArcFace."""
        try:
            h, w = face_image.shape[:2]
            if h < 10 or w < 10:
                return None

            # Preprocess
            input_tensor = self._preprocess_arcface(face_image)

            # Run inference
            input_name = self._ort_session.get_inputs()[0].name
            outputs = self._ort_session.run(None, {input_name: input_tensor})

            embedding = outputs[0].flatten()

            # L2 normalize
            embedding = self._normalize_l2(embedding)

            # Verify dimension (ArcFace w600k_r50 outputs 512-d)
            if embedding.shape[0] != self.embedding_dim:
                print(
                    f"[FaceEmbedder] ONNX output dim mismatch: "
                    f"expected {self.embedding_dim}, got {embedding.shape[0]}"
                )
                # Still return if usable
                if embedding.shape[0] > 0:
                    return embedding[:self.embedding_dim].astype(np.float32) \
                        if embedding.shape[0] >= self.embedding_dim \
                        else None

            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[FaceEmbedder] ONNX error: {e}")
            return None
