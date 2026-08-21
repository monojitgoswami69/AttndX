#!/usr/bin/env python3
"""
Download the InsightFace w600k_r50.onnx recognition model.

Uses the insightface package to download the buffalo_l model pack,
then copies the recognition model (w600k_r50.onnx) to data/models/.

The w600k_r50 model:
  - Architecture: ResNet50 with ArcFace loss
  - Training data: WebFace600K (600K identities, 42M images)
  - Input: [1, 3, 112, 112] float32, NCHW, RGB, normalized to [-1, 1]
  - Output: [1, 512] float32
  - Precision: FP32 (full precision, no quantization)
"""
import shutil
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import onnxruntime as ort


def download_model():
    target_dir = REPO_ROOT / "data" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "w600k_r50.onnx"

    if target_path.exists():
        size_mb = target_path.stat().st_size / 1024 / 1024
        print(f"Model already exists at {target_path} ({size_mb:.1f} MB)")
        # Verify it loads
        try:
            sess = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()[0]
            print(f"  Input: {inp.name} {inp.shape} {inp.type}")
            print(f"  Output: {out.name} {out.shape} {out.type}")
            print("  ✅ Model loads successfully")
            return True
        except Exception as e:
            print(f"  ⚠️ Existing model failed to load: {e}")
            print("  Re-downloading...")

    # Download via insightface
    print("Downloading InsightFace buffalo_l model pack...")
    print("This includes w600k_r50.onnx (recognition model).")

    try:
        from insightface.app import FaceAnalysis
        # This downloads the buffalo_l pack to ~/.insightface/models/buffalo_l/
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1)  # CPU only for download/verification
        print("buffalo_l model pack downloaded successfully.")
    except Exception as e:
        print(f"Error downloading via FaceAnalysis: {e}")
        print("Trying direct download...")
        # Fallback: use insightface model download utility
        try:
            from insightface.utils import storage as if_storage
            if_storage.download("buffalo_l")
            print("Downloaded via storage utility.")
        except Exception as e2:
            print(f"Direct download also failed: {e2}")
            return False

    # Find the downloaded model
    home = Path.home()
    possible_paths = [
        home / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
    ]

    src_path = None
    for p in possible_paths:
        if p.exists():
            src_path = p
            break

    if src_path is None:
        # Search more broadly
        insightface_dir = home / ".insightface"
        if insightface_dir.exists():
            for f in insightface_dir.rglob("w600k_r50.onnx"):
                src_path = f
                break

    if src_path is None:
        print("❌ Could not find w600k_r50.onnx after download.")
        print("   Check ~/.insightface/models/ for the downloaded files.")
        return False

    # Copy to project model directory
    print(f"Copying {src_path} → {target_path}")
    shutil.copy2(src_path, target_path)

    # Verify
    size_mb = target_path.stat().st_size / 1024 / 1024
    print(f"Model size: {size_mb:.1f} MB")

    if size_mb < 50:
        print("⚠️ Model seems too small — may be corrupted")
        return False

    try:
        sess = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  Input: {inp.name} {inp.shape} {inp.type}")
        print(f"  Output: {out.name} {out.shape} {out.type}")

        # Verify input shape
        if inp.shape != [1, 3, 112, 112]:
            print(f"  ⚠️ Unexpected input shape: {inp.shape}")
        if out.shape != [1, 512]:
            print(f"  ⚠️ Unexpected output shape: {out.shape}")

        print("  ✅ Model verified successfully")
        return True
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
