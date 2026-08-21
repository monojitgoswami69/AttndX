#!/usr/bin/env python3
"""
Model downloader for AttndX (AI Smart Face Recognition Attendance System).

Downloads and verifies all required ONNX models:
  1. Face Detector:      scrfd_10g_bnkps.onnx (InsightFace SCRFD-10G, default)
  2. Face Detector (Alt): face_detection_yunet_2023mar.onnx (OpenCV YuNet)
  3. Face Embedder:      glintr100.onnx (ArcFace ResNet100, Glint360K, 512-d)
  4. Face Embedder (Alt): w600k_r50.onnx (ArcFace ResNet50, WebFace600K)
  5. Liveness / Spoof:   MiniFASNetV2.onnx (Silent-Face-Anti-Spoofing, 80x80)

Cross-platform compatible (Windows, macOS, Linux).
"""
import argparse
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Model definitions: name -> dict of URLs (primary + mirrors), description, expected min size (bytes)
MODEL_REGISTRY = {
    "scrfd_10g_bnkps.onnx": {
        "description": "InsightFace SCRFD-10G Face Detector with 5-point Keypoints",
        "min_size_mb": 15.0,
        "urls": [
            "https://huggingface.co/Charles-Elena/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
            "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip",  # fallback as zip
        ],
        "zip_internal_path": "antelopev2/scrfd_10g_bnkps.onnx",
        "required": True,
    },
    "glintr100.onnx": {
        "description": "ArcFace ResNet100 Embedder (Glint360K, 512-d FP32)",
        "min_size_mb": 240.0,
        "urls": [
            "https://huggingface.co/Charles-Elena/antelopev2/resolve/main/glintr100.onnx",
            "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip",  # fallback as zip
        ],
        "zip_internal_path": "antelopev2/glintr100.onnx",
        "required": True,
    },
    "MiniFASNetV2.onnx": {
        "description": "MiniFASNetV2 Silent Face Anti-Spoofing Model",
        "min_size_mb": 1.5,
        "urls": [
            "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
            "https://huggingface.co/spaces/hysts/Silent-Face-Anti-Spoofing/resolve/main/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx",
        ],
        "required": True,
    },
    "face_detection_yunet_2023mar.onnx": {
        "description": "OpenCV YuNet Face Detector (Lightweight Fallback)",
        "min_size_mb": 0.2,
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        ],
        "required": False,
    },
    "w600k_r50.onnx": {
        "description": "ArcFace ResNet50 Embedder (WebFace600K, 512-d FP32)",
        "min_size_mb": 160.0,
        "urls": [
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        ],
        "zip_internal_path": "buffalo_l/w600k_r50.onnx",
        "required": False,
    },
}


def _create_ssl_context():
    """Create SSL context with fallback for systems with missing/outdated root CA certs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        return ssl.create_default_context()
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def _download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress display and atomic rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AttndX/1.0",
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    ssl_context = _create_ssl_context()

    try:
        print(f"  Downloading from: {url}")
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response, open(tmp_path, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size and total_size.isdigit() else 0
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_bytes / (1024 * 1024)
                    print(f"\r  [{pct:5.1f}%] {mb_cur:6.1f} / {mb_tot:6.1f} MB", end="", flush=True)
                else:
                    mb_cur = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded: {mb_cur:6.1f} MB", end="", flush=True)

            print()

        # Atomic rename
        if tmp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.rename(dest_path)
            return True
        return False
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def _extract_model_from_zip(zip_path: Path, internal_path: str, target_path: Path) -> bool:
    """Extract a specific ONNX file from a downloaded zip archive."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            matched = None
            filename = target_path.name

            # Look for exact or relative match
            for name in names:
                if name == internal_path or name.endswith("/" + filename) or name == filename:
                    matched = name
                    break

            if not matched:
                print(f"  ❌ Could not find {filename} inside zip archive (contents: {names[:5]}...)")
                return False

            print(f"  Extracting {matched} → {target_path.name}...")
            with zf.open(matched) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return True
    except Exception as e:
        print(f"  ❌ Failed extracting from zip: {e}")
        return False


def verify_onnx_model(model_path: Path) -> bool:
    """Verify an ONNX model file can be loaded by ONNX Runtime."""
    if not model_path.exists():
        return False
    try:
        import onnxruntime as ort
        # Use CPUExecutionProvider for standard verification
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if not inputs or not outputs:
            return False
        return True
    except Exception as e:
        print(f"  ⚠️ Model verification error for {model_path.name}: {e}")
        return False


def download_single_model(model_name: str, target_dir: Path, force: bool = False) -> bool:
    """Download a single model by name using its registered URLs."""
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        return False

    spec = MODEL_REGISTRY[model_name]
    target_path = target_dir / model_name
    min_size = spec["min_size_mb"] * 1024 * 1024

    if target_path.exists() and not force:
        file_size = target_path.stat().st_size
        if file_size >= min_size and verify_onnx_model(target_path):
            size_mb = file_size / (1024 * 1024)
            print(f"✅ {model_name} already present and valid ({size_mb:.1f} MB)")
            return True
        else:
            print(f"⚠️ {model_name} is corrupted or incomplete. Re-downloading...")

    print(f"\n📦 Fetching {model_name}: {spec['description']}")
    urls = spec["urls"]

    for url in urls:
        if url.endswith(".zip"):
            # Download zip to temp file then extract
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                tmp_zip_path = Path(tmp_zip.name)
            try:
                success = _download_file(url, tmp_zip_path, desc=f"{model_name} (archive)")
                if success:
                    internal_path = spec.get("zip_internal_path", model_name)
                    extract_ok = _extract_model_from_zip(tmp_zip_path, internal_path, target_path)
                    if extract_ok and verify_onnx_model(target_path):
                        print(f"  ✅ {model_name} verified successfully.")
                        return True
            finally:
                if tmp_zip_path.exists():
                    try:
                        tmp_zip_path.unlink()
                    except Exception:
                        pass
        else:
            success = _download_file(url, target_path, desc=model_name)
            if success and verify_onnx_model(target_path):
                print(f"  ✅ {model_name} verified successfully.")
                return True

    print(f"❌ Failed to download {model_name} from all available sources.")
    return False


def download_all_models(target_dir: Path | None = None, include_optional: bool = True, force: bool = False) -> bool:
    """Download all required (and optionally alternative) models into target_dir."""
    if target_dir is None:
        try:
            from config import Config
            target_dir = Config.model_dir()
        except Exception:
            target_dir = REPO_ROOT / "data" / "models"

    target_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"AttndX Model Manager — Destination: {target_dir}")
    print("=" * 70)

    all_ok = True
    for name, spec in MODEL_REGISTRY.items():
        if not include_optional and not spec.get("required", False):
            continue
        ok = download_single_model(name, target_dir, force=force)
        if spec.get("required", False) and not ok:
            all_ok = False

    print("=" * 70)
    if all_ok:
        print("🎉 All required models are ready for AttndX!")
    else:
        print("❌ Some required models failed to download. Please check your network connection.")
    print("=" * 70)
    return all_ok


def check_models_present(target_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Check if all required models are present on disk."""
    if target_dir is None:
        try:
            from config import Config
            target_dir = Config.model_dir()
        except Exception:
            target_dir = REPO_ROOT / "data" / "models"

    missing = []
    for name, spec in MODEL_REGISTRY.items():
        if spec.get("required", False):
            model_path = target_dir / name
            min_size = spec["min_size_mb"] * 1024 * 1024
            if not model_path.exists() or model_path.stat().st_size < min_size:
                missing.append(name)

    return (len(missing) == 0, missing)


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models for AttndX AI Smart Attendance System")
    parser.add_argument("--dir", type=str, default=None, help="Custom directory to store models (default: data/models)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files already exist")
    parser.add_argument("--required-only", action="store_true", help="Download only required models (skip optional w600k_r50 and yunet)")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), help="Download a specific model only")
    args = parser.parse_args()

    target_dir = Path(args.dir) if args.dir else None

    if args.model:
        if target_dir is None:
            target_dir = REPO_ROOT / "data" / "models"
        success = download_single_model(args.model, target_dir, force=args.force)
    else:
        success = download_all_models(
            target_dir=target_dir,
            include_optional=not args.required_only,
            force=args.force,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
