# gen2 — Face Recognition Attendance System (Clean-Room Rebuild)

A complete ground-up rebuild of the face-recognition attendance system, prioritizing **recognition correctness** over feature count.

## Prerequisites

- Python 3.10+
- macOS (Apple Silicon) or Linux
- Webcam
- ~70 MB disk for models (symlinked to existing committed models)

## Installation

```bash
# From the repository root (parent of gen2/):
cd /path/to/AttndX

# Use the existing venv (has all dependencies)
source venv/bin/activate

# Or install gen2's own requirements:
pip install -r gen2/requirements.txt
```

## Model Setup

Models are symlinked from `../models/` into `gen2/data/models/` on first run.
The symlinks point to:
- `face_detection_yunet_2023mar.onnx` (232 KB — face detection + 5 landmarks)
- `arcfaceresnet100-11-int8.onnx` (65 MB — 512-d face embeddings)
- `MiniFASNetV2.onnx` (1.7 MB — anti-spoofing)

If the models are missing, the config (`gen2/config/default.yaml`) has a
fallback path that looks in `../models/`.

## Configuration

All configuration is in `gen2/config/default.yaml`. Key settings:

```yaml
recognition:
  accept_threshold: 0.36    # cosine sim above this → KNOWN (if unambiguous)
  ambiguity_margin: 0.08   # best-second_best margin below this → AMBIGUOUS
  reject_threshold: 0.15   # below this → clearly UNKNOWN

enrollment:
  min_samples: 5            # minimum valid samples to enroll
  min_intra_similarity: 0.25  # below → INCONSISTENT_SAMPLES
  duplicate_threshold: 0.40   # above → POSSIBLE_DUPLICATE
```

**The recognition threshold (0.36) is a conservative default that MUST be
calibrated with real data using the calibration tool (see below).**

## Database Initialization

Databases are SQLite (WAL mode) and auto-create on first run:
- `gen2/data/biometric.db` — identities, embeddings, templates
- `gen2/data/attendance.db` — sessions, checks, attendance records

No manual initialization needed. The DB starts empty.

## Enrollment

1. Launch the app (see below)
2. Go to the **Register** tab
3. Enter Student ID and Name
4. Capture 5+ face images via the browser camera
5. Click **Enroll Student**

The enrollment pipeline:
- Detects face → aligns to 112×112 via similarity transform
- Quality gate (blur, brightness, contrast, size, pose, landmarks)
- ArcFace embedding (512-d, L2-normalized)
- Intra-person consistency check (rejects inconsistent samples)
- Duplicate check against existing identities
- Template computation (centroid of valid embeddings)
- Atomic SQLite persistence (identity + samples + template)
- In-memory index update (no restart needed)

## Starting the Live Application

```bash
# From the repository root:
streamlit run gen2/app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Running Tests

```bash
# From the repository root:
python -m pytest gen2/tests/ -v
```

73 tests covering: alignment, embedding, matching, enrollment, tracking,
quality, persistence, and attendance.

## Running Recognition Evaluation

```bash
# From the repository root:
python -m gen2.evaluation.calibration
```

This measures genuine/impostor similarity distributions and recommends
a calibrated threshold. **Requires enrolled identities with ≥2 samples
for meaningful results.**

## Rebuilding the Recognition Index

```bash
python gen2/scripts/rebuild_index.py
```

The index is in-memory and rebuilt from the SQLite biometric DB on startup.
This script does a manual rebuild for verification.

## Backing Up Biometric Data

```bash
python gen2/scripts/backup.py
```

Copies `biometric.db`, `attendance.db`, and WAL files to a timestamped
backup directory under `gen2/data/`.

## Recovery Procedures

### Corrupted biometric.db

If `biometric.db` is corrupted, `BiometricDB.safe_open()` will:
1. Detect the corruption on load
2. Back up the corrupt file as `biometric.corrupt.<timestamp>.db`
3. Start a fresh empty database
4. Log a CRITICAL error

Re-enroll identities from backup or from fresh captures.

### Lost index (process crash)

The in-memory index is rebuilt from `biometric.db` on every startup.
No data is lost — just restart the application.

### Aborted session

Sessions with `status='in_progress'` that were not properly stopped
remain in `attendance.db`. They can be viewed in the Reports tab and
deleted manually.

## Troubleshooting

### "Model not found"
Models are symlinked to `../models/`. Verify:
```bash
ls -la gen2/data/models/
```
If symlinks are broken, recreate them:
```bash
ln -sf ../../models/arcfaceresnet100-11-int8.onnx gen2/data/models/
ln -sf ../../models/face_detection_yunet_2023mar.onnx gen2/data/models/
ln -sf ../../models/MiniFASNetV2.onnx gen2/data/models/
```

### Camera not opening
- Close other applications using the camera (Zoom, Teams, etc.)
- On macOS: System Settings → Privacy & Security → Camera → allow terminal/streamlit

### All faces show as UNKNOWN
- The threshold (0.36) may be too high for your hardware/lighting
- Run the calibration tool to measure your genuine similarity distribution
- Lower `accept_threshold` in `gen2/config/default.yaml` if needed

### Same person registered as two identities shows AMBIGUOUS
This is **correct behavior**. The system refuses to confidently assign
an identity when two enrolled identities are too similar. This prevents
the old system's bug where the same person was registered as two students.
