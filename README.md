# AttndX — AI Smart Face Recognition Attendance System

A production-grade face-recognition attendance system built on YuNet, ArcFace ResNet100, and MiniFASNet anti-spoofing.

## Prerequisites

- Python 3.10+
- macOS, Windows, or Linux
- Webcam

## Installation

```bash
# Clone the repository
git clone https://github.com/monojitgoswami69/AttndX.git
cd AttndX

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
# or: streamlit run app/streamlit_app.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Architecture

```
AttndX/
├── app.py                     # Main application entry point
├── app/                       # App runtime & Streamlit dashboard
├── attendance/                # Attendance session lifecycle & engine
├── config/                    # Configuration management (default.yaml)
├── data/                      # Storage for DBs, captures, snapshots, models
│   ├── captures/              # Enrolled student face captures (gitignored)
│   ├── snapshots/             # Attendance check frame snapshots (gitignored)
│   ├── models/                # ONNX model files (YuNet, ArcFace, MiniFASNet)
│   ├── biometric.db           # SQLite: identities + templates (gitignored)
│   └── attendance.db          # SQLite: sessions + check logs (gitignored)
├── enrollment/                # Biometric enrollment & consistency validation
├── evaluation/                # Threshold calibration & verification tools
├── recognition/               # ArcFace embeddings, matching, tracking, liveness
├── scripts/                   # Utility scripts (backup, rebuild_index)
├── storage/                   # SQLite database drivers
├── tests/                     # Comprehensive test suite (74 tests)
├── ui/                        # Streamlit UI page components
└── vision/                    # YuNet face detector, alignment, quality gating
```

## Configuration

All configuration is located in `config/default.yaml`. Key settings:

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

## Utilities

### Rebuild Identity Index
```bash
python scripts/rebuild_index.py
```

### Database Backup
```bash
python scripts/backup.py
```

### Calibration Tool
```bash
python -m evaluation.calibration --db data/biometric.db
```
