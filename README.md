# 🎓 AI Smart Attendance System

> **Real-time face recognition attendance tracking** using YOLOv8 + InsightFace + OpenCV + Streamlit.

Automatically detects and recognizes student faces via webcam, runs scheduled attendance checks during a class session, and generates present/late/absent reports — all from a single laptop webcam.

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
cd face_attendance_system
streamlit run app.py
```

The app opens at `http://localhost:8501`. AI models download automatically on first run (~200MB).

---

## 🎬 5-Minute Demo Walkthrough

The system runs in **Demo Mode** by default, compressing a full attendance session into ~90 seconds.

### Step 1 — Register Students (2 min)

1. Open the app → click **📸 Register** in the sidebar.
2. Enter **Student ID** (e.g., `STU001`) and **Name** (e.g., `Alice`).
3. Click the camera input to capture **5 face photos**.
   - Each capture runs a quality check (blur, brightness, size).
   - Green = good quality, Yellow = acceptable, Red = retake.
4. Click **✅ Register Student**.
5. Repeat for 2–3 more students with different faces.
6. Verify in the **👥 Students** tab — you'll see face thumbnails and metadata.

### Step 2 — Start Attendance Session (30 sec)

1. Switch to **📋 Attendance** tab.
2. You'll see the registered student count and their thumbnails.
3. Enter a **Class Name** (e.g., `CS 101`).
4. Click **▶️ START ATTENDANCE SESSION**.
5. The system opens the webcam and starts the live feed.

### Step 3 — Watch Live Recognition (60 sec)

- **Green boxes** appear around recognized faces with name + confidence.
- **Red boxes** appear for unknown faces.
- The status bar shows: elapsed time, checks completed, next check countdown.
- The check schedule panel shows: ✅ Done / ⏳ Next / ⬜ Pending.

### Step 4 — Automatic Checks at 20s, 40s, 60s

- At **20 seconds**: Check 1 runs — captures 5 frames, detects and matches faces.
- At **40 seconds**: Check 2 runs.
- At **60 seconds**: Check 3 runs.
- Each check saves a snapshot with annotated bounding boxes.
- Stay in front of the camera to be marked **present**!

### Step 5 — View Final Report

- After check 3 completes, the **Final Attendance Table** appears:

| Student | Check 1 | Check 2 | Check 3 | Total | Spoofed | Status  |
|---------|---------|---------|---------|-------|---------|---------|
| Alice   | ✅      | ✅      | ✅      | 3/3   | 0/3     | PRESENT |
| Bob     | ✅      | ❌      | ❌      | 1/3   | 0/3     | ABSENT  |
| Charlie | ❌      | ❌      | ❌      | 0/3   | 0/3     | ABSENT  |
| Dave    | ✅      | 🚫      | ❌      | 1/3   | 1/3     | ABSENT  |

- **PRESENT** (green): detected (live) in ≥ 2 checks
- **ABSENT** (red): detected in < 2 checks
- **🚫** = spoofing detected in that phase (does NOT count toward present)
- Download results as CSV via the **📊 Reports** tab.

---

## 🏗️ Architecture

```
face_attendance_system/
├── app.py                          # Streamlit entry point
├── requirements.txt
│
├── core/                           # ML & processing engines
│   ├── config.py                   # All settings & paths
│   ├── face_detector.py            # YOLOv8 face detection
│   ├── face_embedder.py            # InsightFace 512-d embeddings
│   ├── face_matcher.py             # Cosine similarity matching
│   └── image_preprocessor.py       # Quality checks & CLAHE
│
├── services/                       # Business logic
│   ├── registration_service.py     # Student registration pipeline
│   ├── attendance_service.py       # Session monitor (background thread)
│   └── camera_service.py           # OpenCV webcam wrapper
│
├── storage/                        # Persistence
│   ├── face_database.py            # Pickle-based face DB
│   ├── attendance_store.py         # Pickle-based attendance records
│   ├── registered_faces/           # Saved face images on disk
│   └── session_snapshots/          # Check snapshot images
│
├── ui/                             # Streamlit pages
│   ├── register_page.py            # Student registration UI
│   ├── attendance_page.py          # Live session UI
│   ├── gallery_page.py             # Student gallery
│   └── report_page.py              # Attendance reports
│
├── integration/                    # REST API for teammates
│   ├── schemas.py                  # Pydantic request/response models
│   ├── face_api.py                 # POST /register, GET /verify, etc.
│   └── attendance_api.py           # POST /session/start, etc.
│
└── utils/
    └── drawing.py                  # CV2 drawing utilities
```

### Data Flow

```
Camera → YOLOv8 (detect) → InsightFace (embed) → FaceMatcher (cosine sim) → Results
                                                         ↕
                                                   FaceDatabase (pickle)
```

### Attendance Logic

1. **Session starts** → all registered embeddings loaded into FaceMatcher.
2. **Background thread** waits for each check time (20s, 40s, 60s in demo).
3. **Each check** captures 5 frames, detects faces, generates embeddings, matches against DB.
4. **Greedy assignment** prevents the same person matching twice per frame.
5. **Anti-spoofing**: liveness check runs on each matched face. If spoofing is
   detected, the student is flagged as **spoofed** for that phase (🚫) and it
   does **NOT** count toward present.
6. **Final computation**:
   - Per-phase states: **present** / **absent** / **spoofed**
   - Global status is always **present** or **absent** (never late, never spoofed)
   - **≥ 2 present phases → PRESENT**, otherwise **ABSENT**
   - Spoofed phases do not count as present

---

## 🔌 Integration Guide (For Teammates)

### Adding to Your FastAPI Backend

```python
# your_app/main.py
from fastapi import FastAPI
import sys
sys.path.insert(0, "/path/to/face_attendance_system")

from core.face_detector import YOLOFaceDetector
from core.face_embedder import FaceEmbedder
from storage.face_database import FaceDatabase
from storage.attendance_store import AttendanceStore
from services.attendance_service import AttendanceMonitor
from integration.face_api import create_face_router
from integration.attendance_api import create_attendance_router

app = FastAPI(title="Smart Classroom Backend")

# Initialize shared components (do this once at startup)
detector = YOLOFaceDetector()
embedder = FaceEmbedder()
face_db = FaceDatabase()
att_store = AttendanceStore()
monitor = AttendanceMonitor(detector, embedder, face_db, att_store)

# Mount the routers
app.include_router(create_face_router(detector, embedder, face_db))
app.include_router(create_attendance_router(monitor, att_store, face_db))
```

### API Endpoints

#### Face Registration

| Method | Endpoint                   | Description                  |
|--------|----------------------------|------------------------------|
| POST   | `/api/face/register`       | Register with base64 images  |
| GET    | `/api/face/verify/{id}`    | Check if student registered  |
| DELETE | `/api/face/{id}`           | Remove a student             |
| GET    | `/api/face/students`       | List all registered students |

#### Attendance

| Method | Endpoint                        | Description                |
|--------|---------------------------------|----------------------------|
| POST   | `/api/attendance/session/start` | Start attendance session   |
| GET    | `/api/attendance/session/status`| Get live session status    |
| POST   | `/api/attendance/session/stop`  | Stop session early         |
| GET    | `/api/attendance/results/{id}`  | Get session results        |
| GET    | `/api/attendance/sessions`      | List all past sessions     |

### Example: Register via API

```python
import requests
import base64

# Encode face images
with open("face1.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/api/face/register", json={
    "student_id": "STU001",
    "name": "Alice",
    "images": [img_b64]
})
print(response.json())
# {"success": true, "registered_count": 1, "quality_scores": [0.85], ...}
```

### Example: Run Attendance Session via API

```python
import requests
import time

# Start session
r = requests.post("http://localhost:8000/api/attendance/session/start", json={
    "class_name": "CS 101",
    "camera_source": 0
})
session_id = r.json()["session_id"]

# Poll status
while True:
    status = requests.get("http://localhost:8000/api/attendance/session/status").json()
    print(f"Checks: {status['checks_completed']}/{status['total_checks']}")
    if not status["active"]:
        break
    time.sleep(5)

# Get final results
results = requests.get(f"http://localhost:8000/api/attendance/results/{session_id}").json()
for r in results["results"]:
    print(f"{r['name']}: {r['status']}")
```

---

## ⚙️ Configuration

Edit `core/config.py` to tune the system:

| Setting                  | Default     | Description                              |
|--------------------------|-------------|------------------------------------------|
| `DEMO_MODE`              | `True`      | Use seconds instead of minutes for checks|
| `CHECK_TIMES_DEMO`       | `[20,40,60]`| Check times in seconds (demo)            |
| `CHECK_TIMES_NORMAL`     | `[15,30,45]`| Check times in minutes (production)      |
| `AVAILABLE_CLASSES`      | `[...]`     | Hardcoded list of classes for dropdown    |
| `SIMILARITY_THRESHOLD`   | `0.72`      | Cosine similarity cutoff for matching    |
| `IMAGES_PER_REGISTRATION`| `5`         | Faces to capture per student             |
| `MIN_CHECKS_FOR_PRESENT` | `2`         | Minimum present phases for "present"     |
| `FRAMES_PER_CHECK`       | `5`         | Frames sampled during each check         |
| `YOLO_CONFIDENCE`        | `0.5`       | YOLO detection confidence threshold      |

---

## 📋 Requirements

- Python 3.10+
- Webcam (built-in or USB)
- ~1GB disk space for models (auto-downloaded)
- Works on Windows, macOS, and Linux

---

## 🛠️ Tech Stack

| Component       | Technology                     |
|-----------------|--------------------------------|
| Face Detection  | YOLOv8n (Ultralytics)          |
| Face Embeddings | InsightFace buffalo_l (512-d)  |
| Matching        | Cosine Similarity (SciPy)      |
| Camera          | OpenCV VideoCapture            |
| UI              | Streamlit                      |
| REST API        | FastAPI + Pydantic v2          |
| Storage         | Pickle (file-based)            |
