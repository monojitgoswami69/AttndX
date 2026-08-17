"""
Attendance monitoring service.
Manages live attendance sessions with scheduled face-recognition checks
running in a background thread. Includes darkness detection, auto-pause/resume,
retry logic, and low-light frame enhancement.
"""

import time
import threading
import datetime
import cv2
import numpy as np
from pathlib import Path

from core.config import Config
from core.face_detector import YOLOFaceDetector
from core.face_embedder import FaceEmbedder
from core.face_matcher import FaceMatcher
from core.light_monitor import LightMonitor
from core.liveness_detector import LivenessDetector
from storage.face_database import FaceDatabase
from storage.attendance_store import AttendanceStore
from services.camera_service import CameraService


class AttendanceMonitor:
    """Manages a live attendance session with background scheduled checks."""

    def __init__(self, face_detector, face_embedder, face_database, attendance_store, twin_handler=None):
        self.detector = face_detector
        self.embedder = face_embedder
        self.face_db = face_database
        self.attendance_store = attendance_store
        self.matcher = FaceMatcher()
        self.camera = CameraService()
        self.light_monitor = LightMonitor()
        self.liveness_detector = LivenessDetector()
        self.twin_handler = twin_handler

        # External-frame mode (WebRTC VideoProcessor pushes frames instead of
        # the monitor owning a cv2.VideoCapture). When True, the monitor reads
        # frames from _external_frame_buffer (populated by
        # process_external_frame()) and never opens the cv2 camera.
        self._use_external_camera = False
        self._external_frame_lock = threading.Lock()
        self._external_frame_buffer = None  # latest raw frame from WebRTC

        # Session state
        self.session_id = None
        self.session_active = False
        self.session_start_time = 0.0
        self.check_times = []
        self.checks_completed = 0
        self.total_checks = 0
        self.current_check_running = False
        self.class_name = ""

        # Pause state
        self._paused = False
        self._pause_start_time = 0.0
        self._total_paused_duration = 0.0
        self._pause_reason = ""

        # Brightness state (shared with UI)
        self._brightness_lock = threading.Lock()
        self._brightness_status = {
            "brightness": 128.0,
            "quality_label": "GOOD",
            "message": "",
            "is_usable": True,
            "is_recoverable": False,
        }
        self._frame_enhanced = False

        # Retry state
        self._retry_info = {
            "active": False,
            "check_number": 0,
            "attempt": 0,
            "max_attempts": Config.MAX_RETRIES_PER_CHECK,
            "seconds_remaining": 0,
        }

        # Twin / uncertain tracking
        self._review_queue = []  # list of uncertain detection dicts
        self._uncertain_in_frame = []  # current frame uncertain matches

        # Liveness / anti-spoofing tracking
        self._spoofing_count = 0

        # Thread control
        self._thread = None
        self._stop_event = threading.Event()

        # Thread-safe frame sharing
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_detections = []
        self._latest_names = []
        self._latest_matches = []  # raw match dicts for twin rendering
        self._latest_spoofed = []  # indices of spoofed detections in latest frame

        # Status
        self._status_lock = threading.Lock()
        self._status_message = "Idle"
        self._next_check_time = 0.0

    # ──────────────────────────────────────────────
    # Session Lifecycle
    # ──────────────────────────────────────────────

    def start_session(self, class_name, camera_index=None, use_external_camera=False):
        """Start attendance session. Returns session_id or None.

        Args:
            class_name: The class/session label.
            camera_index: cv2 camera index (ignored when use_external_camera).
            use_external_camera: When True, frames are fed via
                process_external_frame() (e.g. from a streamlit-webrtc
                VideoProcessor) and no cv2.VideoCapture is opened.
        """
        if self.session_active:
            return None

        all_embeddings = self.face_db.get_all_embeddings()
        if not all_embeddings:
            print("[Monitor] No registered students.")
            return None

        self.matcher = FaceMatcher()
        if self.twin_handler is not None:
            self.matcher.set_twin_handler(self.twin_handler)
        self.matcher.load_database(all_embeddings)

        self._use_external_camera = use_external_camera
        if not use_external_camera:
            cam_idx = camera_index if camera_index is not None else Config.CAMERA_INDEX
            if not self.camera.open(cam_idx):
                return None
        else:
            print("[Monitor] External-camera mode — frames come from WebRTC.")

        self.check_times = Config.get_check_times()
        self.total_checks = len(self.check_times)
        self.session_id = self.attendance_store.create_session(class_name)
        self.class_name = class_name
        self.session_active = True
        self.session_start_time = time.time()
        self.checks_completed = 0
        self._paused = False
        self._total_paused_duration = 0.0
        self._frame_enhanced = False
        self._retry_info["active"] = False
        self._spoofing_count = 0
        self._review_queue = []
        self._stop_event.clear()
        self._set_status(f"Session started — {self.total_checks} checks scheduled")

        self._thread = threading.Thread(
            target=self._run_scheduled_checks,
            args=(self.session_id,), daemon=True,
        )
        self._thread.start()

        unit = "seconds" if Config.DEMO_MODE else "minutes"
        print(f"[Monitor] Session '{self.session_id}' started. Checks at {self.check_times} {unit}.")
        return self.session_id

    def stop_session(self):
        """Stop session early, compute final from completed checks."""
        if not self.session_active or not self.session_id:
            return None

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        # Store paused duration
        if self._paused:
            self._total_paused_duration += time.time() - self._pause_start_time
        self.attendance_store.add_paused_duration(self.session_id, self._total_paused_duration)

        final = self._compute_final(self.session_id)
        session = self.attendance_store.get_session(self.session_id)
        if session and session["status"] not in ("cancelled_dark",):
            session["status"] = "stopped"
            self.attendance_store.save()

        self._cleanup()
        self._set_status("Session stopped")
        return final

    def _cleanup(self):
        if not self._use_external_camera:
            self.camera.release()
        self.session_active = False
        self.session_id = None
        self._latest_frame = None
        self._latest_detections = []
        self._latest_names = []
        self._latest_matches = []
        self._latest_spoofed = []
        self._paused = False
        self._frame_enhanced = False
        with self._external_frame_lock:
            self._external_frame_buffer = None

    # ──────────────────────────────────────────────
    # External-frame mode (WebRTC VideoProcessor support)
    # ──────────────────────────────────────────────

    def _is_camera_ready(self) -> bool:
        """True if a frame source (cv2 camera or external buffer) is available."""
        if self._use_external_camera:
            with self._external_frame_lock:
                return self._external_frame_buffer is not None
        return self.camera.is_opened()

    def _read_frame(self):
        """Read a frame from the cv2 camera or the external WebRTC buffer."""
        if self._use_external_camera:
            with self._external_frame_lock:
                buf = self._external_frame_buffer
            return buf.copy() if buf is not None else None
        return self.camera.read_frame()

    def process_external_frame(self, frame):
        """Process a frame pushed by the WebRTC VideoProcessor.

        Runs the live-preview pipeline (brightness → detect → embed → match)
        and updates the shared display state. The check thread also reads
        from the raw-frame buffer this method populates.

        Returns the annotated frame for the browser to display.
        """
        if frame is None or frame.size == 0:
            return frame

        # Buffer the raw frame for the check thread
        with self._external_frame_lock:
            self._external_frame_buffer = frame.copy()

        # Brightness check
        brightness_status = self.light_monitor.check_brightness(frame)
        self._update_brightness(brightness_status)

        working_frame = frame
        enhanced = False

        # If paused (darkness), just surface the raw frame with a warning
        if self._paused:
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_detections = []
                self._latest_names = []
                self._latest_matches = []
                self._latest_spoofed = []
                self._frame_enhanced = False
            from utils.drawing import draw_darkness_warning
            return draw_darkness_warning(frame.copy())

        # Low-light enhancement
        if brightness_status["quality_label"] == "LOW_LIGHT" and Config.LOW_LIGHT_ENHANCE:
            enhanced_frame = self.light_monitor.enhance_low_light(frame)
            if enhanced_frame is not None:
                working_frame = enhanced_frame
                enhanced = True

        # Detect + embed + match
        detections = self.detector.detect_faces(working_frame)
        names = []
        matches_raw = []
        spoofed_indices = []
        if detections:
            embeddings = [self.embedder.get_embedding(d["cropped_face"]) for d in detections]
            # Independent matching: every face finds its own best identity.
            # Multiple faces (real person + phone photo) can both resolve to
            # the same student — spoofing is handled below by the liveness
            # layer, not by greedy one-to-one assignment.
            matches_raw = self.matcher.find_all_matches_independent(embeddings)

            for idx, m in enumerate(matches_raw):
                if m and m.get("uncertain"):
                    tc = m.get("twin_conflict", {})
                    names.append(f"{tc.get('name_a','?')}/{tc.get('name_b','?')}?")
                elif m:
                    # Real-time liveness check on each recognized face.
                    # Flags phone/screen/print spoofs with a yellow box.
                    if Config.LIVENESS_ENABLED and Config.LIVENESS_IN_PREVIEW:
                        face_crop = detections[idx]["cropped_face"] if idx < len(detections) else None
                        face_bbox = detections[idx].get("bbox") if idx < len(detections) else None
                        if face_crop is not None and face_crop.size > 0:
                            liveness = self.liveness_detector.quick_liveness_check(
                                face_crop, frame=working_frame, face_bbox=face_bbox)
                            if not liveness["is_live"]:
                                spoof_type = liveness.get("spoofing_type", "unknown")
                                m["spoofed"] = True
                                m["spoof_type"] = spoof_type
                                spoofed_indices.append(idx)
                                names.append(f"SPOOF — {m['name']} ({spoof_type})")
                                continue
                    names.append(f"{m['name']} ({m['confidence']:.0%})")
                else:
                    names.append(None)

        with self._frame_lock:
            self._latest_frame = working_frame
            self._latest_detections = detections
            self._latest_names = names
            self._latest_matches = matches_raw
            self._latest_spoofed = spoofed_indices
            self._frame_enhanced = enhanced

        # Return fully-annotated frame (face boxes + brightness meter + overlays)
        annotated, _ = self.get_live_frame()
        if annotated is None:
            annotated = working_frame

        from utils.drawing import (
            draw_brightness_meter, draw_darkness_warning,
            draw_enhancement_indicator, draw_retry_countdown,
        )
        br = self.get_brightness_status()
        annotated = draw_brightness_meter(annotated, br.get("brightness", 128.0))
        if enhanced:
            annotated = draw_enhancement_indicator(annotated)
        if self._paused:
            annotated = draw_darkness_warning(annotated)
        retry = self.get_retry_info()
        if retry.get("active"):
            annotated = draw_retry_countdown(
                annotated,
                retry.get("check_number", 0),
                retry.get("seconds_remaining", 0),
                retry.get("attempt", 1),
                retry.get("max_attempts", 3),
            )
        return annotated

    # ──────────────────────────────────────────────
    # Pause / Resume
    # ──────────────────────────────────────────────

    def pause_session(self, reason=""):
        """Pause the session (track paused time)."""
        if not self._paused:
            self._paused = True
            self._pause_start_time = time.time()
            self._pause_reason = reason
            if self.session_id:
                self.attendance_store.update_session_status(self.session_id, "paused")
                self.attendance_store.add_session_note(
                    self.session_id, f"Paused: {reason}"
                )
            self._set_status(f"PAUSED — {reason}")
            print(f"[Monitor] Session paused: {reason}")

    def resume_session(self):
        """Resume the session, shift remaining check times forward."""
        if self._paused:
            pause_duration = time.time() - self._pause_start_time
            self._total_paused_duration += pause_duration
            self._paused = False
            self._pause_reason = ""

            # Shift the session start time forward to account for paused time
            self.session_start_time += pause_duration

            if self.session_id:
                self.attendance_store.update_session_status(self.session_id, "in_progress")
                self.attendance_store.add_paused_duration(self.session_id, pause_duration)
                self.attendance_store.add_session_note(
                    self.session_id, f"Resumed after {pause_duration:.0f}s pause"
                )
            self._set_status("Resumed — lights restored")
            print(f"[Monitor] Session resumed after {pause_duration:.1f}s pause.")

    @property
    def is_paused(self):
        return self._paused

    # ──────────────────────────────────────────────
    # Main scheduling loop
    # ──────────────────────────────────────────────

    def _run_scheduled_checks(self, session_id):
        """Background thread: wait for check times, execute checks with darkness handling."""
        try:
            for idx, check_time in enumerate(self.check_times):
                check_num = idx + 1
                if not self._wait_until(check_time):
                    break

                # Pre-check brightness test
                if not self._pre_check_brightness_test(session_id, check_num):
                    # Darkness detected — auto-pause and wait
                    if not self._handle_darkness_before_check(session_id, check_num):
                        # Could not recover — skip via retry system
                        continue

                self._set_status(f"Running check {check_num}/{self.total_checks}...")
                result = self._execute_check(session_id, check_num)

                if result == "failed_dark":
                    # All frames were dark — trigger retry
                    self._trigger_retry(session_id, check_num)
                else:
                    self.checks_completed = check_num
                    label = "completed" if result == "completed" else f"completed ({result})"
                    self._set_status(f"Check {check_num}/{self.total_checks} {label}")

            if not self._stop_event.is_set():
                self._set_status("Computing final results...")
                self._compute_final(session_id)
                self._set_status("Session completed ✓")
                if not self._use_external_camera:
                    self.camera.release()
                self.session_active = False
        except Exception as e:
            print(f"[Monitor] Error: {e}")
            self._set_status(f"Error: {e}")
            self._cleanup()

    def _pre_check_brightness_test(self, session_id, check_num):
        """Test brightness with a single frame before starting a check. Returns True if OK."""
        if not self._is_camera_ready():
            return True
        frame = self._read_frame()
        if frame is None:
            return True
        status = self.light_monitor.check_brightness(frame)
        self._update_brightness(status)
        return status["is_usable"] or status["is_recoverable"]

    def _handle_darkness_before_check(self, session_id, check_num):
        """Auto-pause and wait for brightness to recover. Returns True if recovered."""
        self.pause_session(f"Darkness detected before check {check_num}")

        poll_interval = Config.BRIGHTNESS_MONITOR_INTERVAL
        max_wait = Config.get_dark_retry_delay() * Config.MAX_RETRIES_PER_CHECK

        waited = 0.0
        while not self._stop_event.is_set() and waited < max_wait:
            frame = self._read_frame()
            if frame is not None:
                status = self.light_monitor.check_brightness(frame)
                self._update_brightness(status)

                # Update the live frame even while paused (so UI shows darkness)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_detections = []
                    self._latest_names = []
                    self._latest_matches = []
                    self._latest_spoofed = []
                    self._frame_enhanced = False

                if status["is_usable"] or status["is_recoverable"]:
                    self.resume_session()
                    return True

            self._stop_event.wait(timeout=poll_interval)
            waited += poll_interval

        # Timed out — still dark
        self.attendance_store.record_check(
            session_id, check_num, [],
            status="skipped_dark",
            note=f"Check {check_num} skipped — darkness persisted for {waited:.0f}s",
        )
        self.attendance_store.add_session_note(
            session_id, f"Check {check_num} skipped due to persistent darkness"
        )
        self.checks_completed = check_num
        self.resume_session()
        self._set_status(f"Check {check_num} skipped — darkness persisted")
        return False

    # ──────────────────────────────────────────────
    # Check execution with light handling
    # ──────────────────────────────────────────────

    def _execute_check(self, session_id, check_number):
        """
        Run one attendance check with brightness + twin handling.

        Returns:
            "completed"   — all frames usable
            "partial"     — some frames dark but some usable
            "failed_dark" — all frames too dark
        """
        self.current_check_running = True
        interval = Config.get_frame_interval()
        confirmed_detected = set()
        spoofed_detected = set()  # student IDs flagged as spoofed this check
        uncertain_detected = []  # list of twin_conflict dicts
        snapshot_frame = None
        usable_count = 0
        total_frames = Config.FRAMES_PER_CHECK

        for fi in range(total_frames):
            if self._stop_event.is_set():
                break
            if not self._is_camera_ready():
                break

            frame = self._read_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            # --- Brightness check ---
            brightness_status = self.light_monitor.check_brightness(frame)
            self._update_brightness(brightness_status)

            working_frame = frame
            enhanced = False

            if brightness_status["quality_label"] == "DARK":
                print(f"[Monitor] Check {check_number} frame {fi+1}: DARK — skipped")
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_detections = []
                    self._latest_names = []
                    self._latest_matches = []
                    self._latest_spoofed = []
                    self._frame_enhanced = False
                if fi < total_frames - 1:
                    time.sleep(interval)
                continue

            if brightness_status["quality_label"] == "LOW_LIGHT" and Config.LOW_LIGHT_ENHANCE:
                enhanced_frame = self.light_monitor.enhance_low_light(frame)
                if enhanced_frame is not None:
                    working_frame = enhanced_frame
                    enhanced = True
                else:
                    if fi < total_frames - 1:
                        time.sleep(interval)
                    continue

            usable_count += 1
            if snapshot_frame is None:
                snapshot_frame = working_frame.copy()

            detections = self.detector.detect_faces(working_frame)
            if detections:
                embeddings = [self.embedder.get_embedding(d["cropped_face"]) for d in detections]
                # Independent matching: every face finds its own best identity.
                # Multiple faces (real person + phone photo) can both resolve
                # to the same student; liveness below decides which is live.
                matches = self.matcher.find_all_matches_independent(embeddings)
                names = []
                spoofed_indices = []

                for idx, m in enumerate(matches):
                    if m:
                        # --- Liveness check on each matched face ---
                        face_crop = detections[idx]["cropped_face"] if idx < len(detections) else None
                        face_bbox = detections[idx].get("bbox") if idx < len(detections) else None
                        if face_crop is not None and Config.LIVENESS_ENABLED:
                            liveness = self.liveness_detector.quick_liveness_check(
                                face_crop, frame=working_frame, face_bbox=face_bbox)
                            if not liveness["is_live"]:
                                # SPOOFING DETECTED — flag this student as spoofed
                                spoof_type = liveness.get("spoofing_type", "unknown")
                                names.append(f"FAKE ({spoof_type})")
                                spoofed_indices.append(idx)
                                spoofed_detected.add(m["student_id"])
                                # Save spoofing evidence
                                spoof_dir = Config.SESSION_SNAPSHOTS_DIR / str(session_id) / "spoofing"
                                spoof_dir.mkdir(parents=True, exist_ok=True)
                                spoof_path = str(spoof_dir / f"chk{check_number}_f{fi}_d{idx}.jpg")
                                cv2.imwrite(spoof_path, face_crop)
                                print(f"[Monitor] \u26a0\ufe0f SPOOFING detected: {spoof_type} at check {check_number} frame {fi+1}")
                                continue  # Don't count as present

                        if m.get("uncertain"):
                            # Twin conflict — track for review
                            tc = m.get("twin_conflict", {})
                            names.append(
                                f"{tc.get('name_a','?')}/{tc.get('name_b','?')}?"
                            )
                            crop_path = None
                            if idx < len(detections):
                                snap_dir = Config.SESSION_SNAPSHOTS_DIR / str(session_id) / "uncertain"
                                snap_dir.mkdir(parents=True, exist_ok=True)
                                crop_path = str(snap_dir / f"chk{check_number}_f{fi}_d{idx}.jpg")
                                cv2.imwrite(crop_path, detections[idx]["cropped_face"])
                            uncertain_detected.append({
                                "check_number": check_number,
                                "frame_index": fi,
                                "student_a": tc.get("student_a"),
                                "name_a": tc.get("name_a"),
                                "score_a": tc.get("score_a", 0),
                                "student_b": tc.get("student_b"),
                                "name_b": tc.get("name_b"),
                                "score_b": tc.get("score_b", 0),
                                "difference": tc.get("difference", 0),
                                "crop_path": crop_path,
                                "assigned_to": m["student_id"],
                            })
                            confirmed_detected.add(m["student_id"])
                        else:
                            confirmed_detected.add(m["student_id"])
                            names.append(f"{m['name']} ({m['confidence']:.0%})")
                    else:
                        names.append(None)

                with self._frame_lock:
                    self._latest_frame = working_frame
                    self._latest_detections = detections
                    self._latest_names = names
                    self._latest_matches = matches
                    self._latest_spoofed = spoofed_indices
                    self._frame_enhanced = enhanced
            else:
                with self._frame_lock:
                    self._latest_frame = working_frame
                    self._latest_detections = []
                    self._latest_names = []
                    self._latest_matches = []
                    self._latest_spoofed = []
                    self._frame_enhanced = enhanced

            if fi < total_frames - 1:
                time.sleep(interval)

        self.current_check_running = False

        # With independent matching, the same student can appear as BOTH
        # live (real face → present) and spoofed (phone photo → flagged) in
        # the same frame. A student is "present" if they had at least one
        # LIVE detection. "Spoofed-only" means they appeared ONLY as a spoof
        # (no real face detected) — those don't count as present.
        spoofed_only = spoofed_detected - confirmed_detected
        self._spoofing_count += len(spoofed_only)

        # --- Determine check outcome ---
        if usable_count == 0:
            print(f"[Monitor] ⚠️ Check {check_number} FAILED — classroom too dark")
            return "failed_dark"

        # Add uncertain detections to review queue
        if uncertain_detected:
            self._review_queue.extend(uncertain_detected)
            print(f"[Monitor] Check {check_number}: {len(uncertain_detected)} uncertain (twin) detections")

        # Save snapshot
        snap_path = None
        if snapshot_frame is not None:
            snap_dir = Config.SESSION_SNAPSHOTS_DIR / str(session_id)
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = str(snap_dir / f"check_{check_number:02d}.jpg")
            with self._frame_lock:
                dets, nms = list(self._latest_detections), list(self._latest_names)
            if dets:
                annotated = self.detector.draw_detections(snapshot_frame, dets, nms)
                cv2.imwrite(snap_path, annotated)
            else:
                cv2.imwrite(snap_path, snapshot_frame)

        # Build note about uncertain detections
        check_note = None
        if uncertain_detected:
            check_note = f"{len(uncertain_detected)} twin-uncertain detection(s) need review"

        if usable_count < total_frames:
            note = f"Partial: {usable_count}/{total_frames} frames usable"
            if check_note:
                note += f"; {check_note}"
            self.attendance_store.record_check(
                session_id, check_number, list(confirmed_detected), snap_path,
                status="partial", note=note,
                usable_frames=usable_count, total_frames=total_frames,
                spoofed_student_ids=list(spoofed_only),
            )
            return "partial"
        else:
            self.attendance_store.record_check(
                session_id, check_number, list(confirmed_detected), snap_path,
                status="completed", note=check_note,
                usable_frames=usable_count, total_frames=total_frames,
                spoofed_student_ids=list(spoofed_only),
            )
            print(f"[Monitor] Check {check_number}: detected {list(confirmed_detected)}"
                  f"{f', spoofed-only: {list(spoofed_only)}' if spoofed_only else ''}")
            return "completed"

    # ──────────────────────────────────────────────
    # Retry logic
    # ──────────────────────────────────────────────

    def _trigger_retry(self, session_id, check_number):
        """Trigger retry logic for a failed check."""
        # Record the initial failure
        self.attendance_store.record_check(
            session_id, check_number, [],
            status="failed_dark",
            note=f"Check {check_number} failed — all frames too dark",
        )

        self._retry_check(session_id, check_number, attempt=1)

    def _retry_check(self, session_id, check_number, attempt=1):
        """Wait then retry a check. Recursive up to MAX_RETRIES_PER_CHECK."""
        retry_delay = Config.get_dark_retry_delay()
        max_attempts = Config.MAX_RETRIES_PER_CHECK

        self._retry_info.update({
            "active": True,
            "check_number": check_number,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "seconds_remaining": retry_delay,
        })

        self._set_status(
            f"Check {check_number} failed (dark) — retrying in {retry_delay}s "
            f"(attempt {attempt}/{max_attempts})"
        )
        self.attendance_store.add_session_note(
            session_id,
            f"Check {check_number} retry {attempt}/{max_attempts} scheduled",
        )

        # Wait with countdown
        start = time.time()
        while not self._stop_event.is_set():
            elapsed = time.time() - start
            remaining = retry_delay - elapsed
            if remaining <= 0:
                break
            self._retry_info["seconds_remaining"] = remaining
            self._update_live_preview()
            self._stop_event.wait(timeout=min(remaining, 0.5))

        self._retry_info["active"] = False

        if self._stop_event.is_set():
            return

        # Test brightness
        frame = self._read_frame()
        if frame is not None:
            status = self.light_monitor.check_brightness(frame)
            self._update_brightness(status)

            if status["is_usable"] or status["is_recoverable"]:
                # Lights restored — run the check
                print(f"[Monitor] Lights restored — running check {check_number}")
                self._set_status(f"Lights restored — running check {check_number}")
                result = self._execute_check(session_id, check_number)

                if result == "failed_dark" and attempt < max_attempts:
                    self._retry_check(session_id, check_number, attempt + 1)
                elif result != "failed_dark":
                    # Update the check status to retried
                    self.attendance_store.update_check_status(
                        session_id, check_number, "retried",
                        note=f"Succeeded on retry attempt {attempt}",
                    )
                    self.checks_completed = check_number
                    self._set_status(f"Check {check_number} completed (retry {attempt})")
                else:
                    # Max retries exhausted
                    self.attendance_store.update_check_status(
                        session_id, check_number, "skipped_dark",
                        note=f"Skipped after {max_attempts} retries — darkness persisted",
                    )
                    self.checks_completed = check_number
                    self._set_status(f"Check {check_number} skipped — darkness persisted")
                return

        # Still dark
        if attempt < max_attempts:
            print(f"[Monitor] Still dark, retry {attempt}/{max_attempts}")
            self._retry_check(session_id, check_number, attempt + 1)
        else:
            # Give up
            self.attendance_store.update_check_status(
                session_id, check_number, "skipped_dark",
                note=f"Skipped after {max_attempts} retries — darkness persisted",
            )
            self.attendance_store.add_session_note(
                session_id,
                f"Check {check_number} skipped — darkness persisted after {max_attempts} retries",
            )
            self.checks_completed = check_number
            self._set_status(f"Check {check_number} skipped — darkness persisted")
            print(f"[Monitor] Check {check_number} skipped — darkness persisted")

    # ──────────────────────────────────────────────
    # Live preview
    # ──────────────────────────────────────────────

    def _wait_until(self, check_time):
        """Wait until scheduled check time. Returns False if stopped."""
        target_sec = check_time if Config.DEMO_MODE else check_time * 60
        while not self._stop_event.is_set():
            elapsed = time.time() - self.session_start_time
            remaining = target_sec - elapsed
            if remaining <= 0:
                return True
            self._next_check_time = self.session_start_time + target_sec
            self._update_live_preview()
            self._stop_event.wait(timeout=min(remaining, 0.5))
        return False

    def _update_live_preview(self):
        """Capture frame, detect/match faces, update brightness, store for UI."""
        # In external-camera mode the WebRTC VideoProcessor drives the preview
        # via process_external_frame(); nothing to do here.
        if self._use_external_camera:
            return
        if not self._is_camera_ready():
            return
        frame = self._read_frame()
        if frame is None:
            return

        # Update brightness status
        brightness_status = self.light_monitor.check_brightness(frame)
        self._update_brightness(brightness_status)

        working_frame = frame
        enhanced = False

        # If paused due to darkness, just share the raw frame
        if self._paused:
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_detections = []
                self._latest_names = []
                self._latest_matches = []
                self._latest_spoofed = []
                self._frame_enhanced = False
            return

        # Enhance if low-light
        if brightness_status["quality_label"] == "LOW_LIGHT" and Config.LOW_LIGHT_ENHANCE:
            enhanced_frame = self.light_monitor.enhance_low_light(frame)
            if enhanced_frame is not None:
                working_frame = enhanced_frame
                enhanced = True

        detections = self.detector.detect_faces(working_frame)
        names = []
        matches_raw = []
        spoofed_indices = []
        if detections:
            embeddings = [self.embedder.get_embedding(d["cropped_face"]) for d in detections]
            # Independent matching + real-time liveness (same as WebRTC path).
            matches_raw = self.matcher.find_all_matches_independent(embeddings)
            for idx, m in enumerate(matches_raw):
                if m and m.get("uncertain"):
                    tc = m.get("twin_conflict", {})
                    names.append(f"{tc.get('name_a','?')}/{tc.get('name_b','?')}?")
                elif m:
                    if Config.LIVENESS_ENABLED and Config.LIVENESS_IN_PREVIEW:
                        face_crop = detections[idx]["cropped_face"] if idx < len(detections) else None
                        face_bbox = detections[idx].get("bbox") if idx < len(detections) else None
                        if face_crop is not None and face_crop.size > 0:
                            liveness = self.liveness_detector.quick_liveness_check(
                                face_crop, frame=working_frame, face_bbox=face_bbox)
                            if not liveness["is_live"]:
                                spoof_type = liveness.get("spoofing_type", "unknown")
                                m["spoofed"] = True
                                m["spoof_type"] = spoof_type
                                spoofed_indices.append(idx)
                                names.append(f"SPOOF — {m['name']} ({spoof_type})")
                                continue
                    names.append(f"{m['name']} ({m['confidence']:.0%})")
                else:
                    names.append(None)

        with self._frame_lock:
            self._latest_frame = working_frame
            self._latest_detections = detections
            self._latest_names = names
            self._latest_matches = matches_raw
            self._latest_spoofed = spoofed_indices
            self._frame_enhanced = enhanced

    def _update_brightness(self, status):
        with self._brightness_lock:
            self._brightness_status = status

    # ──────────────────────────────────────────────
    # Final computation
    # ──────────────────────────────────────────────

    def _compute_final(self, session_id):
        all_ids = list(self.face_db.get_all_students().keys())
        final = self.attendance_store.compute_final(session_id, all_ids)
        students = self.face_db.get_all_students()

        # Check which students had uncertain detections
        uncertain_sids = set()
        for item in self._review_queue:
            uncertain_sids.add(item.get("student_a"))
            uncertain_sids.add(item.get("student_b"))

        for sid, r in final.items():
            r["name"] = students[sid]["name"] if sid in students else "Unknown"
            if sid in uncertain_sids and r["status"] == "present":
                r["needs_review"] = True
                if not r.get("note"):
                    r["note"] = "Twin/lookalike — teacher review recommended"

        # Record spoofing stats in session
        if self._spoofing_count > 0:
            self.attendance_store.add_session_note(
                session_id,
                f"\ud83d\udee1\ufe0f {self._spoofing_count} spoofing attempt(s) blocked",
            )

        return final

    # ──────────────────────────────────────────────
    # Public API for UI
    # ──────────────────────────────────────────────

    def get_live_frame(self):
        """Return (annotated_frame, detections) for UI preview.
        Uses orange boxes for twin, red dashed for spoofed."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None, []
            frame = self._latest_frame.copy()
            dets = list(self._latest_detections)
            names = list(self._latest_names)
            matches = list(self._latest_matches) if hasattr(self, '_latest_matches') else []
            spoofed = list(self._latest_spoofed) if hasattr(self, '_latest_spoofed') else []

        if dets:
            from utils.drawing import (
                draw_face_box, draw_uncertain_box, draw_spoof_warning,
                draw_spoof_detected_box,
                COLOR_CONFIRMED, COLOR_UNKNOWN,
            )
            for i, det in enumerate(dets):
                bbox = det.get("bbox", det.get("box"))
                if bbox is None:
                    continue

                m = matches[i] if i < len(matches) else None

                # Live-preview spoof: face recognized (e.g. "MG") but liveness
                # flagged it as a phone/screen/print — yellow "SPOOF DETECTED"
                # box with the recognized name + spoof type.
                if m and m.get("spoofed"):
                    frame = draw_spoof_detected_box(
                        frame, tuple(bbox),
                        m.get("name", ""),
                        m.get("spoof_type", ""),
                    )
                    continue

                # Check-time spoof: red dashed "FAKE" box
                if i in spoofed:
                    n = names[i] if i < len(names) else "FAKE"
                    spoof_type = n.replace("FAKE (", "").rstrip(")") if n and "FAKE" in n else "FAKE"
                    frame = draw_spoof_warning(frame, tuple(bbox), spoof_type)
                    continue

                if m and m.get("uncertain"):
                    tc = m.get("twin_conflict", {})
                    frame = draw_uncertain_box(
                        frame, tuple(bbox),
                        tc.get("name_a", "?"), tc.get("name_b", "?"),
                        tc.get("score_a", 0), tc.get("score_b", 0),
                    )
                elif m:
                    name = names[i] if i < len(names) else m.get("name", "")
                    frame = draw_face_box(frame, tuple(bbox), name or m["name"],
                                         m["confidence"], COLOR_CONFIRMED)
                else:
                    frame = draw_face_box(frame, tuple(bbox), "Unknown", 0.0, COLOR_UNKNOWN)
        return frame, dets

    def get_brightness_status(self):
        """Return current brightness status dict."""
        with self._brightness_lock:
            return dict(self._brightness_status)

    def get_retry_info(self):
        """Return current retry info dict."""
        return dict(self._retry_info)

    def get_spoofing_count(self):
        """Return total spoofing attempts this session."""
        return self._spoofing_count

    @property
    def is_frame_enhanced(self):
        with self._frame_lock:
            return self._frame_enhanced

    def get_session_status(self):
        """Return dict with current session info for UI."""
        if not self.session_active or not self.session_id:
            return {
                "active": False, "session_id": None, "class_name": "",
                "status": "No active session", "checks_completed": 0,
                "total_checks": 0, "elapsed_seconds": 0,
                "next_check_in": 0, "check_running": False,
                "paused": False, "pause_reason": "",
                "brightness": 128.0, "brightness_label": "GOOD",
            }

        elapsed = time.time() - self.session_start_time
        nci = max(0, self._next_check_time - time.time())
        with self._status_lock:
            msg = self._status_message
        with self._brightness_lock:
            brightness = self._brightness_status.get("brightness", 128.0)
            brightness_label = self._brightness_status.get("quality_label", "GOOD")

        return {
            "active": True,
            "session_id": self.session_id,
            "class_name": self.class_name,
            "status": msg,
            "checks_completed": self.checks_completed,
            "total_checks": self.total_checks,
            "elapsed_seconds": round(elapsed, 1),
            "next_check_in": round(nci, 1),
            "check_running": self.current_check_running,
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "brightness": brightness,
            "brightness_label": brightness_label,
        }

    def _set_status(self, message):
        with self._status_lock:
            self._status_message = message
        print(f"[Monitor] {message}")

    def is_session_active(self):
        return self.session_active

    def get_check_schedule_info(self):
        unit = "sec" if Config.DEMO_MODE else "min"
        schedule = []
        session = self.attendance_store.get_session(self.session_id) if self.session_id else None
        checks_data = session.get("checks", {}) if session else {}

        for i, t in enumerate(self.check_times):
            cn = i + 1
            # Check if this check has a recorded status
            check_rec = checks_data.get(cn, checks_data.get(str(cn)))
            if check_rec:
                s = check_rec.get("status", "completed")
            elif cn <= self.checks_completed:
                s = "completed"
            elif cn == self.checks_completed + 1 and self.session_active:
                s = "next"
            else:
                s = "pending"
            schedule.append({
                "check_number": cn, "time_value": t, "unit": unit, "status": s,
            })
        return schedule

    # ──────────────────────────────────────────────
    # Twin Review Queue
    # ──────────────────────────────────────────────

    def get_review_queue(self, session_id=None):
        """Return all uncertain (twin) detections for teacher review."""
        if session_id:
            return [r for r in self._review_queue
                    if r.get("session_id", self.session_id) == session_id]
        return list(self._review_queue)

    def get_review_count(self):
        """Return number of items needing review."""
        return len(self._review_queue)

    def resolve_review(self, review_index, confirmed_student_id):
        """
        Teacher confirms which twin a detection belongs to.
        Updates the review queue item.
        """
        if review_index < 0 or review_index >= len(self._review_queue):
            return False
        item = self._review_queue[review_index]
        item["resolved"] = True
        item["confirmed_student_id"] = confirmed_student_id
        print(
            f"[Monitor] Review resolved: detection at check {item['check_number']} "
            f"confirmed as {confirmed_student_id}"
        )
        return True
