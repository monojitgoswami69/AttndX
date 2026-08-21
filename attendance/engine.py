"""
Attendance engine — separate from recognition.

Responsibilities:
  - Manage session lifecycle (start, scheduled checks, stop)
  - Run scheduled checks at configured times
  - Per check: capture frames, run recognition, collect confirmed identities
  - Liveness: run at attendance-decision time (NOT on every preview frame)
  - Temporal confirmation: require min_confirmation_frames of confirmed identity
  - Idempotency: DB UNIQUE(session_id, identity_id) prevents duplicate attendance
  - Final computation: present in ≥ min_checks_for_present valid checks

Decision flow per check:
  frame → recognition pipeline (detect, align, embed, match, track)
        → for each recognized face, run liveness
        → collect identity_id if liveness=LIVE and track is confirmed
        → record check with detected_ids set (set = no duplicates)
"""
import logging
import threading
import time
import datetime
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from config import Config
from recognition.engine import RecognitionPipeline, FrameResult
from recognition.liveness.minifasnet import LivenessState
from recognition.matching.engine import RecognitionState
from storage.attendance_db import AttendanceDB
from storage.db import BiometricDB
from vision.camera.source import CameraSource, ExternalFrameBuffer

logger = logging.getLogger(__name__)


@dataclass
class SessionStatus:
    active: bool = False
    session_id: str | None = None
    class_name: str = ""
    status_message: str = ""
    checks_completed: int = 0
    total_checks: int = 0
    elapsed_seconds: float = 0.0
    next_check_in: float = 0.0
    check_running: bool = False
    paused: bool = False
    spoofing_count: int = 0


class AttendanceEngine:
    """Manages a live attendance session with scheduled checks."""

    def __init__(self, pipeline: RecognitionPipeline,
                 camera: CameraSource, external_buffer: ExternalFrameBuffer | None,
                 biometric_db: BiometricDB, attendance_db: AttendanceDB):
        self.pipeline = pipeline
        self.camera = camera
        self.external_buffer = external_buffer
        self.biometric_db = biometric_db
        self.attendance_db = attendance_db

        self.demo_mode = Config.get("attendance", "demo_mode")
        self.check_times = (Config.get("attendance", "check_times_demo")
                            if self.demo_mode
                            else Config.get("attendance", "check_times_normal"))
        self.frame_interval = (Config.get("attendance", "frame_interval_demo")
                               if self.demo_mode
                               else Config.get("attendance", "frame_interval_normal"))
        self.frames_per_check = Config.get("attendance", "frames_per_check")
        self.min_checks_present = Config.get("attendance", "min_checks_for_present")
        self.min_confirmation_frames = Config.get("attendance", "min_confirmation_frames")

        self._use_external = external_buffer is not None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.session_id: str | None = None
        self.session_active = False
        self.session_start_time = 0.0
        self.checks_completed = 0
        self.total_checks = len(self.check_times)
        self.class_name = ""
        self.current_check_running = False
        self.spoofing_count = 0

        self._status_message = "Idle"
        self._next_check_time = 0.0
        self._latest_frame_result: FrameResult | None = None

    def start_session(self, class_name: str) -> str | None:
        """Start a new attendance session."""
        if self.session_active:
            return None

        # Check that identities are enrolled
        templates = self.biometric_db.get_all_templates()
        if not templates:
            logger.warning("No enrolled identities — cannot start session")
            return None

        # Camera strategy:
        #   WebRTC mode (_use_external=True):  browser owns the camera.
        #     Do NOT open cv2.VideoCapture — it would grab the device
        #     on the server side, preventing the browser from accessing it
        #     on some OSes, and it stays open as a privacy leak.
        #   cv2 mode (_use_external=False): open the cv2 camera as the
        #     primary source. Release it when the session ends.
        if not self._use_external:
            if not self.camera.is_opened():
                try:
                    self.camera.open()
                    if self.camera.is_opened():
                        logger.info("cv2 camera opened as primary source")
                except Exception as e:
                    logger.error(f"cv2 camera open failed: {e}")
                    return None
        else:
            # Ensure cv2 camera is NOT held open from a previous session
            if self.camera.is_opened():
                self.camera.release()
                logger.info("Released stale cv2 camera before WebRTC session")

        mode = "demo" if self.demo_mode else "normal"
        self.session_id = self.attendance_db.create_session(class_name, mode)
        self.class_name = class_name
        self.session_active = True
        self.session_start_time = time.time()
        self.checks_completed = 0
        self.spoofing_count = 0
        self._stop_event.clear()
        self._set_status("Session started")

        self._thread = threading.Thread(
            target=self._run_scheduled_checks,
            args=(self.session_id,),
            daemon=True,
        )
        self._thread.start()

        unit = "sec" if self.demo_mode else "min"
        logger.info(f"Session {self.session_id} started. "
                    f"Checks at {self.check_times} {unit}.")
        return self.session_id

    def stop_session(self):
        """Stop session early, compute final from completed checks."""
        if not self.session_active or not self.session_id:
            return None

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self._compute_final(self.session_id)
        # ALWAYS release cv2 camera — never leave it dangling
        self._release_camera()
        self.session_active = False
        self._set_status("Session stopped")
        return self.session_id

    def _release_camera(self):
        """Unconditionally release the cv2 camera if it was opened."""
        if self.camera.is_opened():
            self.camera.release()
            logger.info("cv2 camera released (session ended)")

    def _run_scheduled_checks(self, session_id: str):
        """Background thread: wait for check times, execute checks."""
        try:
            for idx, check_time in enumerate(self.check_times):
                check_num = idx + 1
                if self._stop_event.is_set():
                    break
                if not self._wait_until(check_time):
                    break

                self._set_status(f"Running check {check_num}/{self.total_checks}")
                self._execute_check(session_id, check_num)

            if not self._stop_event.is_set():
                self._set_status("Computing final results")
                self._compute_final(session_id)
                self._set_status("Session completed")
                self._release_camera()
                self.session_active = False
        except Exception as e:
            logger.error(f"Session thread error: {e}", exc_info=True)
            self._set_status(f"Error: {e}")
            self._release_camera()
            self.session_active = False

    def _wait_until(self, check_time: int) -> bool:
        """Wait until scheduled check time. Returns False if stopped."""
        target = check_time if self.demo_mode else check_time * 60
        while not self._stop_event.is_set():
            elapsed = time.time() - self.session_start_time
            remaining = target - elapsed
            if remaining <= 0:
                return True
            self._next_check_time = self.session_start_time + target
            self._stop_event.wait(timeout=min(remaining, 0.5))
        return False

    def _execute_check(self, session_id: str, check_number: int):
        """Run one attendance check.

        Captures frames_per_check frames, runs recognition on each,
        collects confirmed LIVE identities, runs liveness at decision time.
        """
        self.current_check_running = True
        confirmed_ids: set[str] = set()
        spoofed_ids: set[str] = set()
        identity_frame_counts: dict[str, int] = defaultdict(int)
        usable_frames = 0

        for fi in range(self.frames_per_check):
            if self._stop_event.is_set():
                break

            frame = self._read_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            usable_frames += 1

            # Run recognition pipeline (no liveness on every frame)
            try:
                frame_result = self.pipeline.process_frame(frame, run_liveness=False)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue

            with self._lock:
                self._latest_frame_result = frame_result

            # Collect confirmed identities
            for face in frame_result.faces:
                if face.confirmed_identity_id is not None:
                    identity_frame_counts[face.confirmed_identity_id] += 1
                    if identity_frame_counts[face.confirmed_identity_id] >= self.min_confirmation_frames:
                        # Run liveness at decision time
                        if self.pipeline.liveness is not None and self.pipeline.liveness.is_available():
                            liveness_result = self.pipeline.liveness.check(frame, face.bbox)
                            if liveness_result.state == LivenessState.LIVE:
                                confirmed_ids.add(face.confirmed_identity_id)
                            elif liveness_result.state == LivenessState.SPOOF:
                                spoofed_ids.add(face.confirmed_identity_id)
                                self.spoofing_count += 1
                                logger.warning(
                                    f"SPOOF detected: {liveness_result.spoofing_type} "
                                    f"for identity {face.confirmed_identity_id}"
                                )
                            # UNCERTAIN/ERROR → don't add to either set
                        else:
                            # No liveness → accept confirmed identity
                            confirmed_ids.add(face.confirmed_identity_id)

            if fi < self.frames_per_check - 1:
                time.sleep(self.frame_interval)

        self.current_check_running = False

        # Save check snapshot to snapshots_dir if available
        if frame is not None:
            try:
                snap_dir = Config.snapshots_dir() / session_id
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"check_{check_number:02d}.jpg"
                cv2.imwrite(str(snap_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:
                logger.warning(f"Failed to save snapshot for check {check_number}: {e}")

        # Record check (idempotent: overwrites if check_number exists)
        self.attendance_db.record_check(
            session_id, check_number,
            detected_ids=list(confirmed_ids),
            spoofed_ids=list(spoofed_ids - confirmed_ids),
            status="completed",
            usable_frames=usable_frames,
            total_frames=self.frames_per_check,
        )
        self.checks_completed = check_number
        self._set_status(f"Check {check_number}/{self.total_checks} completed")

    def _read_frame(self) -> np.ndarray | None:
        """Read a frame from external buffer (WebRTC) or cv2 camera (fallback).
        Tries external buffer first; if empty, falls back to cv2 camera."""
        if self._use_external and self.external_buffer is not None:
            frame = self.external_buffer.peek()
            if frame is not None:
                return frame
            # External buffer empty — try cv2 fallback
        if self.camera.is_opened():
            return self.camera.read_frame()
        return None

    def _compute_final(self, session_id: str):
        """Compute final attendance from recorded checks."""
        checks = self.attendance_db.get_checks(session_id)
        identities = self.biometric_db.get_all_identities()
        identity_map = {i["identity_id"]: i["name"] for i in identities}

        # Valid checks: completed or retried
        valid_checks = {cn: cd for cn, cd in checks.items()
                       if cd["status"] in ("completed", "partial", "retried")}
        num_valid = len(valid_checks)
        min_needed = min(self.min_checks_present, num_valid)

        for identity_id, name in identity_map.items():
            present_count = 0
            spoofed_count = 0
            for cn, cd in valid_checks.items():
                if identity_id in cd.get("spoofed", []):
                    spoofed_count += 1
                elif identity_id in cd.get("detected", []):
                    present_count += 1

            if num_valid == 0:
                status = "insufficient_data"
                note = "No valid checks — manual review needed"
            elif present_count >= min_needed:
                status = "present"
                note = ""
            else:
                status = "absent"
                note = f"Present in {present_count}/{num_valid} checks"

            if spoofed_count > 0:
                note = f"Spoofed in {spoofed_count} check(s)" + (f"; {note}" if note else "")

            # DB-level idempotency: UNIQUE(session_id, identity_id)
            self.attendance_db.upsert_attendance(
                session_id, identity_id, name, status,
                checks_present=present_count,
                checks_spoofed=spoofed_count,
                needs_review=(num_valid < self.min_checks_present),
                note=note,
            )

        if num_valid == 0:
            self.attendance_db.update_session_status(session_id, "cancelled_no_data")
        elif self.attendance_db.get_session(session_id):
            self.attendance_db.update_session_status(
                session_id, "completed",
                end_time=datetime.datetime.now().isoformat()
            )

        logger.info(f"Final attendance computed for {session_id}: "
                    f"{num_valid} valid checks")

    def get_status(self) -> SessionStatus:
        elapsed = (time.time() - self.session_start_time
                   if self.session_active else 0.0)
        nci = max(0, self._next_check_time - time.time()) if self.session_active else 0
        return SessionStatus(
            active=self.session_active,
            session_id=self.session_id,
            class_name=self.class_name,
            status_message=self._status_message,
            checks_completed=self.checks_completed,
            total_checks=self.total_checks,
            elapsed_seconds=round(elapsed, 1),
            next_check_in=round(nci, 1),
            check_running=self.current_check_running,
            spoofing_count=self.spoofing_count,
        )

    def get_latest_frame_result(self) -> FrameResult | None:
        with self._lock:
            return self._latest_frame_result

    def _set_status(self, msg: str):
        self._status_message = msg
        logger.info(f"[Attendance] {msg}")
