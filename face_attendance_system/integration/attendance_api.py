"""
Attendance Session REST API.
FastAPI APIRouter for managing attendance sessions.
Teammates import this router into their FastAPI backend:

    from integration.attendance_api import create_attendance_router
    app.include_router(create_attendance_router(monitor, attendance_store, face_db))
"""

from fastapi import APIRouter, HTTPException

from core.config import Config
from integration.schemas import (
    StartSessionRequest,
    StartSessionResponse,
    SessionStatusResponse,
    StopSessionResponse,
    SessionResultsResponse,
    AttendanceResult,
    AttendanceStatus,
    CheckInfo,
    CheckStatus,
)


def create_attendance_router(monitor, attendance_store, face_db) -> APIRouter:
    """
    Factory that creates an attendance APIRouter with injected dependencies.

    Args:
        monitor: AttendanceMonitor instance.
        attendance_store: AttendanceStore instance.
        face_db: FaceDatabase instance.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

    # ── POST /session/start ──────────────────────

    @router.post("/session/start", response_model=StartSessionResponse)
    async def start_session(req: StartSessionRequest):
        """Start a new attendance session."""
        if monitor.is_session_active():
            return StartSessionResponse(
                success=False,
                session_id=None,
                message="A session is already active. Stop it first.",
                check_times=[],
                mode="demo" if Config.DEMO_MODE else "normal",
            )

        session_id = monitor.start_session(
            class_name=req.class_name,
            camera_index=req.camera_source,
        )

        if session_id is None:
            return StartSessionResponse(
                success=False,
                session_id=None,
                message=(
                    "Failed to start session. "
                    "Ensure students are registered and the camera is available."
                ),
                check_times=[],
                mode="demo" if Config.DEMO_MODE else "normal",
            )

        return StartSessionResponse(
            success=True,
            session_id=session_id,
            message=f"Session started for '{req.class_name}'.",
            check_times=Config.get_check_times(),
            mode="demo" if Config.DEMO_MODE else "normal",
        )

    # ── GET /session/status ──────────────────────

    @router.get("/session/status", response_model=SessionStatusResponse)
    async def get_session_status():
        """Get the current session status."""
        status = monitor.get_session_status()

        # Build check info list
        check_infos = []
        if status["active"]:
            schedule = monitor.get_check_schedule_info()
            session = attendance_store.get_session(status["session_id"])
            checks_data = session.get("checks", {}) if session else {}

            for chk in schedule:
                cn = chk["check_number"]
                cn_str = str(cn)
                detected = None
                if cn_str in checks_data:
                    detected = checks_data[cn_str].get("count", 0)
                elif cn in checks_data:
                    detected = checks_data[cn].get("count", 0)

                check_infos.append(CheckInfo(
                    check_number=chk["check_number"],
                    time_value=chk["time_value"],
                    unit=chk["unit"],
                    status=CheckStatus(chk["status"]),
                    detected_count=detected,
                ))

        return SessionStatusResponse(
            active=status["active"],
            session_id=status["session_id"],
            class_name=status["class_name"],
            status=status["status"],
            checks_completed=status["checks_completed"],
            total_checks=status["total_checks"],
            elapsed_seconds=status["elapsed_seconds"],
            next_check_in=status["next_check_in"],
            check_running=status["check_running"],
            checks=check_infos,
        )

    # ── POST /session/stop ───────────────────────

    @router.post("/session/stop", response_model=StopSessionResponse)
    async def stop_session():
        """Stop the active session early and compute final results."""
        if not monitor.is_session_active():
            return StopSessionResponse(
                success=False,
                message="No active session to stop.",
                session_id=None,
                results=[],
            )

        session_id = monitor.session_id
        final = monitor.stop_session()

        results = _build_results_list(session_id, final)

        return StopSessionResponse(
            success=True,
            message="Session stopped. Final results computed from completed checks.",
            session_id=session_id,
            results=results,
        )

    # ── GET /results/{session_id} ────────────────

    @router.get("/results/{session_id}", response_model=SessionResultsResponse)
    async def get_results(session_id: str):
        """Get attendance results for a specific session."""
        session = attendance_store.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        final = session.get("final_results", {})
        results = _build_results_list(session_id, final)

        present = sum(1 for r in results if r.status == AttendanceStatus.PRESENT)
        late = sum(1 for r in results if r.status == AttendanceStatus.LATE)
        absent = sum(1 for r in results if r.status == AttendanceStatus.ABSENT)

        return SessionResultsResponse(
            session_id=session_id,
            class_name=session.get("class_name", ""),
            date=session.get("date", ""),
            session_status=session.get("status", "unknown"),
            total_students=len(results),
            present_count=present,
            late_count=late,
            absent_count=absent,
            results=results,
        )

    # ── GET /sessions ────────────────────────────

    @router.get("/sessions")
    async def list_sessions():
        """List all attendance sessions (summary only)."""
        sessions = attendance_store.get_all_sessions()
        summaries = []
        for sid, data in sessions.items():
            final = data.get("final_results", {})
            present = sum(1 for r in final.values() if r.get("status") == "present")
            late = sum(1 for r in final.values() if r.get("status") == "late")
            absent = sum(1 for r in final.values() if r.get("status") == "absent")

            summaries.append({
                "session_id": sid,
                "class_name": data.get("class_name", ""),
                "date": data.get("date", ""),
                "status": data.get("status", ""),
                "present": present,
                "late": late,
                "absent": absent,
                "total": len(final),
            })

        summaries.sort(key=lambda x: x.get("date", ""), reverse=True)
        return summaries

    # ── Helper ───────────────────────────────────

    def _build_results_list(session_id, final_results):
        """Convert raw final_results dict into a list of AttendanceResult."""
        if not final_results:
            return []

        session = attendance_store.get_session(session_id)
        checks = session.get("checks", {}) if session else {}
        students = face_db.get_all_students()
        results = []

        for sid, result in final_results.items():
            name = result.get("name", students.get(sid, {}).get("name", sid))

            def _was_detected(check_num):
                cd = checks.get(check_num) or checks.get(str(check_num))
                if cd:
                    return sid in cd.get("detected", [])
                return False

            results.append(AttendanceResult(
                student_id=sid,
                name=name,
                check1=_was_detected(1),
                check2=_was_detected(2),
                check3=_was_detected(3),
                checks_present=result.get("checks_present", 0),
                total_checks=len(checks),
                status=AttendanceStatus(result.get("status", "absent")),
            ))

        return results

    return router
