from services.attendance_service import AttendanceMonitor
from storage.face_database import FaceDatabase
from storage.attendance_store import AttendanceStore
from core.twin_handler import TwinHandler

monitor = AttendanceMonitor(
    face_detector=None,
    face_embedder=None,
    face_database=FaceDatabase(),
    attendance_store=AttendanceStore(),
    twin_handler=TwinHandler(),
)
sid = monitor.start_session('Test Class', camera_index=0)
print('session_id', sid)
if sid is not None:
    final = monitor.stop_session()
    print('final', final is not None)
