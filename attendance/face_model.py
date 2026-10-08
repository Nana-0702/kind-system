from pathlib import Path
from threading import Lock
import cv2
from django.conf import settings
 
 
_model = None
_stamp = None
_lock = Lock()
 
 
def predict_face(face):
    global _model, _stamp
    path = Path(settings.BASE_DIR) / "face_model.yml"
    if not path.exists():
        raise FileNotFoundError("Run train_faces first")
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _lock:
        if _model is None or stamp != _stamp:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(str(path))
            _model = model
            _stamp = stamp
        return _model.predict(face)
