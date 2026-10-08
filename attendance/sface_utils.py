from threading import local
from pathlib import Path
import cv2
import numpy as np
from django.conf import settings

 
_state = local()
 
 
def _engine():
    paths = (Path(settings.FACE_YUNET_MODEL),
             Path(settings.FACE_SFACE_MODEL))
    for path in paths:
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError(f"模型文件不存在或不完整: {path}")
    key = tuple((str(p), p.stat().st_mtime_ns) for p in paths)
    if getattr(_state, "key", None) != key:
        _state.detector = cv2.FaceDetectorYN_create(
            str(paths[0]), "", (320, 320), 0.9, 0.3, 5000)
        _state.recognizer = cv2.FaceRecognizerSF_create(
            str(paths[1]), "")
        _state.key = key
    return _state.detector, _state.recognizer
 
 
def read_photo(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("照片无法读取，请使用有效 JPG 或 PNG")
    return image
 
 
def extract_feature(image):
    if image is None or image.size == 0:
        raise ValueError("图片为空")
    detector, recognizer = _engine()
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) != 1:
        raise ValueError("照片必须检测到且只检测到一张人脸")
    aligned = recognizer.alignCrop(image, faces[0])
    feature = recognizer.feature(aligned).reshape(-1).copy()
    feature = feature.astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(feature).all() or norm <= 0:
        raise ValueError("人脸特征无效")
    return feature / norm
