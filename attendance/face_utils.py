from functools import lru_cache
import os
import cv2
 
 
@lru_cache(maxsize=1)
def get_detector():
    path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise ValueError("Face detector XML is unavailable")
    return detector
 
 
def prepare_face(image):
    if image is None:
        raise ValueError("Image could not be decoded")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = get_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) != 1:
        raise ValueError("Use an image with exactly one face")
    x, y, width, height = faces[0]
    face = gray[y:y + height, x:x + width]
    face = cv2.resize(face, (200, 200))
    return cv2.equalizeHist(face)
 
 
def read_face(path):
    return prepare_face(cv2.imread(str(path)))
