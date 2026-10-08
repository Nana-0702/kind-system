from collections import Counter
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from attendance.models import Student
from attendance.face_utils import read_face
class Command(BaseCommand):
    help = "Train the face model from active students"
    def handle(self, *args, **options):
        faces, labels = [], []
        counts = Counter()
        students = Student.objects.filter(is_active=True)
        for student in students:
            paths = []
            if student.photo:
                paths.append(student.photo.path)
            paths.extend(
                item.photo.path
                for item in student.face_photos.all()
                if item.photo
            )
            for path in paths:
                try:
                    face = read_face(path)
                except ValueError as error:
                    self.stderr.write(f"SKIP {path}: {error}")
                    continue
                faces.append(face)
                labels.append(student.pk)
                counts[student.pk] += 1
        if not faces:
            raise CommandError("No valid training faces found")
        if not hasattr(cv2, "face"):
            raise CommandError("opencv-contrib-python is required")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.asarray(labels, dtype=np.int32))
        target = Path(settings.BASE_DIR) / "face_model.yml"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".yml", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            recognizer.write(str(temporary_path))
            os.replace(temporary_path, target)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        for student in students:
            number = counts[student.pk]
            self.stdout.write(
                f"{student.student_number}: {number} valid photos"
            )
            if number < 10:
                self.stderr.write("Few photos: review this student")
        self.stdout.write(self.style.SUCCESS(
            f"Saved {target}; {len(counts)} students, {len(faces)} faces"
        ))
        if not hasattr(cv2, "face"):
            raise CommandError("opencv-contrib-python is required")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.asarray(labels, dtype=np.int32))
        target = Path(settings.BASE_DIR) / "face_model.yml"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".yml", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            recognizer.write(str(temporary_path))
            os.replace(temporary_path, target)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        for student in students:
            number = counts[student.pk]
            self.stdout.write(
                f"{student.student_number}: {number} valid photos"
            )
            if number < 10:
                self.stderr.write("Few photos: review this student")
        self.stdout.write(self.style.SUCCESS(
            f"Saved {target}; {len(counts)} students, {len(faces)} faces"
        ))


        