import os
import tempfile
from pathlib import Path
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from attendance.models import Student
from attendance.sface_utils import read_photo, extract_feature
 
 
class Command(BaseCommand):
    help = "用 YuNet 和 SFace 建立学生人脸特征库"
 
    def handle(self, *args, **options):
        features, labels, missing = [], [], []
        students = Student.objects.filter(is_active=True)
        students = students.prefetch_related("face_photos")
        for student in students:
            photos = [student.photo]
            photos += [item.photo for item in student.face_photos.all()]
            seen, count = set(), 0
            for photo in photos:
                if not photo:
                    continue
                path = Path(photo.path)
                if path in seen:
                    continue
                seen.add(path)
                try:
                    feature = extract_feature(read_photo(path))
                except FileNotFoundError as exc:
                    # 照片缺失可跳过，模型缺失必须停止。
                    if path.is_file():
                        raise CommandError(str(exc)) from exc
                    self.stderr.write(f"SKIP {path}: 照片不存在")
                    continue
                except (ValueError, OSError) as exc:
                    self.stderr.write(f"SKIP {path}: {exc}")
                    continue
                features.append(feature)
                labels.append(student.pk)
                count += 1
            self.stdout.write(
                f"{student.student_number}: {count} valid photos")
            if count == 0:
                missing.append(student.student_number)
            if missing:
                raise CommandError("以下学生没有有效照片: " + ", ".join(missing))
            if not features:
                raise CommandError("没有有效的学生照片，未更新特征库")
            target = Path(settings.FACE_GALLERY_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, suffix=".npz", delete=False
                ) as file:
                    temporary = file.name
                    np.savez_compressed(
                        file, features=np.stack(features),
                        labels=np.asarray(labels, dtype=np.int64))
                os.replace(temporary, target)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            self.stdout.write(self.style.SUCCESS(
                f"Saved {target}; {len(set(labels))} students, "
                f"{len(features)} features"))


