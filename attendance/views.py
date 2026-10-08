import base64
import json
from datetime import timedelta
from django.db import transaction
from .models import StudentFacePhoto

import cv2
import numpy as np
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from .forms import StudentForm
from .models import Attendance, Student
from django.conf import settings
from .face_utils import prepare_face
from .face_model import predict_face
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from .models import Student
from django.contrib.admin.views.decorators import staff_member_required

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(xml_path)


def top(request):
    return render(request, "attendance/top.html")


def register_student(request):
    if request.method == "POST":
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            with transaction.atomic():
                student = form.save()
                for photo in form.cleaned_data["face_photos"]:
                    StudentFacePhoto.objects.create(
                        student=student, photo=photo
                    )
            messages.success(
                request,
                f"{student.name} さんを登録しました。"
                "顔認識にはモデルの再学習が必要です。",
            )
            return redirect("attendance:top")
    else:
        form = StudentForm()
    return render(
        request, "attendance/register.html", {"form": form}
    )


def camera(request):
    return render(request, "attendance/camera.html")


def _read_student_face(student):
    """登録写真から最大の顔1つを取り出して、200x200のグレースケールにする。"""
    image = cv2.imread(student.photo.path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))


def _build_recognizer():
    """有効な園児の登録写真からLBPH認識器を作る。"""
    training_faces = []
    labels = []
    label_to_student_id = {}

    students = Student.objects.filter(is_active=True)

    label = 0
    for student in students:
        face = _read_student_face(student)
        if face is None:
            continue

        training_faces.append(face)
        labels.append(label)
        label_to_student_id[label] = student.id
        label += 1

    if not training_faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(labels, dtype=np.int32))
    return recognizer, label_to_student_id


def _decode_camera_image(data_url):
    """data:image/jpeg;base64,... をOpenCV画像へ変換する。"""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    binary = base64.b64decode(data_url)
    array = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _find_face_from_camera(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))


def _next_attendance_type(student):
    """本日最初は登園。すでに登園があり降園がなければ降園。
    すでに降園済みの場合は再登録させない。
    """
    today = timezone.localdate()
    records = Attendance.objects.filter(
        student=student,
        timestamp__date=today,
    )

    has_in = records.filter(attendance_type="in").exists()
    has_out = records.filter(attendance_type="out").exists()

    if not has_in:
        return "in"
    if not has_out:
        return "out"
    return None


@require_POST
def recognize_face(request):
    try:
        body = json.loads(request.body)
        data_url = body.get("image")

        if not data_url:
            return JsonResponse({"ok": False, "message": "画像がありません。"}, status=400)

        image = _decode_camera_image(data_url)
        if image is None:
            return JsonResponse({"ok": False, "message": "画像を読み込めません。"}, status=400)

        try:
            camera_face = prepare_face(image)
        except ValueError:
            return JsonResponse({
                "ok": False,
                "message": "一人の顔がはっきり写るようにしてください。",
            })

        try:
            label, distance = predict_face(camera_face)
        except FileNotFoundError:
            return JsonResponse({
                "ok": False,
                "message": "顔認識モデルを先に学習してください。",
            })

        # 距離（distance）がしきい値を超えている場合は認証失敗
        threshold = settings.FACE_DISTANCE_THRESHOLD
        if distance > threshold:
            return JsonResponse({
                "ok": False,
                "message": "登録済みの顔と一致しませんでした。",
                "distance": round(float(distance), 2),
            })

        # 推論された label (通常は student.pk) から有効な生徒を取得
        student = Student.objects.filter(pk=int(label), is_active=True).first()
        if student is None:
            return JsonResponse({
                "ok": False,
                "message": "在園中の園児ではありません。",
            })

        # 数秒以内の連打による二重登録を防ぐ
        recent = Attendance.objects.filter(
            student=student,
            timestamp__gte=timezone.now() - timedelta(seconds=10),
        ).first()
        if recent:
            return JsonResponse({
                "ok": False,
                "message": "連続認証です。少し待ってからもう一度お試しください。"
            })

        attendance_type = _next_attendance_type(student)
        if attendance_type is None:
            return JsonResponse({
                "ok": False,
                "message": f"{student.name} さんは本日の登園・降園がすでに完了しています。"
            })

        record = Attendance.objects.create(
            student=student,
            attendance_type=attendance_type,
        )

        return JsonResponse({
            "ok": True,
            "student": student.name,
            "student_number": student.student_number,
            "type": record.get_attendance_type_display(),
            "time": timezone.localtime(record.timestamp).strftime("%H:%M:%S"),
            "distance": round(float(distance), 2),
            "redirect_url": f"/success/{record.id}/",
        })

    except Exception as e:
        return JsonResponse({
            "ok": False,
            "message": f"エラーが発生しました: {str(e)}"
        }, status=500)


def success(request, record_id):
    record = get_object_or_404(Attendance, id=record_id)
    return render(request, "attendance/success.html", {"record": record})


def dashboard(request):
    today = timezone.localdate()
    students = Student.objects.filter(is_active=True).order_by("student_number")
    rows = []

    for student in students:
        records = Attendance.objects.filter(
            student=student,
            timestamp__date=today,
        )
        in_record = records.filter(attendance_type="in").order_by("timestamp").first()
        out_record = records.filter(attendance_type="out").order_by("timestamp").first()

        rows.append({
            "student": student,
            "in_record": in_record,
            "out_record": out_record,
        })

    return render(
        request,
        "attendance/dashboard.html",
        {"rows": rows, "today": today},
    )
@require_POST
def student_delete_view(request, pk):
    student = get_object_or_404(Student, pk=pk)
    student.delete()
    return redirect("attendance:dashboard")

# Create your views here.
