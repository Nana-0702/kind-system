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
from .models import Attendance, Student,DailyMood
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from .models import Student
from django.contrib.admin.views.decorators import staff_member_required

import os

from .sface_utils import extract_feature
from .sface_model import predict_feature

from django.urls import reverse

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
    mode = request.GET.get("mode", "test")
    titles = {
        "in": "登園",
        "out": "降園",
        "test": "顔認証テスト",
    }
    if mode not in titles:
        return JsonResponse(
            {"ok": False, "message": "操作の種類が正しくありません。"},
            status=400,
        )
    return render(
        request,
        "attendance/camera.html",
        {"mode": mode, "mode_title": titles[mode]},
    )



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
        mode = body.get("mode")
        if mode not in {"in", "out", "test"}:
            return JsonResponse(
                {
                    "ok": False,
                    "message": "操作の種類を選んでください。",
                },
                status=400,
            )

        data_url = body.get("image")

        if not data_url:
            return JsonResponse({"ok": False, "message": "画像がありません。"}, status=400)

        image = _decode_camera_image(data_url)
        if image is None:
            return JsonResponse({"ok": False, "message": "画像を読み込めません。"}, status=400)

        try:
            feature = extract_feature(image)
            active_ids = Student.objects.filter(
                is_active=True
            ).values_list("pk", flat=True)
            label, score, accepted, diagnostics = predict_feature(
                feature, active_ids
            )
            print(diagnostics)
        except FileNotFoundError as exc:
            return JsonResponse({
                "ok": False, "message": str(exc),
            })
        except ValueError as exc:
            return JsonResponse({
                "ok": False, "message": str(exc),
            })
 
        if not accepted:
            return JsonResponse({
                "ok": False,
                "message": "登録済みの顔と一致しませんでした。",
                "similarity": round(score, 4),
            })

        # 推論された label (通常は student.pk) から有効な生徒を取得
        student = Student.objects.filter(pk=int(label), is_active=True).first()
        if student is None:
            return JsonResponse({
                "ok": False,
                "message": "在園中の園児ではありません。",
            })

        # 数秒以内の連打による二重登録を防ぐ
        if mode == "test":
            return JsonResponse({
                "ok": True,
                "mode": "test",
                "student": student.name,
                "student_number": student.student_number,
                "message": "認証成功。出欠は記録していません。",
            })

        recent = Attendance.objects.filter(
            student=student,
            timestamp__gte=timezone.now() - timedelta(seconds=10),
        ).first()
        if recent:
            return JsonResponse({
                "ok": False,
                "message": "連続認証です。少し待ってからもう一度お試しください。"
            })

        today = timezone.localdate()
        records = Attendance.objects.filter(
            student=student,
            timestamp__date=today,
        )
        has_in = records.filter(
            attendance_type="in"
        ).exists()
        has_out = records.filter(
            attendance_type="out"
        ).exists()

        if mode == "in" and (has_in or has_out):
            return JsonResponse({
                "ok": False,
                "message": "本日の登園はすでに記録されています。",
            })

        if mode == "out":
            if not has_in:
                return JsonResponse({
                    "ok": False,
                    "message": "先に登園を記録してください。",
                })
            if has_out:
                return JsonResponse({
                    "ok": False,
                    "message": "本日の降園はすでに記録されています。",
                })

        record = Attendance.objects.create(
            student=student,
            attendance_type=mode,
        )

        redirect_url = reverse(
            "attendance:success",
            args=[record.id],
        )
        if record.attendance_type == "in":
            request.session["pending_mood_record_id"] = record.id
            redirect_url = reverse("attendance:daily_mood")

        return JsonResponse({
            "ok": True,
            "student": student.name,
            "student_number": student.student_number,
            "type": record.get_attendance_type_display(),
            "time": timezone.localtime(record.timestamp).strftime("%H:%M:%S"),
            "similarity": round(score, 4),
            "redirect_url": redirect_url,
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
    moods_by_student = {
        item.student_id: item
        for item in DailyMood.objects.filter(date=today)
    }

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
            "daily_mood": moods_by_student.get(student.pk),
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

def daily_mood(request):
    record_id = request.session.get("pending_mood_record_id")
    if not record_id:
        return redirect("attendance:top")

    record = Attendance.objects.select_related("student").filter(
        pk=record_id,
        attendance_type="in",
        timestamp__date=timezone.localdate(),
        student__is_active=True,
    ).first()
    if record is None:
        request.session.pop("pending_mood_record_id", None)
        return redirect("attendance:top")

    error = None
    if request.method == "POST":
        mood = request.POST.get("mood")
        allowed_moods = {
            value for value, label in DailyMood.MOOD_CHOICES
        }
        if mood in allowed_moods:
            DailyMood.objects.update_or_create(
                student=record.student,
                date=timezone.localdate(record.timestamp),
                defaults={"mood": mood},
            )
            request.session.pop("pending_mood_record_id", None)
            messages.success(request, "本日の気分を記録しました。")
            return redirect("attendance:top")
        error = "気分を選んでください。"

    return render(
        request,
        "attendance/daily_mood.html",
        {"student": record.student, "error": error},
    )
def daily_mood(request):
    record_id = request.session.get("pending_mood_record_id")
    if not record_id:
        return redirect("attendance:top")

    record = Attendance.objects.select_related("student").filter(
        pk=record_id,
        attendance_type="in",
        timestamp__date=timezone.localdate(),
        student__is_active=True,
    ).first()
    if record is None:
        request.session.pop("pending_mood_record_id", None)
        return redirect("attendance:top")

    error = None
    if request.method == "POST":
        mood = request.POST.get("mood")
        allowed_moods = {
            value for value, label in DailyMood.MOOD_CHOICES
        }
        if mood in allowed_moods:
            DailyMood.objects.update_or_create(
                student=record.student,
                date=timezone.localdate(record.timestamp),
                defaults={"mood": mood},
            )
            request.session.pop("pending_mood_record_id", None)
            messages.success(request, "本日の気分を記録しました。")
            return redirect("attendance:top")
        error = "気分を選んでください。"

    return render(
        request,
        "attendance/daily_mood.html",
        {"student": record.student, "error": error},
    )

# Create your views here.
