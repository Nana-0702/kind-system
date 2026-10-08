from django.db import models
 
 
class Student(models.Model):
    CLASS_CHOICES = [
        ("sakura","さくら組"),
        ("himawari","ひまわり組"),
        ("bara","ばら組"),
        ("yuri","ゆり組"), 
        ("tanpopo","たんぽぽ組")

    ]
    name = models.CharField("園児名", max_length=100)
    student_number = models.CharField("園児番号", max_length=30, unique=True)
    class_name = models.CharField("クラス名",max_length=50,choices=CLASS_CHOICES,blank=True)
    photo = models.ImageField("顔写真", upload_to="students/")
    is_active = models.BooleanField("在園", default=True)
 
    def __str__(self):
        return f"{self.student_number} - {self.name}"
 
 
class Attendance(models.Model):
    TYPE_CHOICES = [
        ("in", "登園"),
        ("out", "降園"),
    ]
 
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="attendance_records",
    )
    attendance_type = models.CharField(
        "区分",
        max_length=10,
        choices=TYPE_CHOICES,
    )
    timestamp = models.DateTimeField("時刻", auto_now_add=True)

    
    class Meta:
        ordering = ["-timestamp"]
 
    def __str__(self):
        return f"{self.student.name} {self.get_attendance_type_display()} {self.timestamp}"

class StudentFacePhoto(models.Model):
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="face_photos",
    )
    photo = models.ImageField(
        "追加の顔写真", upload_to="student_faces/"
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return f"{self.student.student_number} / {self.pk}"



# Create your models here.
