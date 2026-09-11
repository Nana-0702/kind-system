from django.db import models
 
 
class Student(models.Model):
    name = models.CharField("園児名", max_length=100)
    student_number = models.CharField("園児番号", max_length=30, unique=True)
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





# Create your models here.
