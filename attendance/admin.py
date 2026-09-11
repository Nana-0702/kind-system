from django.contrib import admin
from .models import Student, Attendance
 
 
@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("student_number", "name", "is_active")
    search_fields = ("student_number", "name")
 
 
@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ("student", "attendance_type", "timestamp")
    list_filter = ("attendance_type", "timestamp")
    search_fields = ("student__name", "student__student_number")

# Register your models here.
