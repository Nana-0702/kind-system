from django.contrib import admin
from .models import Student, Attendance, StudentFacePhoto


class StudentFacePhotoInline(admin.TabularInline):
    model = StudentFacePhoto
    extra = 1


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    # StudentFacePhotoInlineを管理画面に組み込む
    inlines = [StudentFacePhotoInline]
    
    # リスト表示するフィールド（より詳細な設定に統合しました）
    list_display = (
        "student_number", 
        "name", 
        "class_name", 
        "is_active"
    )
    
    # 検索機能の設定
    search_fields = ("student_number", "name")


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ("student", "attendance_type", "timestamp")
    list_filter = ("attendance_type", "timestamp")
    search_fields = ("student__name", "student__student_number")


# Register your models here.
