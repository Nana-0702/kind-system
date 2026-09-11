from django import forms
from .models import Student
 
 
class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ["name", "student_number", "photo"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "student_number": forms.TextInput(attrs={"class": "form-control"}),
            "photo": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }