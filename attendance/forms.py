from django import forms
from .models import Student 


class MultipleImageInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleImageField(forms.ImageField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleImageInput())
        super().__init__(*args, **kwargs)
 
    def clean(self, data, initial=None):
        if not data:
            return []
        clean_one = super().clean
        if isinstance(data, (list, tuple)):
            return [clean_one(file, initial) for file in data]
        return [clean_one(data, initial)]

class StudentForm(forms.ModelForm):
    # モデルにないカスタム複数画像フィールドを定義
    face_photos = MultipleImageField(
        label="追加の顔写真", 
        required=False,
        widget=MultipleImageInput(attrs={"class": "form-control"}) # widgetのスタイルもここで指定
    )
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 保留空选项，但不显示提示文字 (空の選択肢のラベルを非表示に)
        choices = list(self.fields["class_name"].choices)
        if choices and choices[0][0] == "":
            choices[0] = ("", "")
        self.fields["class_name"].choices = choices
 
    class Meta:
        model = Student
        # face_photosはモデルのフィールドではないため、Meta.fieldsからは除外します
        fields = ["name", "student_number", "class_name", "photo"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "student_number": forms.TextInput(attrs={"class": "form-control"}),
            "class_name": forms.Select(attrs={"class": "form-control"}),
            "photo": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }