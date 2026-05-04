from tkinter import Image

from django import forms
class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['fotos']