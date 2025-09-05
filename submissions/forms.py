from django import forms
from .models import Submission

class SubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        # We only want users to upload files. The user and status are set automatically.
        fields = ['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7']