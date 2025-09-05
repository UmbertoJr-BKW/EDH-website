from django.db import models
from django.contrib.auth.models import User

# Defines the state of a submission
class SubmissionStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending Evaluation'
    PROCESSING = 'PROCESSING', 'Processing'
    COMPLETE = 'COMPLETE', 'Complete'
    ERROR = 'ERROR', 'Error'

class Submission(models.Model):
    # Link to the user who submitted
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=SubmissionStatus.choices, default=SubmissionStatus.PENDING)

    # Fields for each of the 7 Parquet files
    # The `upload_to` path will be like: 'uploads/username/submission_id/file1.parquet'
    def get_upload_path(instance, filename):
        return f'uploads/{instance.user.username}/{instance.id}/{filename}'

    file1 = models.FileField(upload_to=get_upload_path)
    file2 = models.FileField(upload_to=get_upload_path)
    file3 = models.FileField(upload_to=get_upload_path)
    file4 = models.FileField(upload_to=get_upload_path)
    file5 = models.FileField(upload_to=get_upload_path)
    file6 = models.FileField(upload_to=get_upload_path)
    file7 = models.FileField(upload_to=get_upload_path)

    def __str__(self):
        return f"Submission by {self.user.username} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class EvaluationResult(models.Model):
    # A one-to-one link to the submission it evaluates
    submission = models.OneToOneField(Submission, on_delete=models.CASCADE, primary_key=True)
    score = models.FloatField()
    details = models.JSONField(blank=True, null=True) # For storing more complex results
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for {self.submission}: Score {self.score}"