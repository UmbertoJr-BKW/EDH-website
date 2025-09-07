# submissions/models.py

import uuid
import os
from django.db import models
from django.contrib.auth.models import User


# Defines the state of a submission
class SubmissionStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending Evaluation'
    PROCESSING = 'PROCESSING', 'Processing'
    COMPLETE = 'COMPLETE', 'Complete'
    ERROR = 'ERROR', 'Error'

class Submission(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=SubmissionStatus.choices, default=SubmissionStatus.PENDING)

    def get_upload_path(instance, filename):
        # --- DEBUG PRINT 4: See if this function is called ---
        print(f"--- INSIDE get_upload_path ---")
        print(f"    -> Instance ID: {instance.id}")
        print(f"    -> Instance User: {instance.user}")
        print(f"    -> Original Filename: {filename}")
        path = f'uploads/{instance.user.username}/{instance.id}/{filename}'
        print(f"    -> Generated Path: {path}")
        print(f"----------------------------")
        return path

    # First 7 are required
    file1 = models.FileField(upload_to=get_upload_path) # base_consumption.parquet
    file2 = models.FileField(upload_to=get_upload_path) # pv_profiles.parquet
    file3 = models.FileField(upload_to=get_upload_path) # ev_profiles.parquet
    file4 = models.FileField(upload_to=get_upload_path) # hp_profiles.parquet
    file5 = models.FileField(upload_to=get_upload_path) # battery_in_profiles.parquet
    file6 = models.FileField(upload_to=get_upload_path) # battery_out_profiles.parquet
    file7 = models.FileField(upload_to=get_upload_path) # battery_soc_profiles.parquet

    # Last 2 are optional
    file8 = models.FileField(upload_to=get_upload_path, blank=True, null=True) # curtailed_energy_profiles.parquet
    file9 = models.FileField(upload_to=get_upload_path, blank=True, null=True) # power_to_hydro.parquet

    def __str__(self):
        return f"Submission by {self.user.username} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    # New helper method to list submitted files
    def get_submitted_files(self):
        files = []
        # Loop through all 9 file fields
        for i in range(1, 10):
            field = getattr(self, f'file{i}')
            if field: # This checks if the field has a file
                # os.path.basename gets just the filename from the full path
                files.append(os.path.basename(field.name))
        return files
    
class EvaluationResult(models.Model):
    # A one-to-one link to the submission it evaluates
    submission = models.OneToOneField(Submission, on_delete=models.CASCADE, primary_key=True)
    score = models.FloatField()
    details = models.JSONField(blank=True, null=True) # For storing more complex results
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for {self.submission}: Score {self.score}"