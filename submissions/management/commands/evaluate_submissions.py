# submissions/management/commands/evaluate_submissions.py

import pandas as pd
from django.core.management.base import BaseCommand
from submissions.models import Submission, EvaluationResult, SubmissionStatus
from storages.backends.gcloud import GoogleCloudStorage # <-- IMPORT

class Command(BaseCommand):
    help = 'Evaluates all pending submissions'

    def handle(self, *args, **options):
        # Manually instantiate the storage backend, just like in the view
        storage = GoogleCloudStorage()

        pending_submissions = Submission.objects.filter(status=SubmissionStatus.PENDING)

        if not pending_submissions.exists():
            self.stdout.write(self.style.SUCCESS('No pending submissions to evaluate.'))
            return

        for sub in pending_submissions:
            self.stdout.write(f'Processing submission {sub.id} from {sub.user.username}...')
            sub.status = SubmissionStatus.PROCESSING
            sub.save()

            try:
                # --- THIS IS THE KEY CHANGE ---
                # Instead of sub.file1.path, we use storage.open()
                with storage.open(sub.file1.name) as f:
                    df1 = pd.read_parquet(f)
                with storage.open(sub.file2.name) as f:
                    df2 = pd.read_parquet(f)
                # ... repeat for all 7 files ...

                # Your evaluation logic remains the same
                calculated_score = len(df1)
                
                EvaluationResult.objects.create(
                    submission=sub,
                    score=calculated_score,
                    details={'message': 'Evaluation successful', 'num_rows': calculated_score}
                )

                sub.status = SubmissionStatus.COMPLETE
                sub.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully evaluated submission {sub.id}. Score: {calculated_score}'))

            except Exception as e:
                sub.status = SubmissionStatus.ERROR
                sub.save()
                EvaluationResult.objects.create(
                    submission=sub,
                    score=0,
                    details={'error': str(e)}
                )
                self.stderr.write(self.style.ERROR(f'Error evaluating submission {sub.id}: {e}'))