# submissions/management/commands/evaluate_submissions.py
# (This is an example, you need to adapt it)

# Make sure to import random if you are testing with random scores
import random 
from django.core.management.base import BaseCommand
from submissions.models import Submission, SubmissionStatus, EvaluationResult

class Command(BaseCommand):
    help = 'Evaluates pending submissions'

    def handle(self, *args, **options):
        pending_submissions = Submission.objects.filter(status=SubmissionStatus.PENDING)
        self.stdout.write(f"Found {pending_submissions.count()} submissions to evaluate.")

        for submission in pending_submissions:
            try:
                submission.status = SubmissionStatus.PROCESSING
                submission.save()

                # --- YOUR SCORING LOGIC GOES HERE ---
                # 1. Download the files from GCS for this submission.
                # 2. Run your three objective functions.
                # 3. Get the three scores.
                
                # --- EXAMPLE with placeholder scores ---
                # Replace this with your actual calculations
                score1 = random.uniform(80.0, 100.0) 
                score2 = random.uniform(500.0, 1000.0)
                score3 = random.uniform(0.1, 0.9)
                # --- End of example ---

                # Use update_or_create to save the results
                EvaluationResult.objects.update_or_create(
                    submission=submission,
                    defaults={
                        # --- USE THE NEW FIELD NAMES ---
                        'grid_costs': score1,
                        'renewables_installed': score2,
                        'autarchy_rate': score3,
                    }
                )

                submission.status = SubmissionStatus.COMPLETE
                self.stdout.write(self.style.SUCCESS(f"Successfully evaluated submission {submission.id}"))

            except Exception as e:
                submission.status = SubmissionStatus.ERROR
                self.stdout.write(self.style.ERROR(f"Error evaluating submission {submission.id}: {e}"))
            
            finally:
                submission.save()