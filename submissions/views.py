# submissions/views.py

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import SubmissionForm
from .models import Submission, EvaluationResult # Ensure all models are imported
from storages.backends.gcloud import GoogleCloudStorage

@login_required
def upload_submission(request):
    if request.method == 'POST':
        
        # --- The Definitive Upload Logic ---

        # 1. Create and save the submission record WITHOUT any files.
        #    This gets a UUID and a record in the database first.
        submission = Submission.objects.create(user=request.user)
        print(f"-> Step 1: Created initial DB record with ID: {submission.id}")

        # 2. Manually instantiate the correct storage backend.
        #    We are no longer relying on Django's `default_storage`.
        storage = GoogleCloudStorage()
        print(f"-> Step 2: Manually created storage object of type: {type(storage)}")

        # 3. Loop through the files, upload them, and update the model.
        files_to_upload = {
            'file1': request.FILES.get('file1'),
            'file2': request.FILES.get('file2'),
            'file3': request.FILES.get('file3'),
            'file4': request.FILES.get('file4'),
            'file5': request.FILES.get('file5'),
            'file6': request.FILES.get('file6'),
            'file7': request.FILES.get('file7'),
        }

        all_files_valid = True
        for field_name, uploaded_file in files_to_upload.items():
            if uploaded_file:
                # Generate the path using our model's function.
                file_path = submission.get_upload_path(uploaded_file.name)
                
                # Use our manual storage object to save the file.
                storage.save(file_path, uploaded_file)
                
                # Set the file path on the model instance.
                setattr(submission, field_name, file_path)
                print(f"   -> Manually uploaded {file_path}")
            else:
                # Handle case where a file is missing
                print(f"   -> WARNING: File for '{field_name}' is missing!")
                all_files_valid = False
        
        if all_files_valid:
            # 4. Save the submission again to update all the file path fields.
            submission.save()
            print("-> Step 4: Updated submission record with all file paths.")
        else:
            # Optional: Delete the submission if it's incomplete.
            submission.delete()
            print("-> ERROR: Incomplete submission. Record deleted.")
            # You could redirect to an error page here.
            return redirect('upload_submission') # Or wherever you want to send them
        

        return redirect('my_submissions')
    else:
        return render(request, 'submissions/upload_form.html')


# Add these to your existing submissions/views.py
from .models import Submission, EvaluationResult

def leaderboard(request):
    # Get all results, order by the highest score first
    results = EvaluationResult.objects.order_by('-score')
    return render(request, 'submissions/leaderboard.html', {'results': results})

@login_required
def my_submissions(request):
    # Get submissions only for the currently logged-in user
    user_submissions = Submission.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'submissions/my_submissions.html', {'submissions': user_submissions})