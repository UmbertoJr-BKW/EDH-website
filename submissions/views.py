from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import SubmissionForm

@login_required
def upload_submission(request):
    if request.method == 'POST':
        form = SubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            # Don't save to the database yet
            submission = form.save(commit=False)
            # Assign the current logged-in user
            submission.user = request.user
            # Now save the submission instance
            submission.save()
            # Redirect to a success page or their dashboard
            return redirect('my_submissions') # We will create this URL name later
    else:
        form = SubmissionForm()
    return render(request, 'submissions/upload_form.html', {'form': form})


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