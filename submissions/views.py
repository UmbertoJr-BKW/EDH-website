# submissions/views.py
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Submission, EvaluationResult
from storages.backends.gcloud import GoogleCloudStorage

@login_required
def upload_submission(request):
    if request.method == 'POST':
        submission = Submission.objects.create(user=request.user)
        storage = GoogleCloudStorage()

        files_to_upload = {
            'file1': request.FILES.get('file1'),
            'file2': request.FILES.get('file2'),
            'file3': request.FILES.get('file3'),
            'file4': request.FILES.get('file4'),
            'file5': request.FILES.get('file5'),
            'file6': request.FILES.get('file6'),
            'file7': request.FILES.get('file7'),
            'file8': request.FILES.get('file8'), # Optional
            'file9': request.FILES.get('file9'), # Optional
        }

        # Validate that the first 7 (required) files are present
        required_files_present = all(files_to_upload[f'file{i}'] for i in range(1, 8))

        if not required_files_present:
            submission.delete()
            # You should add a Django message here to give user feedback
            # messages.error(request, "Submission failed. The first 7 files are required.")
            print("-> ERROR: Incomplete submission. Required files missing. Record deleted.")
            return redirect('upload_submission')

        # Upload all provided files
        for field_name, uploaded_file in files_to_upload.items():
            if uploaded_file:
                file_path = submission.get_upload_path(uploaded_file.name)
                storage.save(file_path, uploaded_file)
                setattr(submission, field_name, file_path)
                print(f"   -> Manually uploaded {file_path}")

        submission.save()
        print("-> Step 4: Updated submission record with all file paths.")
        return redirect('my_submissions')
    else:
        return render(request, 'submissions/upload_form.html')


# Add these to your existing submissions/views.py
from .models import Submission, EvaluationResult

def leaderboard(request):
    # Define valid sort fields to prevent users from sorting by arbitrary columns
    valid_sort_fields = ['score_objective_1', 'score_objective_2', 'score_objective_3']
    
    # Get the sort parameter from the URL, default to the first objective
    sort_by = request.GET.get('sort_by', 'score_objective_1')

    # Ensure the requested sort field is valid, otherwise use the default
    if sort_by not in valid_sort_fields:
        sort_by = 'score_objective_1'
        
    # Order the results. The '-' prefix means descending order (higher score is better).
    order_string = f'-{sort_by}'
    results = EvaluationResult.objects.order_by(order_string)
    
    context = {
        'results': results,
        'current_sort_by': sort_by, # Pass the current sort field to the template
    }
    return render(request, 'submissions/leaderboard.html', context)

@login_required
def my_submissions(request):
    # Get submissions only for the currently logged-in user
    user_submissions = Submission.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'submissions/my_submissions.html', {'submissions': user_submissions})


def find_pareto_frontier(scores):
    """
    Identifies the Pareto optimal frontier from a list of scores.
    Each score is a dict with 's1', 's2', 's3'.
    Objective: Minimize s1, Maximize s2, Maximize s3.
    """
    pareto_indices = set()
    num_scores = len(scores)
    
    for i in range(num_scores):
        current_score = scores[i]
        is_dominated = False
        
        for j in range(num_scores):
            if i == j:
                continue
            
            other_score = scores[j]
            
            # A point is dominated if another point is:
            # - Strictly better in at least one objective
            # - No worse in all other objectives
            
            # Check if other_score dominates current_score
            if (other_score['s1'] <= current_score['s1'] and 
                other_score['s2'] >= current_score['s2'] and 
                other_score['s3'] >= current_score['s3']) and \
               (other_score['s1'] < current_score['s1'] or 
                other_score['s2'] > current_score['s2'] or 
                other_score['s3'] > current_score['s3']):
                is_dominated = True
                break # Dominated, no need to check further
                
        if not is_dominated:
            pareto_indices.add(i)
            
    return pareto_indices


def visualize_scores(request):
    results = EvaluationResult.objects.select_related('submission__user').all()
    
    # First, get the raw scores
    raw_scores = [
        {
            'user': result.submission.user.username,
            's1': result.score_objective_1,
            's2': result.score_objective_2,
            's3': result.score_objective_3,
        }
        for result in results
    ]
    
    # --- NEW LOGIC INTEGRATION ---
    # Find the indices of the points on the frontier
    pareto_indices = find_pareto_frontier(raw_scores)
    
    # Now, add the 'is_pareto' flag to our data
    # We rebuild the list to ensure the order is maintained
    score_data_with_pareto = []
    for i, score in enumerate(raw_scores):
        score['is_pareto'] = (i in pareto_indices)
        score_data_with_pareto.append(score)
    # ----------------------------

    context = {
        'scores_data': score_data_with_pareto
    }
    
    return render(request, 'submissions/visualize.html', context)