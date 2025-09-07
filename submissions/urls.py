# submissions/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Give the leaderboard its own explicit URL again
    path('leaderboard/', views.leaderboard, name='leaderboard'), 
    path('submit/', views.upload_submission, name='upload_submission'),
    path('my-submissions/', views.my_submissions, name='my_submissions'),
]