# submissions/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('leaderboard/', views.leaderboard, name='leaderboard'),
    path('submit/', views.upload_submission, name='upload_submission'),
    path('my-submissions/', views.my_submissions, name='my_submissions'),
    path('visualize/', views.visualize_scores, name='visualize_scores'),
]