# submissions/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.upload_submission, name='upload_submission'),
    path('leaderboard/', views.leaderboard, name='leaderboard'),
    path('my-submissions/', views.my_submissions, name='my_submissions'),
]