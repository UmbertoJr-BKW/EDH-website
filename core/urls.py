# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # This line creates the URL pattern named 'home'
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
]