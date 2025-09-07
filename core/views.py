# core/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import RegisterForm

def home(request):
    """
    Renders the main landing page.
    """
    return render(request, 'core/home.html')

def register(request):
    """
    Handles user registration.
    """
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in immediately after registration
            return redirect('my_submissions') # Redirect to their submissions page
    else:
        form = RegisterForm()
    
    return render(request, 'core/register.html', {'form': form})