# hackathon/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # This line makes sure the 'home' URL is accessible
    path('', include('core.urls')), 
    
    path('', include('submissions.urls')),
    path('accounts/', include('django.contrib.auth.urls')), 
]