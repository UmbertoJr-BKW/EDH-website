# submissions/admin.py

from django.contrib import admin
from .models import Submission, EvaluationResult

class SubmissionAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'status', 'is_disqualified')
    list_filter = ('status', 'is_disqualified', 'created_at')
    # Make the is_disqualified field a clickable checkbox in the list view
    list_editable = ('is_disqualified',)
    search_fields = ('user__username',)
    readonly_fields = ('id', 'created_at')
    
    # Organize the detail view
    fieldsets = (
        (None, {
            'fields': ('id', 'user', 'created_at', 'status')
        }),
        ('Control', {
            'fields': ('is_disqualified',)
        }),
        ('Files', {
            'fields': ('file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8', 'file9')
        }),
    )

# Register your models with the custom admin class
admin.site.register(Submission, SubmissionAdmin)
admin.site.register(EvaluationResult)