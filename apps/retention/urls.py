from django.urls import path
from .views import AIStrategyView, ReportView

urlpatterns = [
    path('ai-strategy/', AIStrategyView.as_view(), name='ai-strategy'),
    path('report/', ReportView.as_view(), name='report'),
]
