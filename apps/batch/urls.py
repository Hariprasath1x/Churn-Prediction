from django.urls import path
from .views import BatchPredictView

urlpatterns = [
    path('predict/', BatchPredictView.as_view(), name='batch-predict'),
]
