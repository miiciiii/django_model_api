from django.urls import path
from . import views
from .views import PredictionAPIView, ResNet50V2APIView  # Import the new API view

urlpatterns = [
    path('api/predict/', PredictionAPIView.as_view(), name='predict'),  # API endpoint for predictions
    path('api/res/', ResNet50V2APIView.as_view(), name='respred'),
    path('', views.test, name='test'),  # Optional: Home view for rendering index.html
]