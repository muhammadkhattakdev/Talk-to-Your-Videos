from django.urls import path
from .views import chat_with_video, health_check

urlpatterns = [
    path('chat/', chat_with_video, name='chat_with_video'),
    path('health/', health_check, name='health_check'),
]
