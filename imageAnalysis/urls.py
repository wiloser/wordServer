from django.urls import path
from imageAnalysis import views

urlpatterns = [
    path('image/', views.Analysis.as_view(), name='image'),
]
