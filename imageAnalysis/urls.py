from django.urls import path
from imageAnalysis import views

urlpatterns = [
    path('image/', views.Analysis.as_view(), name='image'),
    path('getAll/', views.GetImages.as_view(), name='images'),
]
