from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomePage, name='home'),
    path('aboutMe/', views.AboutPage, name='about')
]