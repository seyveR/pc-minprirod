from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path("", views.start, name="start"),
    path("analyz/", views.analyz, name="analyz"),
    path("analyz/filtration/", views.filtration, name="filtration"),
    path("upload_files/",views.upload_files,name="upload_files"),
    path("analyz/results/", views.results, name="results"),
]