from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('text_2_query', views.text_2_query, name='text_2_query'),
    path('rewrite_query', views.rewrite_query, name='rewrite_query'),
    path('ask_file', views.ask_file, name='ask_file'),
]