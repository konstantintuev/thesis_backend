from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload_pdf", views.upload_pdf, name="upload_pdf"),
    path("pdf_to_chunks", views.pdf_to_chunks, name="pdf_to_chunks"),
    path("retrieve_file_from_queue", views.retrieve_file_from_queue, name="retrieve_file_from_queue"),
    path('embeddings', views.generate_embeddings, name='generate_embeddings'),
    path('files_to_chunks', views.files_to_chunks, name='files_to_chunks'),
    path('retrieve_multiple_files_from_queue', views.retrieve_multiple_files_from_queue, name='retrieve_multiple_files_from_queue'),
    path('search_query', views.search_query, name='search_query'),
]