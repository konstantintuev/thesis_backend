from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("retrieve_file_from_queue", views.retrieve_file_from_queue, name="retrieve_file_from_queue"),
    path('embeddings', views.generate_embeddings, name='generate_embeddings'),
    path('files_to_chunks', views.files_to_chunks, name='files_to_chunks'),
    path('retrieve_multiple_files_from_queue', views.retrieve_multiple_files_from_queue, name='retrieve_multiple_files_from_queue'),
    path('search_query', views.search_query, name='search_query'),
    path('available_processors', views.get_available_processors, name='available_processors'),
    path('text_2_query', views.text_2_query, name='text_2_query'),
    path('rewrite_query', views.rewrite_query, name='rewrite_query'),
    path('ask_file', views.ask_file, name='ask_file'),
    path('rerank_results', views.rerank_results, name='rerank_results')
]