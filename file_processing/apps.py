import atexit
import json
import os

from django.apps import AppConfig

from file_processing.document_processor.colbert_utils import colber_local
from file_processing.file_queue_management.file_queue_db import create_sqlite_database
from file_processing.storage_manager import delete_temp_dir


class FileProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "file_processing"
    ran_once = False

    def ready(self):
        if not self.ran_once:
            self.ran_once = True
            # This check ensure we run only once
            create_sqlite_database()
            colber_local.initialise_search_component()

            # test_colbert()
            def on_exit():
                print('Server is stopping...')
                # Perform your cleanup actions here
                delete_temp_dir()

            atexit.register(on_exit)

    def load_document_trees(self):
        directory = os.path.abspath('./trees')

        list_of_trees = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    list_of_trees.append(json.loads(content))
        colber_local.add_documents_to_index(list_of_trees)
