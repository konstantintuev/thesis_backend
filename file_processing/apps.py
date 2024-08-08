import atexit
import os

from django.apps import AppConfig

from file_processing.document_processor.colbert_utils import test_colbert, initialise_search_component
from file_processing.file_queue_management.file_queue_db import create_sqlite_database
from file_processing.storage_manager import delete_temp_dir


class FileProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "file_processing"

    def ready(self):
        if os.environ.get('RUN_MAIN'):
            # This check ensure we run only once
            create_sqlite_database()
            initialise_search_component()

            # test_colbert()
            def on_exit():
                print('Server is stopping...')
                # Perform your cleanup actions here
                delete_temp_dir()

            atexit.register(on_exit)
