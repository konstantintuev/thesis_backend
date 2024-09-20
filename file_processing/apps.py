import atexit
import json
import os

from django.apps import AppConfig

from file_processing.document_processor.colbert_utils_pylate import colbert_local
from file_processing.embeddings import pending_embeddings_singleton, BGEM3Flag
from file_processing.file_queue_management.file_queue_db import create_sqlite_database, get_all_files_queue
from file_processing.query_processor.rerankers_local import rerankers_instance
from file_processing.storage_manager import delete_temp_dir


def load_document_trees():
    directory = os.path.abspath('./trees')

    list_of_trees = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):# and filename.startswith("do_"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                list_of_trees.append(json.loads(content))
    if len(list_of_trees) > 0:
        colbert_local.add_documents_to_index(list_of_trees)

def load_all_from_db():
    files = get_all_files_queue()
    if not files or "files" not in files:
        return
    files = files["files"]
    for file in files:
        file["tree"] = file["result"]["tree"]
        file["uuid_items"] = file["result"]["uuid_items"]
        file["metadata"] = file["result"]["metadata"]
        file["result"] = None
    colbert_local.add_documents_to_index(files)


class FileProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "file_processing"
    ran_once = False

    def ready(self):
        if not self.ran_once:
            self.ran_once = True
            # This check ensure we run only once
            create_sqlite_database()

            if colbert_local.init_model():
                # Need to restore index
                load_all_from_db()
            colbert_local.initialise_search_component()
            rerankers_instance.init_rerankers()
            pending_embeddings_singleton.init_mode(BGEM3Flag())
            #load_document_trees()

            # test_colbert()
            def on_exit():
                print('Server is stopping...')
                # Perform your cleanup actions here
                delete_temp_dir()

            atexit.register(on_exit)
