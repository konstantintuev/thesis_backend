import json
import uuid
from typing import List, Dict

from django.db import connection


def create_sqlite_database():
    """ create a database connection to an SQLite database """
    with connection.cursor() as cursor:
        # track multiple files uploaded at the same time
        cursor.execute("""CREATE TABLE IF NOT EXISTS multiple_files_queue (
    multiple_files_uuid TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS file_queue (
    file_uuid TEXT PRIMARY KEY,
    status TEXT DEFAULT 'pending',
    result TEXT DEFAULT NULL,
    file_path TEXT,
    mime_type TEXT,
    parent_queue TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    done_at TIMESTAMP DEFAULT NULL,
    FOREIGN KEY (parent_queue) REFERENCES multiple_files_queue (multiple_files_uuid) ON DELETE SET NULL
)""")

def add_multiple_files_to_queue(file_list: List[Dict[str, str]]) -> str:
    with connection.cursor() as cursor:
        multiple_files_uuid = str(uuid.uuid4())
        cursor.execute("INSERT INTO multiple_files_queue (multiple_files_uuid) VALUES (%s)", [multiple_files_uuid])
        for file in file_list:
            cursor.execute("INSERT INTO file_queue (file_uuid, file_path, mime_type, parent_queue) VALUES (%s, %s, %s, %s)",
                           [file["file_uuid"], file["file_path"], file["mime_type"], multiple_files_uuid])
    return multiple_files_uuid

def add_file_to_queue(parent_queue_uuid: str or None, file_uuid: str, file_path: str, mime_type: str) -> str:
    with connection.cursor() as cursor:
        if parent_queue_uuid:
            cursor.execute("INSERT INTO file_queue (file_uuid, file_path, mime_type, parent_queue) VALUES (%s, %s, %s, %s)",
                           [file_uuid, file_path, mime_type, parent_queue_uuid])
        else:
            cursor.execute("INSERT INTO file_queue (file_uuid, file_path, mime_type) VALUES (%s, %s, %s)",
                           [file_uuid, file_path, mime_type])
    return file_uuid


def get_file_from_queue(file_uuid: str) -> dict or None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM file_queue WHERE file_uuid = %s", [file_uuid])
        file_queue_item = cursor.fetchone()
        if file_queue_item and file_queue_item[1] == 'done':
            return {"file_uuid": file_queue_item[0],
                    "status": file_queue_item[1],
                    "result": json.loads(file_queue_item[2])}
        else:
            return None

def get_all_files_queue() -> dict:
    with connection.cursor() as cursor:
        cursor.execute("SELECT file_uuid, status, result FROM file_queue")
        res = cursor.fetchall()
        if res:
            return {"files": [{"file_uuid": file_queue_item[0],
                               "status": file_queue_item[1],
                               "result": json.loads(file_queue_item[2])}
                    for file_queue_item in res if file_queue_item[1] == 'done']}
        else:
            return {"status": "pending"}

def set_file_status(file_uuid: str, status: str, result: str) -> str:
    with connection.cursor() as cursor:
        cursor.execute("UPDATE file_queue SET status = %s, result = %s, done_at=CURRENT_TIMESTAMP WHERE file_uuid = %s",
                       [status, result, file_uuid])
    return file_uuid

def get_multiple_files_queue(multiple_files_uuid: str) -> dict or None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT file_uuid, status, result FROM file_queue WHERE parent_queue = %s", [multiple_files_uuid])
        res = cursor.fetchall()
        if res and all(file_queue_item[1] == 'done' for file_queue_item in res):
            return {"files": [{"file_uuid": file_queue_item[0],
                               "status": file_queue_item[1],
                               "result": json.loads(file_queue_item[2])}
                    for file_queue_item in res]}
        else:
            return {"status": "pending"}
