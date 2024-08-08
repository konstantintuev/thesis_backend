import shutil
import tempfile

global_temp_dir = tempfile.mkdtemp()


def delete_temp_dir():
    shutil.rmtree(global_temp_dir, ignore_errors=True)
