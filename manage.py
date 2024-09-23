#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
    #logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
    logging.getLogger('langchain_community.document_loaders.parsers.doc_intelligence').setLevel(logging.ERROR)
    #logging.getLogger('httpx').setLevel(logging.WARNING)

    class LoggerWriter:
        def __init__(self, logfct):
            self.logfct = logfct
            self.buf = []

        def write(self, msg):
            if msg.endswith('\n'):
                self.buf.append(msg.removesuffix('\n'))
                self.logfct(''.join(self.buf))
                self.buf = []
            else:
                self.buf.append(msg)

        def flush(self):
            pass

    # To access the original stdout/stderr, use sys.__stdout__/sys.__stderr__
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thesis_backend.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
