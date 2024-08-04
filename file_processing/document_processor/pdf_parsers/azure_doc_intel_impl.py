import logging
import os
import tempfile
import time
import uuid
from threading import Lock

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

from file_processing.document_processor.pdf_utils import split_pdf

MODELS = {
    "default": {"requests_per_second": 1}, }


class AzureDocIntelTPS:
    def __init__(self):
        self.tps = int(os.environ.get("AZURE_DOC_INTEL_MAX_TPS", "1"))
        self.interval = 1 / self.tps
        self.last_call = time.time() - self.interval
        self.lock = Lock()
        self.temp_dir = tempfile.mkdtemp()

    def _wait_if_needed(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

    def pdf_to_md_azure_doc_intel(self, pdf_filepath: str) -> str | None:
        self._wait_if_needed()
        split_pdf_paths = split_pdf(pdf_filepath,
                                    os.path.join(self.temp_dir, f'{uuid.uuid4()}'),
                                    int(os.environ.get("AZURE_DOC_INTEL_MAX_PAGES", "2")))
        md_output = ""
        for split_pdf_path in split_pdf_paths:
            # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTEL_ENDPOINT"),
                api_key=os.environ.get("AZURE_DOC_INTEL_API_KEY"),
                file_path=split_pdf_path[0],
                api_model="prebuilt-layout",
                mode="markdown"
            )

            documents = loader.load()
            for document in documents:
                md_output += document.page_content + "\n\n"
        # TODO: decide what to do with the pages
        return md_output


azure_doc_intel_impl = AzureDocIntelTPS()


def pdf_to_md_azure_doc_intel(pdf_filepath: str) -> str | None:
    return azure_doc_intel_impl.pdf_to_md_azure_doc_intel(pdf_filepath)


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure Document Intelligence')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_azure_doc_intel(args.input_pdf))
