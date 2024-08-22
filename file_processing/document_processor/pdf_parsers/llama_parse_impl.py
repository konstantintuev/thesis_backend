import os

from llama_parse import LlamaParse

from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument, PdfToMdPageInfo
from file_processing.document_processor.pdf_utils import split_pdf

llama_parser = LlamaParse(
    # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=6,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)


def pdf_to_md_llama_parse_standard(pdf_filepath: str) -> str | None:
    documents = llama_parser.load_data(pdf_filepath)
    if documents is None or len(documents) == 0:
        return None
    document = documents[0]
    return document.text


def pdf_to_md_llama_parse(pdf_filepath: str, output_dir: str) -> PdfToMdDocument:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_pdf_paths = split_pdf(pdf_filepath,
                                output_dir,
                                1,
                                True)
    pages_out = PdfToMdDocument()
    for split_pdf_path in split_pdf_paths:
        # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
        documents = llama_parser.load_data(pdf_filepath)
        if documents is None or len(documents) == 0:
            return PdfToMdDocument()
        raw_page_md_content = ""
        for document in documents:
            raw_page_md_content += document.text + "\n\n"
        pages_out.append(PdfToMdPageInfo(
            split_pdf_path.from_original_start_page,
            raw_page_md_content,
            "",
            "",
            split_pdf_path.screenshots_per_page[0]
        ))
    return pages_out
