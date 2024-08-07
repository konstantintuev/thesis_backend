from file_processing.document_processor.pdf_parsers.azure_doc_intel_impl import pdf_to_md_azure_doc_intel
from file_processing.document_processor.pdf_parsers.gpt4o_pdf_parse_impl import pdf_to_md_gpt4o
from file_processing.document_processor.pdf_parsers.llama_parse_impl import pdf_to_md_llama_parse
from file_processing.document_processor.pdf_parsers.open_source_local_parsers import pdf_to_md_pymupdf, \
    pdf_to_md_pdf_miner


class PdfToMdPageInfo:
    original_page_index: int
    raw_md_content: str
    fixed_md_content: str
    fix_log: str
    page_screenshot_path: str

    def __init__(self, original_page_index: int, raw_md_content: str, fixed_md_content: str, fix_log: str,
                 page_screenshot_path: str):
        super().__init__()
        self.original_page_index = original_page_index
        self.raw_md_content = raw_md_content
        self.fixed_md_content = fixed_md_content
        self.fix_log = fix_log
        self.page_screenshot_path = page_screenshot_path

def pdf_to_md(pdf_filepath: str,
              how_good: int = 1,
              only_local: bool = False) -> str | None:
    if only_local:
        if how_good == 1:
            # TODO: better
            return pdf_to_md_pymupdf(pdf_filepath)
        elif how_good == 2:
            return pdf_to_md_pymupdf(pdf_filepath)
        elif how_good == 3:
            return pdf_to_md_pdf_miner(pdf_filepath)
    else:
        # TODO: add Azure Doc Intelligence + GPT 4o
        if how_good == 1:
            return pdf_to_md_gpt4o(pdf_filepath)
        elif how_good == 2:
            return pdf_to_md_azure_doc_intel(pdf_filepath)
        elif how_good == 3:
            return pdf_to_md_llama_parse(pdf_filepath)