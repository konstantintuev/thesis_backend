import os
import uuid

from file_processing.document_processor.pdf_parsers.azure_doc_intel_impl import pdf_to_md_azure_doc_intel
from file_processing.document_processor.pdf_parsers.azure_gpt_4o import pdf_to_md_azure_doc_gpt4o
from file_processing.document_processor.pdf_parsers.gpt4o_pdf_parse_impl import pdf_to_md_gpt4o
from file_processing.document_processor.pdf_parsers.llama_parse_impl import pdf_to_md_llama_parse
from file_processing.document_processor.pdf_parsers.open_source_local_parsers import pdf_to_md_pymupdf, \
    pdf_to_md_pdf_miner
from file_processing.storage_manager import global_temp_dir


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

    def get_best_text_content(self) -> str:
        # fixed_md_content is a processed version of raw_md_content,
        #   if we have it, we return it, else the raw_md_content
        if len(self.fixed_md_content) > 0:
            return self.fixed_md_content
        else:
            return self.raw_md_content


class PdfToMdDocument(list[PdfToMdPageInfo]):
    def get_best_text_content(self) -> str:
        return "\n\n".join([it.get_best_text_content() for it in self])


def pdf_to_md(pdf_filepath: str,
              how_good: int = 1,
              only_local: bool = False,
              file_uuid: str = f'{uuid.uuid4()}') -> PdfToMdDocument:
    out_dir = os.path.join(global_temp_dir, file_uuid)
    if only_local:
        if how_good == 1:
            # TODO: better
            return pdf_to_md_pymupdf(pdf_filepath, out_dir)
        elif how_good == 2:
            return pdf_to_md_pymupdf(pdf_filepath, out_dir)
        elif how_good == 3:
            return pdf_to_md_pdf_miner(pdf_filepath, out_dir)
    else:
        if how_good == 1:
            return pdf_to_md_azure_doc_gpt4o(pdf_filepath, out_dir)
        elif how_good == 2:
            return pdf_to_md_gpt4o(pdf_filepath, out_dir)
        elif how_good == 3:
            return pdf_to_md_azure_doc_intel(pdf_filepath, out_dir)
        elif how_good == 4:
            return pdf_to_md_llama_parse(pdf_filepath, out_dir)
        elif how_good == 5:
            return pdf_to_md_pymupdf(pdf_filepath, out_dir)
        elif how_good == 6:
            return pdf_to_md_pdf_miner(pdf_filepath, out_dir)
