import os
import uuid

from file_processing.document_processor.pdf_parsers.azure_doc_intel_impl import pdf_to_md_azure_doc_intel
from file_processing.document_processor.pdf_parsers.azure_gpt_4o import pdf_to_md_azure_doc_gpt4o
from file_processing.document_processor.pdf_parsers.gpt4o_pdf_parse_impl import pdf_to_md_gpt4o
from file_processing.document_processor.pdf_parsers.llama_parse_impl import pdf_to_md_llama_parse
from file_processing.document_processor.pdf_parsers.open_source_local_parsers import pdf_to_md_pymupdf, \
    pdf_to_md_pdf_miner
from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument
from file_processing.storage_manager import global_temp_dir

def pdf_to_md_by_quality(pdf_filepath: str,
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


def pdf_to_md_by_type(pdf_filepath: str,
                      file_processor: str,
                      file_uuid: str = f'{uuid.uuid4()}') -> PdfToMdDocument:
    out_dir = os.path.join(global_temp_dir, file_uuid)

    if file_processor == "pdf_to_md_azure_doc_gpt4o":
        return pdf_to_md_azure_doc_gpt4o(pdf_filepath, out_dir)
    elif file_processor == "pdf_to_md_gpt4o":
        return pdf_to_md_gpt4o(pdf_filepath, out_dir)
    elif file_processor == "pdf_to_md_azure_doc_intel":
        return pdf_to_md_azure_doc_intel(pdf_filepath, out_dir)
    elif file_processor == "pdf_to_md_llama_parse":
        return pdf_to_md_llama_parse(pdf_filepath, out_dir)
    elif file_processor == "pdf_to_md_pymupdf":
        return pdf_to_md_pymupdf(pdf_filepath, out_dir)
    elif file_processor == "pdf_to_md_pdf_miner":
        return pdf_to_md_pdf_miner(pdf_filepath, out_dir)
