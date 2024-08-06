from file_processing.document_processor.pdf_parsers.azure_doc_intel_impl import pdf_to_md_azure_doc_intel
from file_processing.document_processor.pdf_parsers.gpt4o_pdf_parse_impl import pdf_to_md_gpt4o
from file_processing.document_processor.pdf_parsers.llama_parse_impl import pdf_to_md_llama_parse
from file_processing.document_processor.pdf_parsers.open_source_local_parsers import pdf_to_md_pymupdf, \
    pdf_to_md_pdf_miner


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