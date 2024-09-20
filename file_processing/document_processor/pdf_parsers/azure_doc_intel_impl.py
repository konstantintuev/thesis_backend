import logging
import os

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from ratelimiter import RateLimiter

from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument, PdfToMdPageInfo
from file_processing.document_processor.pdf_utils import split_pdf


def pdf_to_md_azure_doc_intel_pdfs(pdf_filepath: str, output_dir: str) -> PdfToMdDocument:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_pdf_paths = split_pdf(pdf_filepath,
                                output_dir,
                                2,
                                True)
    pages_out = PdfToMdDocument()
    analysis_features = ["ocrHighResolution", "formulas"]
    for split_pdf_path in split_pdf_paths:
        tries = 0
        while not parse_this_pdf_split(analysis_features, pages_out, split_pdf_path):
            logging.error(f"Retry({os.path.basename(split_pdf_path.split_pdf_path)}: {tries}")
            tries += 1

    return pages_out


rate_limiter = RateLimiter(max_calls=1, period=2)

def parse_this_pdf_split(analysis_features, pages_out, split_pdf_path):
    with rate_limiter:
        try:
            # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTEL_ENDPOINT"),
                api_key=os.environ.get("AZURE_DOC_INTEL_API_KEY"),
                file_path=split_pdf_path.split_pdf_path,
                api_model="prebuilt-layout",
                mode="markdown",
                analysis_features=analysis_features
            )
            raw_page_md_content = ""
            documents = loader.load()
            for document in documents:
                raw_page_md_content += document.page_content + "\n\n"
            pages_out.append(PdfToMdPageInfo(
                split_pdf_path.from_original_start_page,
                raw_page_md_content,
                "",
                "",
                split_pdf_path.screenshots_per_page[0]
            ))
            return True
        except BaseException as e:
            logging.error(f"AzureAIDocumentIntelligenceLoader FAILED({os.path.basename(split_pdf_path.split_pdf_path)}): {e}")
            return False


def pdf_to_md_azure_doc_intel(pdf_filepath: str, out_dir: str) -> PdfToMdDocument:
    res = pdf_to_md_azure_doc_intel_pdfs(pdf_filepath, out_dir)
    return res


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure Document Intelligence')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_azure_doc_intel(args.input_pdf))