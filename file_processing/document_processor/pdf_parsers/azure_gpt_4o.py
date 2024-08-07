# Idea: page by page processing of pdf with Azure Document Intelligence + correction with GPT 4o

""" Steps:
1. Split into PDFs - 1 page per pdf
2. Extract text from each PDF with Document Intelligence or something
3. Fix each PDF markdown with GPT 4o
4. Write metadata
4. Concat markdowns
"""

import logging

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

from file_processing.document_processor.md_parser import extract_code_blocks
from file_processing.document_processor.pdf_parsers import PdfToMdPageInfo

import base64
import os
import tempfile
import time
import uuid
from threading import Lock

import os
import re
from typing import List, Tuple, Optional, Dict
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

from file_processing.document_processor.pdf_utils import split_pdf


class AzureDocAndGptTPS:
    model = AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
        azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
        openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
    )

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

    def fix_document_intelligence_small_prompt(self, intial_md: str):
        return """You are given a screenshot of a page from a document and an initial markdown conversion of that page. Your task is to correct the markdown, ensuring all math equations, matrices, and tables are formatted accurately.
        
        **Instructions:**
        1. Analyze the provided screenshot and the initial markdown text.
        2. Correct any inaccuracies in math equations, matrices, and tables in the markdown text, referencing the screenshot as needed.
        3. Try to stay as close to the original text and tables as possible, but make necessary corrections for accuracy.
        4. Infer any figures, illustrations, or graphs from the screenshot and add indicators with full text descriptions in place of each, using the format: `<visual type="{figure, illustration, or graph}">{description}</visual>`.
        5. Remove any irrelevant text from the markdown (e.g., page headers, footers, logos, images [](), 'click here', 'Listen to this article', page numbers).
        6. Return only the corrected markdown and a JSON log detailing any changes, corrections, or additions.
        
        **Output:**
        - **Corrected Markdown:**
          - Use a single ```markdown``` code block for the corrected markdown text.
          - Ensure all math equations are in the form of $ $$ for inline and $$ $$ for block equations.
          - Ensure all matrices and tables are properly formatted and aligned.
        
        - **JSON Log:**
          - Use a single ```json``` code block for the change log.
          - Structure: `{"changed":[{"what": string, "why": string}], "corrected":[{"what": string, "why": string}], "added": [{"what": string, "why": string}]}`
          - Detail any changes made to the markdown text.
        
        **Initial Markdown:**
        ```
        """+intial_md+"""
        ```
        
        **Example:**
        
        *Initial Markdown:*
        ```
        Equation:
        
        E = 
        m
        c
        2
        
        Matrix:
        
        B = ( 5 
        0 
        
        0, 0 3 
        0, 0 
        0 1)
        ```
        
        *Screenshot Analysis:*
        - Equation is incorrect, missing proper formatting.
        - Matrix is incorrectly formatted.
        - Figure present in the screenshot but not in markdown.
        
        *Corrected Markdown:*
        ```markdown
        Equation:
        
        $$ E = mc^2 $$
        
        Matrix:
        
        $$
        B = \begin{pmatrix}
        5 & 0 & 0 \\
        0 & 3 & 0 \\
        0 & 0 & 1
        \end{pmatrix}
        $$
        
        <visual type="figure">Figure 1: Diagram showing the steps to assemble the device.</visual>
        ```
        
        *JSON Log:*
        ```json
        {
          "changed": [
            {"what": "Equation formatting", "why": "Corrected to proper LaTeX format"}
          ],
          "corrected": [
            {"what": "Matrix formatting", "why": "Ensured correct representation of the matrix"}
          ],
          "added": [
            {"what": "Figure 1 description", "why": "Added visual indicator for figure inferred from screenshot"}
          ]
        }
        ```
    
    **Your task is to follow these instructions precisely, ensuring accurate and clean markdown output along with a detailed JSON log.**
    """

    def pdf_to_md_azure_doc_intel_pdfs(self, pdf_filepath: str, output_dir: str) -> List[PdfToMdPageInfo]:
        self._wait_if_needed()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        split_pdf_paths = split_pdf(pdf_filepath,
                                    output_dir,
                                    1,
                                    True)
        pages_out = []
        analysis_features = ["ocrHighResolution", "formulas"]
        for split_pdf_path in split_pdf_paths:
            # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTEL_ENDPOINT"),
                api_key=os.environ.get("AZURE_DOC_INTEL_API_KEY"),
                file_path=split_pdf_path.split_pdf_path,
                api_model="prebuilt-layout",
                mode="markdown",
                analysis_features=analysis_features,
            )

            raw_page_md_content = ""
            documents = loader.load()
            for document in documents:
                raw_page_md_content += document.page_content + "\n\n"
            # Just the first screenshot as we split page by page
            with open(split_pdf_path.screenshots_per_page[0], "rb") as screenshot:
                image_data = base64.b64encode(screenshot.read()).decode("utf-8")
            system = SystemMessage(
                content="You are a PDF document parser that outputs the content of images using markdown and latex syntax.")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.fix_document_intelligence_small_prompt(raw_page_md_content)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            response = self.model.invoke([system, message])

            parsed_md_content = extract_code_blocks("markdown", response.content)
            parsed_json_log = extract_code_blocks("json", response.content)
            pages_out.append(PdfToMdPageInfo(
                split_pdf_path.from_original_start_page,
                raw_page_md_content,
                parsed_md_content[0],
                parsed_json_log[0],
                split_pdf_path.screenshots_per_page[0]
            ))
        # TODO: decide what to do with the pages
        return pages_out


azure_doc_and_gpt_impl = AzureDocAndGptTPS()


def pdf_to_md_azure_doc_intel(pdf_filepath: str) -> str | None:
    temp_dir = tempfile.mkdtemp()
    out_dir = os.path.join(temp_dir, f'{uuid.uuid4()}')
    res = azure_doc_and_gpt_impl.pdf_to_md_azure_doc_intel_pdfs(pdf_filepath, out_dir)
    return "\n\n".join([it.fixed_md_content for it in res])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure Document Intelligence + GPT 4o')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_azure_doc_intel(args.input_pdf))