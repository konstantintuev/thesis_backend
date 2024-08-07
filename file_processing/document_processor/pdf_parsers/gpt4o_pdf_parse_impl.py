import base64
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from file_processing.document_processor.md_parser import extract_code_blocks
from file_processing.document_processor.pdf_utils import split_pdf

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)


import os
import tempfile
import uuid
from typing import List

from langchain_openai import AzureChatOpenAI

from file_processing.document_processor.pdf_parsers import PdfToMdPageInfo

model = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
)

def pdf_to_md_gpt4o(pdf_filepath: str) -> List[PdfToMdPageInfo]:
    # Prompt translated from gptpdf generally
    prompt = {
        "prompt": (
            "Use markdown syntax to convert the text recognised in the image to markdown format for output. You must do:\n"
            "    1. output the same language as the one in which the recognised image is used, e.g. for fields recognised in English, the output must be in English.\n"
            "    2. don't interpret the text which is not related to the output, and output the content in the image directly. For example, it is strictly forbidden to output examples like ``Here is the markdown text I generated based on the content of the image:`` Instead, you should output markdown directly.\n"
            "    3. Content should not be included in ```markdown ```, paragraph formulas should be in the form of $$ $$, in-line formulas should be in the form of $ $$, long straight lines should be ignored, and page numbers should be ignored.\n"
            "    Again, don't interpret text that has nothing to do with the output, and output the content directly from the image.\n"),
        "rect_prompt": "The picture has areas marked out with red boxes and names (%s). If the regions are tables or pictures, use the ! []() form to insert it into the output, otherwise output the text content directly.",
        "role_prompt": "You are a PDF document parser that outputs the content of images using markdown and latex syntax."
    }

    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, f'{uuid.uuid4()}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_pdf_paths = split_pdf(pdf_filepath,
                                output_dir,
                                1,
                                True,
                                False)
    pages_out = []
    for split_pdf_path in split_pdf_paths:
        # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
        # Just the first screenshot as we split page by page
        with open(split_pdf_path.screenshots_per_page[0], "rb") as screenshot:
            image_data = base64.b64encode(screenshot.read()).decode("utf-8")
        system = SystemMessage(
            content=prompt.get("role_prompt"))
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt.get("prompt")},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        response = model.invoke([system, message])

        parsed_md_content = extract_code_blocks("markdown", response.content)
        pages_out.append(PdfToMdPageInfo(
            split_pdf_path.from_original_start_page,
            "",
            parsed_md_content[0],
            "",
            split_pdf_path.screenshots_per_page[0]
        ))
    return pages_out

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure GPT 4o')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_gpt4o(args.input_pdf))