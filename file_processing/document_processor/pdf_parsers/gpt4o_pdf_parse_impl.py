import base64
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from file_processing.document_processor.md_parser import extract_code_blocks
from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument, PdfToMdPageInfo
from file_processing.document_processor.pdf_utils import split_pdf
from file_processing.llm_chat_support import LLMTemp, get_llm

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

import os


def create_prompt_for_page(image_data: str, prompt: dict):
    system_message = SystemMessage(content=prompt.get("role_prompt"))
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": prompt.get("prompt")},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ],
    )
    return ChatPromptTemplate.from_messages([system_message, human_message])


def pdf_to_md_gpt4o(pdf_filepath: str, output_dir: str, n: int = 8) -> PdfToMdDocument:
    # Prompt translated from gptpdf generally
    prompt = {
        "prompt": (
            "Use markdown syntax to convert the text recognised in the image to markdown format for output. You must do:\n"
            "    1. Output the same language as the one in which the recognised image is used, e.g. for fields recognised in English, the output must be in English.\n"
            "    2. Don't interpret the text which is not related to the output, and output the content in the image directly. For example, it is strictly forbidden to output examples like ``Here is the markdown text I generated based on the content of the image:`` Instead, you should output markdown directly.\n"
            "    3. Content should not be included in ```markdown ```, paragraph formulas should be in the form of $$ $$, in-line formulas should be in the form of $ $$, long straight lines should be ignored, and page numbers should be ignored.\n"
            "    4. Complex tables should be included in html format.\n"
            "    Again, don't interpret text that has nothing to do with the output, and output the content directly from the image.\n"),
        "rect_prompt": "The picture has areas marked out with red boxes and names (%s). If the regions are tables or pictures, use the ! []() form to insert it into the output, otherwise output the text content directly.",
        "role_prompt": "You are a PDF document parser that outputs the content of images using markdown and latex syntax."
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_pdf_paths = split_pdf(pdf_filepath,
                                output_dir,
                                1,
                                True,
                                False)
    pages_out = PdfToMdDocument()

    # Batch process with up to n parallel tasks at a time
    for i in range(0, len(split_pdf_paths), n):
        batch = split_pdf_paths[i:i + n]
        chains = {}
        for idx, split_pdf_path in enumerate(batch):
            with open(split_pdf_path.screenshots_per_page[0], "rb") as screenshot:
                image_data = base64.b64encode(screenshot.read()).decode("utf-8")
            prompt_template = create_prompt_for_page(image_data, prompt)
            chains[f"page_{i + idx}"] = prompt_template | get_llm(LLMTemp.CONCRETE)

        # Run this batch in parallel
        map_chain = RunnableParallel(**chains)
        results = map_chain.invoke({})

        # Extract and process the markdown content from each result
        for idx, result in results.items():
            parsed_md_content = extract_code_blocks("markdown", result.content)
            pages_out.append(PdfToMdPageInfo(
                split_pdf_paths[int(idx.split("_")[1])].from_original_start_page,
                parsed_md_content[0] if parsed_md_content else "",
                "",
                "",
                split_pdf_paths[int(idx.split("_")[1])].screenshots_per_page[0]
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
