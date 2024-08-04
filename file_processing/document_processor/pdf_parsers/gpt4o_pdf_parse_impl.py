import logging
import os
import tempfile
import uuid

from gptpdf import parse_pdf


def pdf_to_md_gpt4o(pdf_filepath: str) -> str | None:
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

    content, image_paths = parse_pdf(
        pdf_path=pdf_filepath,
        output_dir=os.path.join(temp_dir, f'{uuid.uuid4()}'),
        model=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
        prompt=prompt,
        verbose=False,
        gpt_worker=3,
        api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
        base_url=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    )
    return content

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure GPT 4o')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_gpt4o(args.input_pdf))