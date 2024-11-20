import sys
import os
import tempfile
import json
from typing import List

import pymupdf4llm


def pdf_2_md_chunkable(pdf_filepath, page_chunks):
    try:
        if page_chunks:
            """Type of output given page_chunks=True:
            [{
                "metadata": metadata,
                "toc_items": page_tocs,
                "tables": tables,
                "images": images,
                "graphics": graphics,
                "text": page_output,
            }]
            """
            # noinspection PyTypeChecker
            md_text: List[dict] = pymupdf4llm.to_markdown(pdf_filepath, page_chunks=page_chunks)
            md_text: List[str] = [page.get("text", "") for page in md_text]
        else:
            md_text: str = pymupdf4llm.to_markdown(pdf_filepath, page_chunks=page_chunks)
            md_text: List[str] = [md_text]

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        sys.exit(1)

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
        json.dump(md_text, temp_file)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python pdf_2_md_chunkable_internal.py <pdf_filepath> <page_chunks: true/false>")
        sys.exit(1)

    pdf_filepath = sys.argv[1]
    page_chunks_input = sys.argv[2].lower()

    if not os.path.isfile(pdf_filepath):
        print(f"Error: File '{pdf_filepath}' not found.")
        sys.exit(1)

    if page_chunks_input not in ["true", "false"]:
        print(f"Error: Invalid value for page_chunks. Please provide 'true' or 'false'.")
        sys.exit(1)

    page_chunks = page_chunks_input == "true"

    chunks_filepath = pdf_2_md_chunkable(pdf_filepath, page_chunks)

    print(f"PDF content has been successfully extracted and saved to: {chunks_filepath}")


if __name__ == "__main__":
    main()
