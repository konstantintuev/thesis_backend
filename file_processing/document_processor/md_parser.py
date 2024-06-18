import os
import re
import uuid
from typing import Dict, Union

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
import markdown
from bs4 import BeautifulSoup

from file_processing.document_processor.basic_text_processing_utils import concat_chunks
from file_processing.document_processor.types import ListItem, TableItem, ExtractedItemHtml, UUIDExtractedItemDict

"""The general idea is:
 1. Extract useful text data like tables and lists from Markdown
 2. Preliminary chunk based on main headings.
"""

def markdown_to_html(md_content):
    # Convert markdown to HTML
    return markdown.markdown(md_content)


def extract_and_replace_lists(html_content) -> (str, UUIDExtractedItemDict):
    soup = BeautifulSoup(html_content, 'html.parser')
    items_dict = {}

    # Find all unordered and ordered lists
    for list_tag in soup.find_all(['ul', 'ol']):
        list_items = list_tag.find_all('li')
        list_type = list_tag.name
        children = [item.get_text() for item in list_items]

        if len(children) == 1:
            # If the list has only one item, replace it with the item text
            single_item_text = children[0]
            list_tag.replace_with(single_item_text)
        else:
            # Otherwise, generate a unique identifier and replace the list with it
            list_uuid = str(uuid.uuid4())
            list_content = list_tag.get_text(separator='\n')
            list_length = len(list_content)
            items_dict[list_uuid] = {"type": list_type, "children": children, "length": list_length}
            list_tag.replace_with(f"{list_uuid}\n")

    return str(soup), items_dict


def extract_and_replace_tables(md_content) -> (str, UUIDExtractedItemDict):
    table_pattern = re.compile(r'(\|.*\|.*\n(\|[-:]*\|[-|:]*\n)?(\|.*\|.*\n)+)')
    items_dict = {}

    def replace_with_uuid(match):
        table = match.group(0)
        table_uuid = str(uuid.uuid4())
        table_length = len(table)
        items_dict[table_uuid] = {"type": "table", "content": table, "length": table_length}
        return f"{table_uuid}\n"

    modified_content = re.sub(table_pattern, replace_with_uuid, md_content)

    return modified_content, items_dict


def html_to_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def semantic_markdown_chunks(md_content: str, headers_to_split_on: list, min_length: int = int(os.environ.get("MIN_CHUNK_LENGTH"))) -> (list[str], UUIDExtractedItemDict):
    # Extract and replace tables first
    modified_content, tables_dict = extract_and_replace_tables(md_content)

    # Convert modified markdown to HTML for list processing
    html_content = markdown_to_html(modified_content)
    modified_html, lists_dict = extract_and_replace_lists(html_content)

    # Merge tables and lists into one dictionary
    items_dict = {**tables_dict, **lists_dict}

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text(modified_html)
    chunks = [chunk.page_content for chunk in html_header_splits]
    # Merging small chunks measured by character count
    final_chunks = concat_chunks(chunks, min_length, max_length=None)
    return final_chunks, items_dict

def test():
    with open("../../raptor/demo/manual.md", 'r', encoding='utf-8') as file:
        md_content = file.read()
    headers_to_split_on = [
        ("h1", "Header 1"),
        #("h2", "Header 2"),
        #("h3", "Header 3"),
    ]

    html_header_splits = semantic_markdown_chunks(md_content, headers_to_split_on, 300)

    for chunk in html_header_splits:
        plain_text = html_to_plain_text(chunk)
        print("ok")

    with open("../../raptor/demo/manual_chunked.txt", 'w', encoding='utf-8') as file:
        file.write("\n\n\n".join([f"Chunk {index+1}:\n{html_to_plain_text(chunk)}" for index, chunk in enumerate(html_header_splits)]))
    print("OK")

if __name__ == "__main__":
    test()