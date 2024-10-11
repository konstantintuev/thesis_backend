import os
import re
import uuid
from typing import Dict, Union, List

import html2text
import markdown
from bs4 import BeautifulSoup

from file_processing.document_processor.basic_text_processing_utils import concat_chunks
from file_processing.document_processor.types_local import ListItem, TableItem, ExtractedItemHtml, UUIDExtractedItemDict

"""The general idea is:
 1. Extract useful text data like tables and lists from Markdown
 2. Preliminary chunk based on main headings.
"""


def markdown_to_html(md_content):
    # Convert markdown to HTML
    return markdown.markdown(md_content)


def extract_code_blocks(code_block_language: str, markdown_text: str) -> List[str]:
    # Define the regex pattern to find code blocks of the specified language
    pattern = rf'```{code_block_language}(.*?)```'
    # Find all matches in the markdown text
    code_blocks = re.findall(pattern, markdown_text, re.DOTALL)
    if (code_blocks is None or len(code_blocks) == 0) and code_block_language == 'markdown':
        return [markdown_text]
    return code_blocks


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


def extract_and_replace_html_tables(html_content) -> (str, dict):
    soup = BeautifulSoup(html_content, 'html.parser')
    items_dict = {}

    # Find all tables
    for table_tag in soup.find_all('table'):
        table_html = str(table_tag)
        table_uuid = str(uuid.uuid4())
        table_length = len(table_html)

        table_tag.replace_with(f"{table_uuid}\n")

        items_dict[table_uuid] = {"type": "table", "content": table_html, "length": table_length}

    return str(soup), items_dict


def extract_and_replace_tables(md_content) -> (str, UUIDExtractedItemDict):
    table_pattern = re.compile(r'(\|(?:[^\n]*\|)+\n(?:\|[-: ]*\|[-|: ]*\n)?(?:\|(?:[^\n]*\|)+\n)+)', re.MULTILINE)
    items_dict = {}

    def replace_with_uuid(match):
        table = match.group(0)
        table_uuid = str(uuid.uuid4())
        table_length = len(table)
        items_dict[table_uuid] = {"type": "table", "content": table, "length": table_length}
        return f"{table_uuid}\n"

    modified_content = re.sub(table_pattern, replace_with_uuid, md_content)

    return modified_content, items_dict


def extract_and_replace_multiblock_math(md_content) -> (str, UUIDExtractedItemDict):
    # Extract multiline latex math in markdown
    multiline_latex_pattern = re.compile(
        r"(?s)\$\$(.*?)\$\$"
    )

    items_dict = {}

    def replace_with_uuid(match):
        math = match.group(0)
        math_uuid = str(uuid.uuid4())
        math_length = len(math)
        items_dict[math_uuid] = {"type": "math", "content": math, "length": math_length}
        return f"{math_uuid}\n"

    modified_content = re.sub(multiline_latex_pattern, replace_with_uuid, md_content)

    return modified_content, items_dict


def html_to_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

# Custom Markdown splitter as MarkdownHeaderTextSplitter omits recurring headlines
def split_markdown_by_header(markdown: str,
                             max_length: int,
                             header_level: int = 1) -> list:

    # We only want to split up to ###
    if header_level > 3:
        return [markdown]


    # Note: Regex will include the matched header in the result
    if header_level == 1:
        # Match `#` or `===` for level 1 headers
        header_pattern = r'(?=\n*(?:\s*^#\s+|^.+\n[=]{3,}\s*))'
    elif header_level == 2:
        # Match `##` or `---` for level 2 headers
        header_pattern = r'(?=\n*(?:\s*^##\s+|^.+\n[-]{3,}\s*))'
    else:
        # Match `###` for level 3 headers
        header_pattern = r'(?=\n*(?:\s*^###\s+))'

    sections = re.split(header_pattern, markdown, flags=re.MULTILINE)

    sections = [section.strip() for section in sections if section.strip()]

    split_sections = []
    for section in sections:
        # If the section is too large -> recursively split with the next header level
        if len(section) > max_length and header_level < 3:
            split_sections.extend(split_markdown_by_header(section, max_length, header_level + 1))
        else:
            split_sections.append(section)

    return split_sections

def semantic_markdown_chunks(md_content: str, headers_to_split_on: list,
                             min_length: int = int(os.environ.get("MIN_CHUNK_LENGTH", "1500")),
                             max_length: int = int(os.environ.get("MAX_CHUNK_LENGTH", "3000"))) -> (
        list[str], UUIDExtractedItemDict):

    parser = html2text.HTML2Text()
    parser.unicode_snob = True
    chunks = split_markdown_by_header(md_content, max_length)
    final_chunks = concat_chunks(chunks, min_length, max_length)
    items_dict = {}
    chunks_w_attachable_content = []
    for chunk in final_chunks:
        modified_chunk = chunk + "\n"

        # Extract and replace tables first
        modified_chunk, tables_dict = extract_and_replace_tables(modified_chunk)

        # Extract and replace math second
        modified_chunk, math_dict = extract_and_replace_multiblock_math(modified_chunk)

        # Convert modified markdown to HTML for list processing
        html_chunk = markdown_to_html(modified_chunk)
        modified_html_chunk, lists_dict = extract_and_replace_lists(html_chunk)
        modified_html_chunk, html_tables_dict = extract_and_replace_html_tables(modified_html_chunk)

        # Merge tables and lists into one dictionary
        items_dict = {**items_dict, **tables_dict, **html_tables_dict, **lists_dict, **math_dict}

        plain_text = parser.handle(modified_html_chunk)
        chunks_w_attachable_content.append(plain_text)

    return chunks_w_attachable_content, items_dict


def test():
    with open("../../raptor/demo/manual.md", 'r', encoding='utf-8') as file:
        md_content = file.read()
    headers_to_split_on = [
        ("#", "Header 1")
    ]

    html_header_splits = semantic_markdown_chunks(md_content, headers_to_split_on, 300)

    for chunk in html_header_splits:
        plain_text = html_to_plain_text(chunk)
        print("ok")

    with open("../../raptor/demo/manual_chunked.txt", 'w', encoding='utf-8') as file:
        file.write("\n\n\n".join(
            [f"Chunk {index + 1}:\n{html_to_plain_text(chunk)}" for index, chunk in enumerate(html_header_splits)]))
    print("OK")


if __name__ == "__main__":
    test()
