# TODO: add from the thesis_prototyping repo
import json
import os
import subprocess
from typing import List

from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument, PdfToMdPageInfo
from file_processing.document_processor.pdf_utils import split_pdf


# Mostly from https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
def pdf_to_md_pdf_miner(pdf_filepath: str, output_dir: str) -> PdfToMdDocument:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_pdf_paths = split_pdf(pdf_filepath,
                                output_dir,
                                1,
                                True)
    pages_out = PdfToMdDocument()
    for split_pdf_path in split_pdf_paths:
        loader = PDFMinerPDFasHTMLLoader(split_pdf_path.split_pdf_path)

        data = loader.load()[0]  # entire PDF is loaded as a single Document
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data.page_content, 'html.parser')
        content = soup.find_all('div')

        import re
        cur_fs = None
        cur_text = ''
        snippets = []  # first collect all snippets that have the same font size
        for c in content:
            sp = c.find('span')
            if not sp:
                continue
            st = sp.get('style')
            if not st:
                continue
            fs = re.findall('font-size:(\d+)px', st)
            if not fs:
                continue
            fs = int(fs[0])
            if not cur_fs:
                cur_fs = fs
            if fs == cur_fs:
                cur_text += c.text
            else:
                snippets.append((cur_text, cur_fs))
                cur_fs = fs
                cur_text = c.text
        snippets.append((cur_text, cur_fs))
        # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
        # headers/footers in a PDF appear on multiple pages so if we find duplicates it's safe to assume that it is redundant info)

        from langchain_community.docstore.document import Document
        cur_idx = -1
        semantic_snippets = []
        # Assumption: headings have higher font size than their respective content
        for s in snippets:
            # if current snippet's font size > previous section's heading => it is a new heading
            if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
                metadata = {'heading': s[0], 'content_font': 0, 'heading_font': s[1]}
                metadata.update(data.metadata)
                semantic_snippets.append(Document(page_content='', metadata=metadata))
                cur_idx += 1
                continue

            # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
            # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
            if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata[
                'content_font']:
                semantic_snippets[cur_idx].page_content += s[0]
                semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata[
                    'content_font'])
                continue

            # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
            # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
            metadata = {'heading': s[0], 'content_font': 0, 'heading_font': s[1]}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='', metadata=metadata))
            cur_idx += 1
        new_line = "\n"

        raw_page_md_content = "\n\n".join(
            [f"# {chunk.metadata.get('heading').replace(new_line, ' ')}\n\n{chunk.page_content}" for chunk in
             semantic_snippets])

        pages_out.append(PdfToMdPageInfo(
            split_pdf_path.from_original_start_page,
            raw_page_md_content,
            "",
            "",
            split_pdf_path.screenshots_per_page[0]
        ))
    return pages_out


def pdf_to_md_pymupdf(pdf_filepath: str, output_dir: str) -> PdfToMdDocument:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_screenshots = split_pdf(
        pdf_filepath, output_dir, 1, True, True
    )
    bash_script = os.environ.get("PATH_TO_PDF_PARSER", None)
    bash_script_path = os.path.abspath(os.path.expanduser(os.path.expandvars(bash_script)))

    parent_dir = os.path.dirname(bash_script_path)
    if bash_script is None:
        return PdfToMdDocument()

    if not os.path.isfile(bash_script):
        print(f"Error: Bash script '{bash_script}' not found.")
        return PdfToMdDocument()

    command = [bash_script, pdf_filepath, "true"]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                cwd=parent_dir)

        stdout = result.stdout
        print("Output from the Bash script:")
        print(stdout)

        output_file = ""
        for line in stdout.splitlines():
            if "PDF content has been successfully extracted and saved to:" in line:
                output_file = line.split(": ")[-1].strip()
                print(f"Extracted content saved to: {output_file}")
                break

        if output_file is not None and len(output_file) > 0:
            with open(output_file, 'r', encoding='utf-8') as file:
                content = file.read()
                return PdfToMdDocument([PdfToMdPageInfo(
                    index + 1,
                    page,
                    "",
                    "",
                    pdf_screenshots[index].screenshots_per_page[0]
                ) for index, page in enumerate(json.loads(content))])

        stderr = result.stderr
        if stderr:
            print("Errors from the Bash script:")
            print(stderr)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the Bash script: {e}")
        print(f"Error output: {e.stderr}")
        return PdfToMdDocument()

    return PdfToMdDocument()


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Open Source Tools')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    parser.add_argument("--tool_name", type=str, default='pdf_miner')
    args = parser.parse_args()
    if (args.tool_name == 'pdf_to_md_pdf_miner'):
        print(pdf_to_md_pdf_miner(args.input_pdf))
    elif (args.tool_name == 'pdf_to_md_pymupdf'):
        print(pdf_to_md_pymupdf(args.input_pdf))
