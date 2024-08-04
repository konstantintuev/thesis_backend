# TODO: add from the thesis_prototyping repo
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
import pymupdf4llm


# Mostly from https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
def pdf_to_md_pdf_miner(pdf_filepath: str) -> str | None:
    loader = PDFMinerPDFasHTMLLoader(pdf_filepath)

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
    return "\n\n".join([f"# {chunk.metadata.get('heading').replace(new_line, ' ')}\n\n{chunk.page_content}" for chunk in
                        semantic_snippets])
def pdf_to_md_pymupdf(pdf_filepath: str) -> str | None:
    md_text = pymupdf4llm.to_markdown(pdf_filepath)
    return md_text

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
