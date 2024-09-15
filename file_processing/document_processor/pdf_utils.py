import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import fitz
import tiktoken

from file_processing.llm_chat_support import model_abstract

"""Sample:
{
'format': 'PDF 1.4',
'title': 'A Comparative Survey of Text Summarization Techniques',
'author': 'Patcharapruek Watanangura ',
'subject': 'SN Computer Science, https://doi.org/10.1007/s42979-023-02343-6',
'keywords': 'Artificial intelligence; Natural language processing; Text summarization',
'creator': 'Springer',
'producer': 'Acrobat Distiller 10.1.8 (Windows); modified using iText® 5.3.5 ©2000-2012 1T3XT BVBA (SPRINGER SBM; licensed version)',
'creationDate': "D:20231122123931+05'30'",
'modDate': "D:20231202142021+01'00'",
'trapped': '',
'encryption': None
}"""


class PDFMetadata:
    def __init__(self,
                 format: Optional[str] = None,
                 title: Optional[str] = None,
                 author: Optional[str] = None,
                 subject: Optional[str] = None,
                 keywords: Optional[str] = None,
                 creator: Optional[str] = None,
                 producer: Optional[str] = None,
                 creationDate: Optional[str] = None,
                 modDate: Optional[str] = None,
                 trapped: Optional[str] = None,
                 encryption: Optional[str] = None):
        self.format = format or ''
        self.title = title or ''
        self.author = author or ''
        self.subject = subject or ''
        self.keywords = keywords or ''
        self.creator = creator or ''
        self.producer = producer or ''
        self.creationDate = self.decode_pdf_date(creationDate) if creationDate else None
        self.modDate = self.decode_pdf_date(modDate) if modDate else None
        self.trapped = trapped or ''
        self.encryption = encryption or ''

        # Additional attributes
        self.fileName: Optional[str] = None
        self.addedDate: Optional[datetime] = None
        self.numPages: Optional[int] = None
        self.pageDimensions: Optional[Dict] = None
        self.fileSize: Optional[int] = None
        self.avgWordsPerPage: Optional[float] = None
        self.wordCount: Optional[int] = None

    def decode_pdf_date(self, date_str: str) -> datetime:
        if date_str.startswith('D:'):
            date_str = date_str[2:]

        date_patterns = [
            "%Y%m%d%H%M%S%z",  # D:YYYYMMDDHHmmSS+HH'mm'
            "%Y%m%d%H%M%S",  # D:YYYYMMDDHHmmSS
            "%Y%m%d%H%M",  # D:YYYYMMDDHHmm
            "%Y%m%d%H",  # D:YYYYMMDDHH
            "%Y%m%d",  # D:YYYYMMDD
            "%Y%m",  # D:YYYYMM
            "%Y",  # D:YYYY
        ]

        # Replace timezone offset format 'OHH'mm' with 'OHHmm' for strptime compatibility
        date_str = re.sub(r"([+-]\d{2})'(\d{2})'", r"\1\2", date_str)

        for pattern in date_patterns:
            try:
                if pattern.endswith('%z'):
                    dt = datetime.strptime(date_str, pattern)
                else:
                    dt = datetime.strptime(date_str, pattern)
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue

        return datetime.now()

    @classmethod
    def from_dict(cls, metadata: Dict[str, Any]) -> 'PDFMetadata':
        return cls(
            format=metadata.get('format'),
            title=metadata.get('title'),
            author=metadata.get('author'),
            subject=metadata.get('subject'),
            keywords=metadata.get('keywords'),
            creator=metadata.get('creator'),
            producer=metadata.get('producer'),
            creationDate=metadata.get('creationDate'),
            modDate=metadata.get('modDate'),
            trapped=metadata.get('trapped'),
            encryption=metadata.get('encryption')
        )

    @classmethod
    def from_pymupdf(cls, file_name: str, file_path: str) -> 'PDFMetadata':
        document = fitz.open(file_path)
        metadata = document.metadata
        file_info = {
            'format': metadata.get('format', ''),
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creationDate': metadata.get('creationDate', ''),
            'modDate': metadata.get('modDate', ''),
            'trapped': metadata.get('trapped', ''),
            'encryption': None if not document.is_encrypted else 'Encrypted'
        }
        instance = cls.from_dict(file_info)
        instance.fileName = file_name
        instance.addedDate = datetime.now()
        instance.numPages = document.page_count
        first_page = document.load_page(0)
        instance.pageDimensions = {
            'width': first_page.rect.width,
            'height': first_page.rect.height,
            'measure': "pt"
        }
        instance.fileSize = os.path.getsize(file_path)

        # Calculate word count and average words per page
        word_count = 0
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text = page.get_text("text")
            word_count += len(text.split())

        instance.wordCount = word_count
        instance.avgWordsPerPage = word_count / document.page_count if document.page_count > 0 else 0

        return instance

    def to_dict(self) -> Dict[str, Any]:
        def datetime_to_millis(dt: Optional[datetime]) -> Optional[int]:
            return int(dt.timestamp() * 1000) if dt else None

        return {
            'format': self.format,
            'title': self.title,
            'author': self.author,
            'subject': self.subject,
            'keywords': self.keywords,
            'creator': self.creator,
            'producer': self.producer,
            'creationDate': datetime_to_millis(self.creationDate),
            'modDate': datetime_to_millis(self.modDate),
            'trapped': self.trapped,
            'encryption': self.encryption,
            'fileName': self.fileName,
            'addedDate': datetime_to_millis(self.addedDate),
            'numPages': self.numPages,
            'pageDimensions': self.pageDimensions,
            'fileSize': self.fileSize,
            'avgWordsPerPage': self.avgWordsPerPage,
            'wordCount': self.wordCount
        }

    def __str__(self) -> str:
        creation_date_str = self.creationDate.strftime('%Y-%m-%d %H:%M:%S') if self.creationDate else 'N/A'
        mod_date_str = self.modDate.strftime('%Y-%m-%d %H:%M:%S') if self.modDate else 'N/A'
        added_date_str = self.addedDate.strftime('%Y-%m-%d %H:%M:%S') if self.addedDate else 'N/A'
        return (
            f"PDF Metadata:\n"
            f"Format: {self.format}\n"
            f"Title: {self.title}\n"
            f"Author: {self.author}\n"
            f"Subject: {self.subject}\n"
            f"Keywords: {self.keywords}\n"
            f"Creator: {self.creator}\n"
            f"Producer: {self.producer}\n"
            f"Creation Date: {creation_date_str}\n"
            f"Modification Date: {mod_date_str}\n"
            f"Trapped: {self.trapped}\n"
            f"Encryption: {self.encryption}\n"
            f"File Name: {self.fileName or 'N/A'}\n"
            f"Added Date: {added_date_str}\n"
            f"Number of Pages: {self.numPages if self.numPages is not None else 'N/A'}\n"
            f"Page Dimensions: {self.pageDimensions or 'N/A'}\n"
            f"File Size: {self.fileSize if self.fileSize is not None else 'N/A'}\n"
            f"Average Words per Page: {self.avgWordsPerPage if self.avgWordsPerPage is not None else 'N/A'}\n"
            f"Word Count: {self.wordCount if self.wordCount is not None else 'N/A'}\n"
        )
    @staticmethod
    def merge_json(obj1, obj2):
        def get_new_key(existing_keys, base_key):
            """
            Get the next available key when there are conflicts.
            """
            i = 1
            new_key = f"{base_key}_{i}"
            while new_key in existing_keys:
                i += 1
                new_key = f"{base_key}_{i}"
            return new_key

        """
        Recursively merges obj2 into obj1. Concats strings, adds numbers, extends lists,
        merges dictionaries and handles sets.
        """

        if not isinstance(obj1, dict) or not isinstance(obj2, dict):
            raise ValueError("Both need to be dictionaries.")

        for key, value in obj2.items():
            if key in obj1:
                if isinstance(value, dict) and isinstance(obj1[key], dict):
                    PDFMetadata.merge_json(obj1[key], value)
                elif isinstance(value, str) and isinstance(obj1[key], str):
                    obj1[key] += value
                elif isinstance(value, (int, float)) and isinstance(obj1[key], (int, float)):
                    obj1[key] += value
                elif isinstance(value, list) and isinstance(obj1[key], list):
                    obj1[key].extend(value)
                elif isinstance(value, set) and isinstance(obj1[key], set):
                    obj1[key].update(value)
                elif isinstance(value, tuple) and isinstance(obj1[key], tuple):
                    obj1[key] = obj1[key] + value
                elif value is None:
                    continue
                else:
                    # Handle conflicting types by adding the new value under a new key
                    new_key = get_new_key(obj1.keys(), key)
                    obj1[new_key] = value
            else:
                obj1[key] = value
        return obj1

    @staticmethod
    def extract_from_text(semantic_chapters: List[str]):
        """ Idea:
        1. Get max context window for specific LLM
        2. Concat chapters up to context window (need tokeniser or rough statistic with leeway)
        3. Prompt for json metadata and extract like basic rule extractor
        """
        context_window = 16000  #tokens
        encoding = tiktoken.encoding_for_model('gpt-4o')
        semantic_chapter_tokens = [encoding.encode(chapter) for chapter in semantic_chapters]
        # Add all semantic_chapter_tokens up to context_window
        token_splits: List[str] = []
        token_count = 0
        current_split = []
        for token in semantic_chapter_tokens:
            if token_count + len(token) > context_window:
                token_splits.append(encoding.decode(current_split))
                current_split = []
                token_count = 0
            current_split.extend(token)
            token_count += len(token)
        token_splits.append(encoding.decode(current_split))
        # Extract metadata from each split
        metadata = {}
        json_llm = model_abstract.bind(response_format={"type": "json_object"})
        for split in token_splits:
            ai_msg = json_llm.invoke(
                f"Given the following chapter:\n{split}\n\n"
                "Return a JSON object with describing the most important characteristics of the chapter as metadata in form:\n"
                "{ [short key describing the important characteristics]: \"key value\", ... }\n"
                "When preparing the metadata json take inspiration from Industry 4.0 and metadata for manual of a digital twin!\n"
                "Focus on product specifications and descriptions of processes. You can ignore irrelevant info!"
            )

            try:
                json_object = json.loads(ai_msg.content)
                # Data is written to metadata
                PDFMetadata.merge_json(metadata, json_object)
            except ValueError as e:
                pass  # invalid json

        return metadata


def get_filename_from_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    # Extract the path from the parsed URL
    path = parsed_url.path
    # Get the filename from the path
    filename = os.path.basename(path)
    return filename


class SplitPDFOutput:
    split_pdf_path: str
    from_original_start_page: int
    from_original_end_page: int
    screenshots_per_page: List[str]

    def __init__(self, split_pdf_path: str, from_original_start_page: int, from_original_end_page: int,
                 screenshots_per_page: List[str]):
        super().__init__()
        self.split_pdf_path = split_pdf_path
        self.from_original_start_page = from_original_start_page
        self.from_original_end_page = from_original_end_page
        self.screenshots_per_page = screenshots_per_page


def split_pdf(input_pdf_path: str,
              output_folder: str,
              pages_per_file: int = 1,
              page_screenshots: bool = False,
              only_screenshots: bool = False) -> List[SplitPDFOutput]:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(input_pdf_path)
    split_files = []

    for start_page in range(0, len(pdf_document), pages_per_file):
        pdf_writer = fitz.open()
        end_page = min(start_page + pages_per_file, len(pdf_document))
        page_screenshots_list = []
        output_pdf_path = ""

        if not only_screenshots:
            pdf_writer.insert_pdf(pdf_document, from_page=start_page, to_page=end_page - 1)

            output_pdf_path = f"{output_folder}/pages_{start_page + 1}_to_{end_page}.pdf"
            pdf_writer.save(output_pdf_path)
        if page_screenshots:
            # Save each page to the same folder
            for page_index in range(start_page, end_page):
                page = pdf_document.load_page(page_index)

                original_size = page.mediabox
                min_size = 900
                max_size = 1800

                if max(original_size.width, original_size.height) > max_size:
                    # Zoom to bring down to max_size
                    zoom = max_size / max(original_size.width, original_size.height)
                elif min(original_size.width, original_size.height) < min_size:
                    # Zoom to bring up to min_size
                    zoom = min_size / min(original_size.width, original_size.height)
                else:
                    zoom = 1

                # Define the transformation matrix for zoom
                mat = fitz.Matrix(zoom, zoom)

                # Render the page to a pixmap (image)
                pix = page.get_pixmap(matrix=mat)
                screenshot_output_path = f'{output_folder}/page_{page_index + 1}.png'
                pix.save(screenshot_output_path)
                page_screenshots_list.append(screenshot_output_path)
        pdf_writer.close()
        split_files.append(SplitPDFOutput(output_pdf_path, start_page + 1, end_page, page_screenshots_list))
    print(
        f"PDF split into {len(split_files)} files with up to {pages_per_file} pages each{' + page screenshots' if page_screenshots else ''} and saved in {output_folder}.")
    pdf_document.close()

    return split_files

if __name__ == '__main__':
    import argparse
    import tempfile
    import uuid

    parser = argparse.ArgumentParser(description='PDF Split')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp()
    out_dir = os.path.join(temp_dir, f'{uuid.uuid4()}')

    print(split_pdf(args.input_pdf, output_folder=out_dir, page_screenshots_list=True))