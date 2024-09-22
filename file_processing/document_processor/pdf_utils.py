import os
import re
from datetime import datetime
from io import StringIO
from typing import List
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import pdfplumber
import tiktoken
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjRef, resolve1
from pdfminer.psparser import PSLiteral
from pypdf import PdfWriter, PdfReader

from file_processing.document_processor.semantic_metadata import extract_semantic_metadata_together

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
    def from_pdfminer(cls, file_name: str, file_path: str) -> 'PDFMetadata':
        with open(file_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            metadata = document.info[0] if document.info else {}

            pdf_version = f"PDF {parser.doc.version}" if hasattr(parser, 'doc') and hasattr(parser.doc,
                                                                                            'version') else 'Unknown'
            encryption_status = 'Encrypted' if 'Encrypt' in document.catalog else None

            def extract_metadata_value(value):
                if isinstance(value, (bytes, str)):
                    return value.decode('utf-8') if isinstance(value, bytes) else value
                elif isinstance(value, PSLiteral):
                    return str(value.name)
                elif isinstance(value, PDFObjRef):
                    return str(resolve1(value))
                else:
                    return str(value)

            file_info = {
                'format': pdf_version,
                'title': extract_metadata_value(metadata.get('Title', '')),
                'author': extract_metadata_value(metadata.get('Author', '')),
                'subject': extract_metadata_value(metadata.get('Subject', '')),
                'keywords': extract_metadata_value(metadata.get('Keywords', '')),
                'creator': extract_metadata_value(metadata.get('Creator', '')),
                'producer': extract_metadata_value(metadata.get('Producer', '')),
                'creationDate': extract_metadata_value(metadata.get('CreationDate', '')),
                'modDate': extract_metadata_value(metadata.get('ModDate', '')),
                'trapped': extract_metadata_value(metadata.get('Trapped', '')),
                'encryption': encryption_status
            }

            num_pages = sum(1 for _ in PDFPage.create_pages(document))

            instance = cls.from_dict(file_info)
            instance.fileName = file_name
            instance.addedDate = datetime.now()
            instance.numPages = num_pages
            instance.fileSize = os.path.getsize(file_path)

            # Calculate word count and average words per page
            instance.wordCount, instance.avgWordsPerPage = cls.get_word_count(file_path, num_pages)

        return instance

    @staticmethod
    def get_word_count(file_path: str, num_pages: int):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        word_count = 0

        with open(file_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)

            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)
                text = retstr.getvalue()
                word_count += len(text.split())
                retstr.truncate(0)
                retstr.seek(0)

        device.close()
        retstr.close()

        avg_words_per_page = word_count / num_pages if num_pages > 0 else 0
        return word_count, avg_words_per_page

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
            'creation_date': datetime_to_millis(self.creationDate),
            'mod_date': datetime_to_millis(self.modDate),
            'trapped': self.trapped,
            'encryption': self.encryption,
            'file_name': self.fileName,
            'added_date': datetime_to_millis(self.addedDate),
            'num_pages': self.numPages,
            'page_dimensions': self.pageDimensions,
            'file_size': self.fileSize,
            'avg_words_per_page': self.avgWordsPerPage,
            'word_count': self.wordCount
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
        context_window = 16000  # tokens
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
        for split in token_splits:
            try:
                # Data is written to metadata
                PDFMetadata.merge_json(metadata, extract_semantic_metadata_together(split))
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

# Use pdfminer, PyPDF2, pdfplumber for better license than PyMuPDF
class SplitPDFOutput:
    def __init__(self, split_pdf_path: str, from_original_start_page: int, from_original_end_page: int,
                 screenshots_per_page: List[str]):
        self.split_pdf_path = split_pdf_path
        self.from_original_start_page = from_original_start_page
        self.from_original_end_page = from_original_end_page
        self.screenshots_per_page = screenshots_per_page


def save_page_screenshot(page, output_path, width):
    image = page.to_image(None, round(width))
    image_path = f"{output_path}.png"
    image.save(image_path)
    return image_path


def split_pdf(input_pdf_path: str,
              output_folder: str,
              pages_per_file: int = 1,
              page_screenshots: bool = False,
              only_screenshots: bool = False) -> List[SplitPDFOutput]:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    split_files = []

    pdf_reader = PdfReader(input_pdf_path)
    total_pages = len(pdf_reader.pages)

    pdf = pdfplumber.open(input_pdf_path) if page_screenshots or only_screenshots else None

    for start_page in range(0, total_pages, pages_per_file):
        end_page = min(start_page + pages_per_file, total_pages)
        output_pdf_path = f"{output_folder}/pages_{start_page + 1}_to_{end_page}.pdf"
        page_screenshots_list = []

        if page_screenshots or only_screenshots:
            for page_index in range(start_page, end_page):
                page = pdf.pages[page_index]
                screenshot_output_path = f'{output_folder}/page_{page_index + 1}'
                original_height = page.height
                original_width = page.width
                min_size = 900
                max_size = 1800

                if max(original_width, original_height) > max_size:
                    # Zoom to bring down to max_size
                    zoom = max_size / max(original_width, original_height)
                elif min(original_width, original_height) < min_size:
                    # Zoom to bring up to min_size
                    zoom = min_size / min(original_width, original_height)
                else:
                    zoom = 1

                screenshot_path = save_page_screenshot(page,
                                                       screenshot_output_path,
                                                       original_width * zoom)
                page_screenshots_list.append(screenshot_path)

        if not only_screenshots:
            pdf_writer = PdfWriter()
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            with open(output_pdf_path, 'wb') as output_pdf_file:
                pdf_writer.write(output_pdf_file)
        else:
            output_pdf_path = ""

        split_files.append(SplitPDFOutput(output_pdf_path, start_page + 1, end_page, page_screenshots_list))

    if pdf:
        pdf.close()

    print(
        f"PDF split into {len(split_files)} files with up to {pages_per_file} pages each{' + page screenshots' if page_screenshots else ''} and saved in {output_folder}.")
    return split_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PDF Metadata Extractor')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()

    metadata = PDFMetadata.from_pdfminer(os.path.basename(args.input_pdf), args.input_pdf)
    print(metadata)
