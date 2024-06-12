import asyncio
import datetime
import json
import tempfile
import uuid
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fitz  #PyMuPDF
import magic
from asgiref.sync import async_to_sync
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import nest_asyncio
from django.views.decorators.csrf import csrf_exempt
import os
import aiofiles

from file_processing.document_processor.embeddings import PendingLangchainEmbeddings, embeddings_model
from file_processing.document_processor.md_parser import semantic_markdown_chunks, html_to_plain_text
from file_processing.document_processor.pdf_utils import PDFMetadata, llama_parser
from file_processing.document_processor.summarisation_utils import chunk_into_semantic_chapters
from file_processing.document_processor.raptor_utils import custom_config, tree_to_dict
from raptor.raptor import RetrievalAugmentation

nest_asyncio.apply()

from llama_parse import LlamaParse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


temp_dir = tempfile.mkdtemp()


@csrf_exempt
@async_to_sync
async def upload_pdf(request):
    if request.method == 'POST':
        file = request.FILES['file']
        temp_pdf_received = os.path.join(temp_dir, f'{uuid.uuid4()}.pdf')
        async with aiofiles.open(temp_pdf_received, 'wb') as f:
            for chunk in file.chunks():
                await f.write(chunk)
        parser = LlamaParse(
            # can also be set in your env as LLAMA_CLOUD_API_KEY
            result_type="markdown",  # "markdown" and "text" are available
            num_workers=6,  # if multiple files passed, split in `num_workers` API calls
            verbose=True,
            language="en",  # Optionally you can define a language, default=en
        )

        documents = await parser.aload_data(temp_pdf_received)
        documents_text = ' '.join([doc.text for doc in documents])
        print(documents_text)
        return HttpResponse(documents_text, content_type="text/markdown")
    else:
        return HttpResponse('Invalid request')


@csrf_exempt
@async_to_sync
async def pdf_to_chunks(request):
    if request.method == 'POST':
        #with open("raptor/demo/sample_response.json", 'r', encoding='utf-8') as test_file:
         #   test_json_content = test_file.read()
        #return HttpResponse(test_json_content, content_type="application/json")
        ret = RetrievalAugmentation(
            config=custom_config)  #, tree="/Users/konstantintuev/Projects/Thesis/thesis_backend/raptor/demo/random")
        #tree_dict = tree_to_dict(ret.tree)

        # Convert the dictionary to JSON
        #tree_json = json.dumps(tree_dict, indent=4)

        #return HttpResponse(tree_json, content_type="application/json")
        uploaded_files = request.FILES.getlist('file')

        if not uploaded_files:
            return HttpResponse("No files uploaded")
        file = uploaded_files[0]

        temp_pdf_received = os.path.join(temp_dir, f'{uuid.uuid4()}.pdf')
        async with aiofiles.open(temp_pdf_received, 'wb') as f:
            for chunk in file.chunks():
                await f.write(chunk)
        file_mime_type = magic.from_file(temp_pdf_received, mime=True)

        if file_mime_type != 'application/pdf':
            # TODO: accept more files
            return HttpResponse("File is not a PDF")

        documents = await llama_parser.aload_data(temp_pdf_received)
        #with open("raptor/demo/cs_paper.md", 'r', encoding='utf-8') as test_file:
         #   test_file_content = test_file.read()

          #  class PDFDocument:
           #     def __init__(self, text):
            #        self.text = text
        #documents = [
         #   PDFDocument(test_file_content)
        #]
        if documents is None or len(documents) == 0:
            return HttpResponse('Invalid request')
        document = documents[0]

        # Extract document information
        pdf_metadata = PDFMetadata.from_pymupdf(file.name, temp_pdf_received)

        md_content = document.text
        headers_to_split_on = [
            ("h1", "Header 1"),
            # ("h2", "Header 2"),
            # ("h3", "Header 3"),
        ]

        semantic_chapters: List[str] = []
        html_header_splits, uuid_items = semantic_markdown_chunks(md_content, headers_to_split_on)
        pending_embeddings = PendingLangchainEmbeddings(embeddings_model)

        def create_chunk(index, chunk):
            plain_text = html_to_plain_text(chunk.page_content)
            out = chunk_into_semantic_chapters(pending_embeddings, plain_text, uuid_items)
            return out  #(index, out)

        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor() as executor:
            # Wrap the blocking `executor.submit` call in `loop.run_in_executor`
            tasks = [
                loop.run_in_executor(executor, create_chunk, index, chunk)
                for index, chunk in enumerate(html_header_splits)
            ]
            # List is lists divided by markdown chunks
            semantic_chapters_raw = await asyncio.gather(*tasks)
            semantic_chapters = [chapter for sublist in semantic_chapters_raw for chapter in sublist]

        ret.add_semantic_chapters(semantic_chapters)
        # SAVE_PATH = "../raptor/demo/random"
        # ret.save(SAVE_PATH)

        tree_dict = tree_to_dict(ret.tree)

        out = {
            "tree": tree_dict,
            "metadata": pdf_metadata.to_dict(),
            "uuid_items": uuid_items
        }

        # Convert the dictionary to JSON
        out_json = json.dumps(out, indent=4)

        return HttpResponse(out_json, content_type="application/json")
    else:
        return HttpResponse('Invalid request')
