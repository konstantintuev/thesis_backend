import asyncio
import datetime
import json
import tempfile
import threading
import uuid
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fitz  #PyMuPDF
import magic
import requests
from asgiref.sync import async_to_sync, sync_to_async
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
import nest_asyncio
from django.views.decorators.csrf import csrf_exempt
import os
import aiofiles

from file_processing.document_processor.basic_text_processing_utils import concat_chunks
from file_processing.document_processor.embeddings import PendingLangchainEmbeddings, embeddings_model, \
    pending_embeddings_singleton
from file_processing.document_processor.md_parser import semantic_markdown_chunks, html_to_plain_text
from file_processing.document_processor.pdf_utils import PDFMetadata, llama_parser, get_filename_from_url
from file_processing.document_processor.summarisation_utils import chunk_into_semantic_chapters
from file_processing.document_processor.raptor_utils import custom_config, tree_to_dict
from file_processing.file_queue_management.file_queue_db import add_file_to_queue, get_file_from_queue, set_file_status, \
    add_multiple_files_to_queue, get_multiple_files_queue
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


# Function to run async function in a separate thread
def run_async_task(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def create_chunk(index, chunk, pending_embeddings, uuid_items):
    plain_text = html_to_plain_text(chunk)
    out = chunk_into_semantic_chapters(pending_embeddings, plain_text, uuid_items)
    return out  # (index, out)

async def pdf_to_chunks_task(file_uuid: uuid, file_name: str, temp_pdf_received: str,
                             file_mime_type: str):
    # with open("raptor/demo/sample_response.json", 'r', encoding='utf-8') as test_file:
    #   test_json_content = test_file.read()
    # return HttpResponse(test_json_content, content_type="application/json")
    ret = RetrievalAugmentation(
        config=custom_config)  # , tree="/Users/konstantintuev/Projects/Thesis/thesis_backend/raptor/demo/random")
    # tree_dict = tree_to_dict(ret.tree)

    # Convert the dictionary to JSON
    # tree_json = json.dumps(tree_dict, indent=4)

    # return HttpResponse(tree_json, content_type="application/json")
    documents = await llama_parser.aload_data(temp_pdf_received)
    """with open("raptor/demo/cs_paper.md", 'r', encoding='utf-8') as test_file:
        test_file_content = test_file.read()

    class PDFDocument:
        def __init__(self, text):
           self.text = text
    documents = [
      PDFDocument(test_file_content)
    ]"""
    if documents is None or len(documents) == 0:
        return HttpResponse('Invalid request')
    document = documents[0]

    # Extract document information
    pdf_metadata = PDFMetadata.from_pymupdf(file_name, temp_pdf_received)

    md_content = document.text
    headers_to_split_on = [
        ("h1", "Header 1"),
        # ("h2", "Header 2"),
        # ("h3", "Header 3"),
    ]

    semantic_chapters: List[str] = []
    html_header_splits, uuid_items = semantic_markdown_chunks(
        md_content,
        headers_to_split_on,
        int(os.environ.get("MIN_CHUNK_LENGTH")))

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as executor:
        # Wrap the blocking `executor.submit` call in `loop.run_in_executor`
        tasks = [
            loop.run_in_executor(executor, create_chunk, index, chunk, pending_embeddings_singleton, uuid_items)
            for index, chunk in enumerate(html_header_splits)
        ]
        # List is lists divided by markdown chunks
        semantic_chapters_raw = await asyncio.gather(*tasks)
        semantic_chapters = [chapter for sublist in semantic_chapters_raw for chapter in sublist]

    semantic_chapters_concat = concat_chunks(semantic_chapters, int(os.environ.get("MIN_CHUNK_LENGTH")),
                                             int(os.environ.get("MAX_CHUNK_LENGTH")))
    ret.add_semantic_chapters(semantic_chapters_concat)
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

    queue_id = await sync_to_async(set_file_status)(str(file_uuid), 'done', out_json)
    # delete temp_pdf_received if exists
    if os.path.exists(temp_pdf_received):
        os.remove(temp_pdf_received)
    print(f"Done: {file_uuid}={queue_id}")


@csrf_exempt
@async_to_sync
async def pdf_to_chunks(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('file')

        if not uploaded_files:
            return HttpResponse("No files uploaded")
        file = uploaded_files[0]

        file_uuid = uuid.uuid4()
        temp_pdf_received = os.path.join(temp_dir, f'{file_uuid}.pdf')
        async with aiofiles.open(temp_pdf_received, 'wb') as f:
            for chunk in file.chunks():
                await f.write(chunk)
        file_mime_type = magic.from_file(temp_pdf_received, mime=True)

        if file_mime_type != 'application/pdf':
            # TODO: accept more files
            return HttpResponse("File is not a PDF")

        # Schedule the long-running async task to run in the thread
        thread = threading.Thread(target=run_async_task,
                                  args=[pdf_to_chunks_task(file_uuid, file.name, temp_pdf_received,
                                                                 file_mime_type)])
        thread.start()

        queue_id = await sync_to_async(add_file_to_queue)(str(file_uuid), temp_pdf_received, file_mime_type)

        return JsonResponse({"queue_id": queue_id})
    else:
        return HttpResponse('Invalid request')


@csrf_exempt
@async_to_sync
async def files_to_chunks(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        upload_urls = request.POST.getlist('fileURLs')
        file_ids = request.POST.getlist('fileIDs')

        if not uploaded_files and not upload_urls:
            return HttpResponse("No files uploaded")

        for_queue = []
        def handle_file(file_uuid, file_name, temp_pdf_received, ):
            file_mime_type = magic.from_file(temp_pdf_received, mime=True)

            if file_mime_type != 'application/pdf':
                # TODO: accept more files
                return HttpResponse("File is not a PDF")

            # Schedule the long-running async task to run in the thread
            thread = threading.Thread(target=run_async_task,
                                      args=[pdf_to_chunks_task(file_uuid, file_name, temp_pdf_received,
                                                               file_mime_type)])
            thread.start()

            for_queue.append({"file_uuid": str(file_uuid), "file_path": temp_pdf_received, "mime_type": file_mime_type})

        # TODO: limit files loaded in memory (66 page pdf is 9MBs of processed data and 2 MBs of pdf -> 0,1667 MB per Page)
        # -> 10 GBs of ram is good for 60_000 pages of pdfs or 600 pdfs with 100 pages
        if upload_urls:
            for i in range(0, len(upload_urls)):
                url = upload_urls[i]
                file_uuid = file_ids[i]
                file_name = get_filename_from_url(url)
                temp_pdf_received = os.path.join(temp_dir, f'{file_uuid}.pdf')
                response = requests.get(url, stream=True)
                with open(temp_pdf_received, 'wb') as output:
                    output.write(response.content)
                handle_file(file_uuid, file_name, temp_pdf_received)
        elif uploaded_files:
            for file in uploaded_files:
                file_uuid = uuid.uuid4()
                temp_pdf_received = os.path.join(temp_dir, f'{file_uuid}.pdf')
                async with aiofiles.open(temp_pdf_received, 'wb') as f:
                    for chunk in file.chunks():
                        await f.write(chunk)
                handle_file(file_uuid, file.name, temp_pdf_received)

        queue_id = await sync_to_async(add_multiple_files_to_queue)(for_queue)
        return JsonResponse({"multiple_file_queue_id": queue_id})
    else:
        return HttpResponse('Invalid request')

def retrieve_file_from_queue(request):
    if request.method == 'GET':
        file_uuid = request.GET.get('file_uuid')
        file_data = get_file_from_queue(file_uuid)
        if file_data:
            return JsonResponse(file_data)
        else:
            return JsonResponse({"error": "File not found"}, status=404)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

def retrieve_multiple_files_from_queue(request):
    if request.method == 'GET':
        multiple_files_uuid = request.GET.get('multiple_files_uuid')
        file_data = get_multiple_files_queue(multiple_files_uuid)
        if file_data:
            return JsonResponse(file_data)
        else:
            return JsonResponse({"error": "File not found"}, status=404)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
@async_to_sync
async def generate_embeddings(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_texts = data.get('input', [])
            model = data.get('model', 'default-model')

            if isinstance(input_texts, str):
                input_texts = [input_texts]

            all_embeddings = [embeddings_model.embed_query(input_text) for input_text in input_texts]
            embeddings = [
                {
                    "object": "embedding",
                    "embedding": all_embeddings[idx],
                    "index": idx
                }
                for idx, text in enumerate(input_texts)
            ]

            response = {
                "object": "list",
                "data": embeddings,
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in input_texts),
                    "total_tokens": sum(len(text.split()) for text in input_texts)
                }
            }

            return JsonResponse(response)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
