import asyncio
import json
import os
import re
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import aiofiles
import magic
import nest_asyncio
import requests
from asgiref.sync import async_to_sync, sync_to_async
# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.chains.query_constructor.schema import AttributeInfo

from file_processing.document_processor.advanced_filters import ask_file_llm

from file_processing.document_processor.colbert_utils_pylate import colbert_local, add_uuid_object_to_string
from file_processing.document_processor.md_parser import semantic_markdown_chunks
from file_processing.document_processor.pdf_parsers import pdf_to_md_by_type
from file_processing.document_processor.pdf_utils import PDFMetadata, get_filename_from_url
from file_processing.document_processor.raptor_utils import custom_config, tree_to_dict
from file_processing.document_processor.semantic_text_splitter import uuid_pattern
from file_processing.document_processor.summarisation_utils import chunk_into_semantic_chapters
from file_processing.embeddings import pending_embeddings_singleton
from file_processing.file_queue_management.file_queue_db import get_file_from_queue, set_file_status, \
    add_multiple_files_to_queue, get_multiple_files_queue
from file_processing.query_processor.basic_rule_extractor import query_to_structured_filter
from file_processing.query_processor.process_search_query import rewrite_search_query_based_on_history
from file_processing.query_processor.rerankers_local import rerankers_instance
from raptor.raptor import RetrievalAugmentation

nest_asyncio.apply()


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


temp_dir = tempfile.mkdtemp()

# Function to run async function in a separate thread
def run_async_task(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def create_chunk(index, chunk, pending_embeddings, uuid_items):
    out = chunk_into_semantic_chapters(pending_embeddings, chunk, uuid_items)
    return out  # (index, out)

async def pdf_to_chunks_task(file_uuid: uuid, file_name: str, temp_pdf_received: str,
                             file_mime_type: str, file_processor: str):
    # with open("raptor/demo/sample_response.json", 'r', encoding='utf-8') as test_file:
    #   test_json_content = test_file.read()
    # return HttpResponse(test_json_content, content_type="application/json")
    ret = RetrievalAugmentation(
        config=custom_config)  # , tree="/Users/konstantintuev/Projects/Thesis/thesis_backend/raptor/demo/random")
    # tree_dict = tree_to_dict(ret.tree)

    # Convert the dictionary to JSON
    # tree_json = json.dumps(tree_dict, indent=4)

    # return HttpResponse(tree_json, content_type="application/json")

    md_content = (pdf_to_md_by_type(temp_pdf_received, file_processor, file_uuid=f'{file_uuid}')
                  .get_best_text_content())
    headers_to_split_on = [
        ("#", "Header 1")
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

    # Tables, lists and math are valuable content for summarisation
    # ... BUT adding them f ups the raptor tree
    # TODO: decide what to do!, unused for now
    semantic_chapters_w_attachable_content = [re.sub(uuid_pattern,
                                                     lambda match: add_uuid_object_to_string(match, uuid_items),
                                                     chapter)
                                              for chapter in semantic_chapters]
    # Extract document information
    pdf_metadata = PDFMetadata.from_pymupdf(file_name, temp_pdf_received)
    semantic_metadata = PDFMetadata.extract_from_text(semantic_chapters_w_attachable_content)
    total_metadata = {}
    total_metadata.update({
        "file_metadata": pdf_metadata.to_dict(),
        "semantic_metadata": semantic_metadata
    })
    ret.add_semantic_chapters(semantic_chapters)
    # SAVE_PATH = "../raptor/demo/random"
    # ret.save(SAVE_PATH)

    tree_dict = tree_to_dict(ret.tree)

    out = {
        "tree": tree_dict,
        "metadata": total_metadata,
        "uuid_items": uuid_items,
        "file_uuid": str(file_uuid)
    }
    # Convert the dictionary to JSON
    out_json = json.dumps(out, indent=4)

    # if colbert corrupts it's index, we can reacreate it
    await sync_to_async(set_file_status)(str(file_uuid), 'preliminary', out_json)

    # TODO: Enable colbert again when it gets more stable
    await sync_to_async(colbert_local.add_documents_to_index)([out])

    queue_id = await sync_to_async(set_file_status)(str(file_uuid), 'done', out_json)
    # delete temp_pdf_received if exists
    if os.path.exists(temp_pdf_received):
        os.remove(temp_pdf_received)
    print(f"Done: {file_uuid}={queue_id}")

@csrf_exempt
@async_to_sync
async def files_to_chunks(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        upload_urls = request.POST.getlist('fileURLs')
        file_ids = request.POST.getlist('fileIDs')
        file_processor = request.POST.get('fileProcessor')

        if not uploaded_files and not upload_urls:
            return HttpResponse("No files uploaded")

        for_queue = []
        def handle_file(file_uuid, file_name, temp_pdf_received, file_processor):
            file_mime_type = magic.from_file(temp_pdf_received, mime=True)

            if file_mime_type != 'application/pdf':
                # TODO: accept more files
                return HttpResponse("File is not a PDF")

            # Schedule the long-running async task to run in the thread
            thread = threading.Thread(target=run_async_task,
                                      args=[pdf_to_chunks_task(file_uuid, file_name, temp_pdf_received,
                                                               file_mime_type, file_processor)])
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
                handle_file(file_uuid, file_name, temp_pdf_received, file_processor)
        elif uploaded_files:
            for file in uploaded_files:
                file_uuid = uuid.uuid4()
                temp_pdf_received = os.path.join(temp_dir, f'{file_uuid}.pdf')
                async with aiofiles.open(temp_pdf_received, 'wb') as f:
                    for chunk in file.chunks():
                        await f.write(chunk)
                handle_file(file_uuid, file.name, temp_pdf_received, file_processor)

        queue_id = await sync_to_async(add_multiple_files_to_queue)(for_queue)
        return JsonResponse({"multiple_file_queue_id": queue_id})
    else:
        return HttpResponse('Invalid request')


@csrf_exempt
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


@csrf_exempt
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

            all_embeddings = [pending_embeddings_singleton.embed_query(input_text) for input_text in input_texts]
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


@csrf_exempt
def search_query(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query_text = data.get('query', None)
            high_level_summary = data.get('high_level_summary', None)
            unique_file_ids = data.get('unique_file_ids', None)
            source_count = data.get('source_count', None)
            if query_text is None:
                return JsonResponse({"error": "Query not provided!"}, status=400)

            res = colbert_local.search_colbert_index(query_text,
                                                    high_level_summary,
                                                    unique_file_ids,
                                                    source_count)
            return JsonResponse(res,
                                status=(200 if isinstance(res, list) or res["error"] is None else 400),
                                # 'Safe' serialises only dicts and we have a list here
                                safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def get_available_processors(request):
    if request.method == 'GET':
        return JsonResponse(
            [
                {
                    "processorId": "pdf_to_md_azure_doc_gpt4o",
                    "processorName": "Azure Document Intelligence + GPT-4o",
                    "provider": "Azure",
                    "fileTypesSupported": ["pdf"]
                },
                {
                    "processorId": "pdf_to_md_gpt4o",
                    "processorName": "GPT-4o",
                    "provider": "Azure",
                    "fileTypesSupported": ["pdf"]
                },
                {
                    "processorId": "pdf_to_md_azure_doc_intel",
                    "processorName": "Azure Doc Intel",
                    "provider": "Azure",
                    "fileTypesSupported": ["pdf"]
                },
                {
                    "processorId": "pdf_to_md_llama_parse",
                    "processorName": "LLaMA Parse",
                    "provider": "LLaMA",
                    "fileTypesSupported": ["pdf"]
                },
                {
                    "processorId": "pdf_to_md_pymupdf",
                    "processorName": "PyMuPDF",
                    "provider": "PyMuPDF",
                    "fileTypesSupported": ["pdf"]
                },
                {
                    "processorId": "pdf_to_md_pdf_miner",
                    "processorName": "PDFMiner",
                    "provider": "PDFMiner",
                    "fileTypesSupported": ["pdf"]
                }
            ],
            # 'Safe' serialises only dicts and we have a list here
            safe=False
        )
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def text_2_query(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            attribute_info = [AttributeInfo(
                name=item.get('name'),
                description=item.get('description'),
                type=item.get('type'),
            ) for item in data.get('attributes', [])]
            query_text = data.get('query', None)
            document_content_description = data.get('files_description', None)
            if not query_text or not document_content_description or len(attribute_info) == 0:
                return JsonResponse({"error": "Bad inputs"}, status=400)
            res = query_to_structured_filter(
                query_text,
                document_content_description,
                attribute_info
            )
            return JsonResponse(res,
                                status=200,
                                # 'Safe' serialises only dicts and we should have a list here
                                safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def rewrite_query(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query_text = data.get('query', None)
            previous_queries = data.get('previous_queries', None)
            if query_text is None or previous_queries is None:
                return JsonResponse({"error": "Queries not provided!"}, status=400)

            res = rewrite_search_query_based_on_history(query_text, previous_queries)
            return HttpResponse(res,
                                status=200)
        except json.JSONDecodeError:
            return HttpResponse("Invalid JSON", status=400)
        except Exception as e:
            return HttpResponse(str(e), status=500)
    else:
        return HttpResponse("Invalid request method", status=405)

@csrf_exempt
def ask_file(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', None)
            file_sections = data.get('file_sections', None)
            if query is None or file_sections is None:
                return JsonResponse({"error": "Query or file sections not provided!"}, status=400)

            res = ask_file_llm(query, file_sections)
            return JsonResponse(res.dict(),
                                status=200,
                                # 'Safe' serialises only dicts and we should have a list here
                                safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def rerank_results(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query_text: str = data.get('query', None)
            res: List[dict] = data.get('res', None)
            reorder: bool = data.get('reorder', False)
            if query_text is None or res is None:
                return JsonResponse({"error": "Query or res not provided!"}, status=400)

            # We either reduce 8 chunks to 4 (RAG) OR 150 to 100 (file search)
            if len(res) <= 8:
                res = rerankers_instance.do_llama_rerank(query=query_text, res=res, reorder=reorder)
            else:
                res = rerankers_instance.do_colbert_rerank(query=query_text, res=res, reorder=reorder)

            return JsonResponse(res,
                                status=(200 if isinstance(res, list) else 400),
                                safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except BaseException as e:
            print('An exception occurred: {}'.format(e))
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)