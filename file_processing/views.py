import json
import tempfile
import uuid

from asgiref.sync import async_to_sync
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import nest_asyncio
from django.views.decorators.csrf import csrf_exempt
import os
import aiofiles

from file_processing.utils import RA, tree_to_dict, custom_config
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
        ret = RetrievalAugmentation(config=custom_config, tree="/Users/konstantintuev/Projects/Thesis/thesis_backend/raptor/demo/random")
        tree_dict = tree_to_dict(ret.tree)

        # Convert the dictionary to JSON
        tree_json = json.dumps(tree_dict, indent=4)

        return HttpResponse(tree_json, content_type="application/json")
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

        RA.add_documents(documents_text)
        SAVE_PATH = "../raptor/demo/random"
        RA.save(SAVE_PATH)

        tree_dict = tree_to_dict(RA.tree)

        # Convert the dictionary to JSON
        tree_json = json.dumps(tree_dict, indent=4)

        return HttpResponse(tree_json, content_type="application/json")
    else:
        return HttpResponse('Invalid request')