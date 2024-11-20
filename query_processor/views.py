import json
import tempfile

import nest_asyncio
# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.chains.query_constructor.schema import AttributeInfo

from file_processing.document_processor.advanced_filters import ask_file_llm
from query_processor.basic_rule_extractor import query_to_structured_filter
from query_processor.process_search_query import rewrite_search_query_based_on_history

nest_asyncio.apply()


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


temp_dir = tempfile.mkdtemp()


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
