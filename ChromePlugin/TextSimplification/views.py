import inspect
import sys
import os

SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from django.shortcuts import render
import json
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse
from hebrew_ts.hebrew_text_simplification import text_simplification_pipeline

def index(request):
    """
    Renders the HTML page for the Chrome plugin popup.
    """
    return render(request, "HTML/popup.html")

def get_simplified(request):
    """
    Handles the AJAX request for text simplification.
    Extracts the necessary parameters from the request and returns the simplified text as a JSON response.
    """
    text = request.GET.get('text', None)
    simplification_checkbox = True if request.GET.get('simplificationCheckbox', False) in ['True', 'true', True] else False
    summarization_checkbox = True if request.GET.get('summarizationCheckbox', False) in ['True', 'true', True] else False

    simple = text_simplification_pipeline(text, word_sub=simplification_checkbox, sentence_filter=summarization_checkbox)

    data = {
        'simple_text': simple,
        'raw': 'Successful',
    }

    return JsonResponse(data)