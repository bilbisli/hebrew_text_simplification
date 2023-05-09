import inspect
import sys
import os

SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from django.shortcuts import render
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####
from hebrew_ts.hebrew_ts import text_simplification_pipeline

import wikipedia

def index(request):
    return render(request, "HTML/popup.html")


# https://pypi.org/project/wikipedia/#description
def get_simplified(request):
    text = request.GET.get('text', None)
    simplification_checkbox = True if request.GET.get('simplificationCheckbox', False) in ['True','true', True] else False
    summarization_checkbox = True if request.GET.get('summarizationCheckbox', False) in ['True','true', True] else False

    simple = text_simplification_pipeline(text, word_sub=simplification_checkbox, sentence_filter=summarization_checkbox)
    
    data = {
        'simple_text': simple,
        'raw': 'Successful',
    }

    return JsonResponse(data)