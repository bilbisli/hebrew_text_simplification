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

    print('text:', text)
    simple = text_simplification_pipeline(text)
    data = {
        'simple_text': simple,
        'raw': 'Successful',
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)