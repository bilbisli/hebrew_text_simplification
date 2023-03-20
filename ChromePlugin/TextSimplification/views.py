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
    return HttpResponse("Hello, world. You're at the wiki index.")


# https://pypi.org/project/wikipedia/#description
def get_simplified(request):
    topic = request.GET.get('topic', None)

    print('topic:', topic)
    summ = text_simplification_pipeline(topic)
    data = {
        'summary': summ,
        'raw': 'Successful',
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)