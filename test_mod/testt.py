import inspect
import sys
import os

SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hebrew_ts.hebrew_ts import text_simplification_pipeline

print(text_simplification_pipeline('בדיקה'))
