"""Run original generation unchanged, importing the isolated repaired tree parser."""
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / 'scripts'
FIXED = HERE.parent / 'evaluation-v2-20261003/scripts'

def main():
    # Fixed parser and ontology take precedence; model_utils remains original.
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(FIXED))
    runpy.run_path(str(SCRIPTS / 'predict.py'), run_name='__main__')

if __name__ == '__main__':
    main()
