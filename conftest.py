"""
Root conftest.py — ensures the project root is on sys.path so that
``from src.xxx import yyy`` works regardless of how pytest is invoked.
"""
import sys
import pathlib

# Insert project root at the front of sys.path
_ROOT = str(pathlib.Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
