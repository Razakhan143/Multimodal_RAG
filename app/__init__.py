"""Multimodal RAG backend package.

Windows consoles default to the cp1252 codec, which cannot encode the emoji
used in the backend's progress ``print`` statements and raises
``UnicodeEncodeError`` mid-pipeline. Reconfigure stdout/stderr to UTF-8 (with a
safe fallback) as soon as the package is imported so logging never crashes the
RAG flow.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Stream may not support reconfigure (e.g. already wrapped); ignore.
        pass
