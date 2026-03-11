"""Shared pytest configuration — must run before any test module imports tools.py."""

import os
import tempfile

# Redirect MEMORY_DB to a temp file for ALL tests.
# This MUST happen before tools.py is imported anywhere, because DB_PATH is
# read at module-import time. conftest.py is imported first by pytest, so
# placing the override here guarantees isolation from the production DB.
if "MEMORY_DB" not in os.environ or "local-memory-agent/memory.db" in os.environ.get("MEMORY_DB", ""):
    _tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    _tmp.close()
    os.environ["MEMORY_DB"] = _tmp.name
