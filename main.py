"""Backward-compatible entry point — delegates to src/server.py.

Usage (unchanged):
    uvicorn main:app --host 127.0.0.1 --port 8000

This file is kept for compatibility.  The new entry point is src/server.py.
"""

from __future__ import annotations

# Re-export the FastAPI app from the new src/ module.
# All API endpoints, MCP mount, and static files are defined in src/server.py.
from src.server import app  # noqa: F401
