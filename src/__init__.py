"""Agent SDR — Claude Code architecture.

Layers (top → bottom):
    Entry    — server.py (SSE), cli.py (stdio)
    Agent    — loop.py (agentic loop), context.py (prompt assembly)
    MCP      — server.py (tool registry, single source of truth)
    Skills   — registry.py (regex matching, prompt injection)
    Domain   — hardware/, core/rag.py, core/memory.py
"""
