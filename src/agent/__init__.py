"""Agent layer — the core agentic loop + context assembly.

Claude Code pattern:
    Context assembly → LLM call with tools → tool execution → observe → repeat

The loop is model-agnostic (any OpenAI-compatible API) and MCP-centric
(all tools come from MCPToolRegistry).
"""
