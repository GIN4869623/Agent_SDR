"""Agent layer: LangGraph orchestration, routing, context and tracing.

The public ``AgentLoop`` API remains model-agnostic. Internal graph nodes and
external MCP calls share the protocol-independent ``ToolRegistry``.
"""
