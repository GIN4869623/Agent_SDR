"""Agent loop — the core of the Claude Code architecture.

Flow:
    1. Match skill by regex
    2. Build context (system prompt + skill injection + history + user msg)
    3. Loop:
        a. Call LLM with MCP tools, stream text deltas
        b. Collect tool calls from LLM response
        c. Execute each tool call via MCPToolRegistry
        d. Inject tool results into messages, continue
    4. Save conversation memory
    5. Yield final "done" event

All events are streamed as dicts for SSE transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)

from core.memory import ConversationMemory
from src.agent.context import build_messages
from src.config import settings
from src.mcp import MCPToolRegistry
from skill.skill_spec import SkillRegistry

# ═══════════════════════════════════════════════════════════════════════════
# Console event reporting config
# ═══════════════════════════════════════════════════════════════════════════

CONSOLE_URL: str = os.getenv("CONSOLE_EVENT_URL", "http://127.0.0.1:3000")
CONSOLE_TOKEN: str = os.getenv(
    "CONSOLE_EVENT_TOKEN", "dev-local-token-change-in-production"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _assistant_msg(content: str, tool_calls: list[dict] | None = None) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_result_msg(tool_call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def _agent_id_for_tool(tool_name: str) -> str:
    """Map tool name to agent ID for the multi-agent console.

    search_sdr_knowledge → "rag"
    all hardware tools       → "usrp"
    """
    if tool_name == "search_sdr_knowledge":
        return "rag"
    return "usrp"


async def _post_console_event(
    agent_id: str,
    run_id: str,
    session_id: str,
    phase: str,
    task: str,
    tool_name: str,
    detail: str = "",
) -> None:
    """Send a single event POST to the multi-agent console (awaitable).

    Callers that need guaranteed ordering (e.g. the initial "started" event)
    should await this directly.  For fire-and-forget use, call
    :func:`report_to_console` instead.
    """
    try:
        payload = {
            "type": "agent.activity",
            "agentId": agent_id,
            "runId": run_id,
            "sessionId": session_id,
            "phase": phase,
            "task": task,
            "toolName": tool_name,
            "detail": detail,
        }
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{CONSOLE_URL}/api/integrations/agent-events",
                json=payload,
                headers={
                    "Authorization": f"Bearer {CONSOLE_TOKEN}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning(
                        "Console event rejected (HTTP %s): %s",
                        resp.status,
                        body[:200],
                    )
    except asyncio.TimeoutError:
        logger.warning(
            "Console event timeout (%s) — is the console running?",
            CONSOLE_URL,
        )
    except aiohttp.ClientConnectorError as exc:
        logger.warning(
            "Console unreachable (%s): %s — check IP/port and network",
            CONSOLE_URL,
            exc,
        )
    except aiohttp.ClientError as exc:
        logger.warning("Console event network error: %s", exc)
    except Exception:
        logger.warning("Console event unexpected error", exc_info=True)


def report_to_console(
    agent_id: str,
    run_id: str,
    session_id: str,
    phase: str,
    task: str,
    tool_name: str,
    detail: str = "",
) -> None:
    """Send an agent-activity event to the multi-agent console via HTTP POST.

    Always fire-and-forget — never blocks the SSE main loop.

    Note: 原来的 MCP SSE 广播路径方向写反（往客户端→服务端的 read_stream
    塞消息，客户端收不到），故统一走 HTTP POST 兜底，保证多智能体控制台
    始终能收到事件。
    """
    try:
        asyncio.create_task(
            _post_console_event(
                agent_id=agent_id,
                run_id=run_id,
                session_id=session_id,
                phase=phase,
                task=task,
                tool_name=tool_name,
                detail=detail,
            )
        )
    except RuntimeError:
        logger.debug("Console event skipped: no running event loop")


# ═══════════════════════════════════════════════════════════════════════════
# Agent Loop
# ═══════════════════════════════════════════════════════════════════════════


class AgentLoop:
    """MCP-centric agent loop — model-agnostic, streaming, skill-aware."""

    def __init__(
        self,
        llm_client: Any,
        tool_registry: MCPToolRegistry,
        skill_registry: SkillRegistry | None,
        memory: ConversationMemory,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.skills = skill_registry
        self.memory = memory
        self._pending_console_run_id: str | None = None
        self._pending_console_session_id: str = ""
        self._pending_console_instruction: str = ""

    # ── public API ──────────────────────────────────────────────────────

    def normalize_mode(self, mode: str) -> str:
        key = (mode or "think").strip().lower()
        return key if key in settings.model_map else "think"

    def resolve_model(self, mode: str) -> str:
        return settings.model_map[self.normalize_mode(mode)]

    def resolve_temperature(self, mode: str, override: float | None) -> float:
        if override is not None:
            return max(0.0, min(2.0, float(override)))
        return float(settings.temperature_map[self.normalize_mode(mode)])

    def _flush_pending_completion(self) -> None:
        """If hardware became idle and a completion is pending, send it now."""
        if (
            self._pending_console_run_id is not None
            and not self.tools.is_hardware_busy()
        ):
            report_to_console(
                agent_id="usrp",
                run_id=self._pending_console_run_id,
                session_id=self._pending_console_session_id,
                phase="completed",
                task=self._pending_console_instruction,
                tool_name="",
                detail="USRP 后台任务已结束",
            )
            self._pending_console_run_id = None

    async def run_stream(
        self,
        instruction: str,
        session_id: str,
        mode: str = "think",
        temperature: float | None = None,
        max_turns: int | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the agent loop, yielding SSE-compatible events.

        Yields:
            {"type": "meta", ...}         — once, at start
            {"type": "thinking", ...}     — each turn
            {"type": "delta", "text": ""} — LLM text chunks
            {"type": "tool_call", ...}    — tool invocation requested
            {"type": "tool_result", ...}  — tool execution result
            {"type": "done", ...}         — final event
            {"type": "error", ...}        — on failure
        """
        selected_mode = self.normalize_mode(mode)
        active_model = self.resolve_model(selected_mode)
        active_temp = self.resolve_temperature(selected_mode, temperature)
        max_turns = max_turns or settings.max_tool_turns
        run_id = run_id or str(uuid.uuid4())

        # ── Step 1: Skill matching ──
        skill = None
        if self.skills is not None:
            result = self.skills.match_best(
                instruction, category="hardware", min_confidence=0.4
            )
            if result is not None:
                skill, confidence = result

        # ── Step 2: Build context ──
        history = self.memory.get_history(session_id)
        # Trim history for the mode
        history_limit = (
            settings.fast_history_messages
            if selected_mode == "fast"
            else settings.general_history_messages
        )
        trimmed_history = history[-history_limit:] if history_limit > 0 else []

        messages = build_messages(
            instruction=instruction,
            history=trimmed_history,
            skill=skill,
        )

        # ── Step 3: Get tools ──
        openai_tools = self.tools.as_openai_tools()

        # ── Meta event ──
        request_start = time.perf_counter()
        yield {
            "type": "meta",
            "active_mode": selected_mode,
            "active_model": active_model,
            "active_temperature": active_temp,
            "skill_matched": skill.name if skill else None,
            "skill_confidence": confidence if skill else 0.0,
        }

        tool_logs: list[str] = []
        final_reply = ""
        tool_call_history: list[dict] = []

        try:
            # ── Step 4: Agent loop ──
            for turn in range(max_turns):
                yield {
                    "type": "thinking",
                    "turn": turn + 1,
                    "max_turns": max_turns,
                }

                # Call LLM with streaming
                tool_choice = "required" if (skill and turn == 0) else "auto"

                stream_kwargs: dict[str, Any] = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": active_temp,
                    "stream": True,
                }
                if openai_tools:
                    stream_kwargs["tools"] = openai_tools
                    stream_kwargs["tool_choice"] = tool_choice

                stream = self.llm.chat.completions.create(**stream_kwargs)

                # Collect streaming response
                tc_parts: dict[int, dict[str, str]] = defaultdict(
                    lambda: {"id": "", "name": "", "arguments": ""}
                )
                text_parts: list[str] = []

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue

                    # Text delta — split into small chunks for visible streaming.
                    # Some Ollama versions batch many tokens per chunk;
                    # splitting ensures the browser renders incrementally.
                    if delta.content:
                        text = delta.content
                        text_parts.append(text)
                        # Yield 4-char micro-deltas so the frontend sees
                        # streaming even when the upstream batches heavily.
                        for i in range(0, len(text), 4):
                            yield {"type": "delta", "text": text[i:i+4]}

                    # Tool call delta
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = int(getattr(tc, "index", 0) or 0)
                            if getattr(tc, "id", None):
                                tc_parts[idx]["id"] = tc.id or ""
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    tc_parts[idx]["name"] = fn.name or ""
                                if getattr(fn, "arguments", None):
                                    tc_parts[idx]["arguments"] += fn.arguments

                # Build tool_calls list
                tool_calls: list[dict] = []
                for idx in sorted(tc_parts.keys()):
                    part = tc_parts[idx]
                    if not part["name"] and not part["arguments"]:
                        continue
                    tool_calls.append(
                        {
                            "id": part["id"] or f"call_{part['name']}_{idx}",
                            "type": "function",
                            "function": {
                                "name": part["name"],
                                "arguments": part["arguments"],
                            },
                        }
                    )

                assistant_text = "".join(text_parts)

                # No tool calls → LLM chose to answer directly
                if not tool_calls:
                    # If skill was matched but LLM skipped the tool, retry once with
                    # a stern reminder.  Some models (especially smaller ones) ignore
                    # tool_choice="required" and hallucinate a "success" reply.
                    if skill is not None and turn == 0 and not tool_call_history:
                        messages.append(_assistant_msg(assistant_text))
                        messages.append({
                            "role": "system",
                            "content": (
                                "⚠️ 你刚才没有调用任何工具！当前请求是硬件执行动作，"
                                "你必须调用工具来实际操作 USRP，禁止直接用文字假装已经执行。"
                                f"请立即调用 {skill.name} 工具。"
                            ),
                        })
                        yield {
                            "type": "tool_log",
                            "line": "模型未调用工具（疑似幻觉），强制重试中…",
                        }
                        yield {"type": "retry", "reason": "no_tool_call"}
                        continue

                    # Chat / knowledge route — text reply is acceptable
                    final_reply = assistant_text
                    messages.append(_assistant_msg(assistant_text))
                    break

                # Append assistant message with tool calls
                messages.append(_assistant_msg(assistant_text, tool_calls))

                # Execute each tool call
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    try:
                        tool_args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Validate tool exists
                    if self.tools.get(tool_name) is None:
                        err_result = json.dumps(
                            {
                                "status": "error",
                                "error": f"未知技能: {tool_name}",
                            },
                            ensure_ascii=False,
                        )
                        messages.append(
                            _tool_result_msg(tc["id"], tool_name, err_result)
                        )
                        tool_logs.append(f"未知技能 {tool_name}，已跳过。")
                        yield {
                            "type": "tool_call",
                            "name": tool_name,
                            "args": tool_args,
                        }
                        yield {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": {"status": "error", "error": f"未知技能: {tool_name}"},
                        }
                        continue

                    # Log and yield tool call
                    log_line = f"触发技能 {tool_name}，参数: {tool_args}"
                    tool_logs.append(log_line)
                    yield {
                        "type": "tool_call",
                        "name": tool_name,
                        "args": tool_args,
                    }
                    yield {"type": "tool_log", "line": log_line}

                    # Per-tool unique runId — keeps each tool node independent
                    # from the main task node and from other tool nodes.
                    tool_run_id = str(uuid.uuid4())

                    # Fire-and-forget: notify console tool started
                    report_to_console(
                        agent_id=_agent_id_for_tool(tool_name),
                        run_id=tool_run_id,
                        session_id=session_id,
                        phase="tool_started",
                        task=instruction,
                        tool_name=tool_name,
                    )

                    # Execute via MCP
                    tool_start = time.perf_counter()
                    result_json = await self.tools.call(tool_name, tool_args)
                    tool_elapsed = _elapsed_ms(tool_start)

                    # Parse result for logging
                    try:
                        result_obj = json.loads(result_json)
                    except json.JSONDecodeError:
                        result_obj = {"status": "unknown", "raw": result_json}

                    log_line = (
                        f"技能 {tool_name} 执行耗时 {tool_elapsed:.0f} ms，"
                        f"结果 {result_obj.get('status', 'unknown')}。"
                    )
                    tool_logs.append(log_line)
                    yield {"type": "tool_log", "line": log_line}
                    yield {
                        "type": "tool_result",
                        "name": tool_name,
                        "result": result_obj,
                        "elapsed_ms": tool_elapsed,
                    }

                    # Fire-and-forget: notify console tool completed
                    report_to_console(
                        agent_id=_agent_id_for_tool(tool_name),
                        run_id=tool_run_id,
                        session_id=session_id,
                        phase="tool_completed",
                        task=instruction,
                        tool_name=tool_name,
                        detail=f"耗时 {tool_elapsed:.0f}ms, 状态 {result_obj.get('status', 'unknown')}",
                    )

                    # Flush any pending task completion (e.g. stop_hardware_task
                    # may have just released the USRP)
                    self._flush_pending_completion()

                    tool_call_history.append(
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "status": result_obj.get("status", "unknown"),
                            "elapsed_ms": tool_elapsed,
                        }
                    )

                    # Inject tool result into messages
                    messages.append(
                        _tool_result_msg(tc["id"], tool_name, result_json)
                    )

                # If this was a forced first turn with skill, switch to auto for next
                if turn == 0 and skill:
                    # After skill-matched tool executes, let LLM decide next step
                    pass

            # ── Step 5: Save memory ──
            # Extract user/assistant pairs from messages for clean history
            clean_history: list[dict[str, str]] = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str):
                    if role == "assistant" and msg.get("tool_calls"):
                        continue  # skip tool-call-only assistant msgs in history
                    if content.strip():
                        clean_history.append({"role": role, "content": content})

            self.memory.replace(session_id, clean_history)

            # ── Done ──
            total_ms = _elapsed_ms(request_start)
            tools_executed = len([tc for tc in tool_call_history if tc.get("status") == "success"])

            # Notify console: task completed → main node disappears.
            # BUT: if the USRP is still running a background task
            # (e.g. tone_loopback_visualize), defer completion until
            # stop_hardware_task is called and hardware becomes idle.
            if self.tools.is_hardware_busy():
                # Hardware still busy — save completion for later flush
                self._pending_console_run_id = run_id
                self._pending_console_session_id = session_id
                self._pending_console_instruction = instruction
                logger.info(
                    "Deferred console completion for run %s — USRP still active",
                    run_id,
                )
            else:
                report_to_console(
                    agent_id="usrp",
                    run_id=run_id,
                    session_id=session_id,
                    phase="completed",
                    task=instruction,
                    tool_name="",
                    detail=f"执行 {tools_executed}/{len(tool_call_history)} 个工具, 总耗时 {total_ms:.0f}ms",
                )

            yield {
                "type": "done",
                "status": "success",
                "reply": final_reply,
                "hardware_logs": tool_logs,
                "updated_history": clean_history,
                "active_mode": selected_mode,
                "active_model": active_model,
                "active_temperature": active_temp,
                "tools_executed": tools_executed,
                "tools_total": len(tool_call_history),
                "timing": {
                    "total_ms": total_ms,
                    "tool_calls": tool_call_history,
                    "turns": turn + 1,
                },
            }

        except Exception as exc:
            # Flush any pending completion first (hardware may have been released)
            self._flush_pending_completion()
            # Fire-and-forget: notify console of failure
            report_to_console(
                agent_id="usrp",
                run_id=run_id,
                session_id=session_id,
                phase="failed",
                task=instruction,
                tool_name="",
                detail=str(exc),
            )
            yield {
                "type": "error",
                "message": str(exc),
                "hardware_logs": tool_logs,
                "reply": final_reply,
            }
