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
from src.agent.router import (
    build_task_plan,
    completed_actions_for_result,
    route_instruction,
    route_system_note,
)
from src.agent.trace import RunTrace, RunTraceStore, build_diagnosis
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


def _usage_metric(usage: Any, name: str) -> int | None:
    """Read an optional token metric from dict- or object-style SDK usage."""
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _tool_effect_status(raw_status: Any, completed_actions: set[str]) -> str:
    """Normalize a tool result to success/incomplete/failed for diagnostics."""
    normalized = str(raw_status or "unknown").lower()
    if normalized in {"error", "failed", "failure"}:
        return "failed"
    if normalized == "success" and completed_actions:
        return "success"
    return "incomplete"


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
        trace_store: RunTraceStore | None = None,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.skills = skill_registry
        self.memory = memory
        self.trace_store = trace_store
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
        request_started_at: float | None = None,
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
        agent_start = time.perf_counter()
        request_start = request_started_at or agent_start
        pre_agent_ms = round(max(0.0, (agent_start - request_start) * 1000), 2)
        selected_mode = self.normalize_mode(mode)
        active_model = self.resolve_model(selected_mode)
        active_temp = self.resolve_temperature(selected_mode, temperature)
        max_turns = max_turns or settings.max_tool_turns
        run_id = run_id or str(uuid.uuid4())
        trace = RunTrace(
            run_id=run_id,
            session_id=session_id,
            instruction=instruction,
            mode=selected_mode,
            model=active_model,
            temperature=active_temp,
        )

        # ── Step 1: Read short-term memory and route the request ──
        memory_start = time.perf_counter()
        history = self.memory.get_history(session_id)
        history_limit = (
            settings.fast_history_messages
            if selected_mode == "fast"
            else settings.general_history_messages
        )
        trimmed_history = history[-history_limit:] if history_limit > 0 else []
        memory_read_ms = _elapsed_ms(memory_start)

        route_start = time.perf_counter()
        route = route_instruction(instruction, trimmed_history)
        route_ms = _elapsed_ms(route_start)
        plan_start = time.perf_counter()
        plan = build_task_plan(instruction, route, self.skills, trimmed_history)
        plan_build_ms = _elapsed_ms(plan_start)
        skill = plan.skill
        confidence = plan.skill_score
        if skill is not None:
            business_intent = skill.name
            business_intent_label = skill.description
        elif route.route == "knowledge":
            business_intent = "answer_sdr_question"
            business_intent_label = "回答 SDR 专业知识问题"
        elif route.route == "chat":
            business_intent = "direct_conversation"
            business_intent_label = "普通交流，直接生成回复"
        else:
            business_intent = "clarify_hardware_request"
            business_intent_label = "硬件执行意图不明确，需要澄清"
        trace.request = {
            "raw_instruction": instruction,
            "history_messages_available": len(history),
            "history_messages_used": len(trimmed_history),
            "max_turns": max_turns,
            "pre_agent_ms": pre_agent_ms,
        }
        trace.route = {
            "name": route.route,
            "business_intent": business_intent,
            "business_intent_label": business_intent_label,
            "confidence": route.confidence,
            "reason": route.reason,
            "requires_hardware_action": route.requires_hardware_action,
            "requires_knowledge": route.requires_knowledge,
            "needs_post_analysis": route.needs_post_analysis,
        }
        trace.plan = {
            "skill": skill.name if skill else None,
            "skill_score": confidence,
            "skill_margin": plan.skill_margin,
            "skill_candidates": [
                {"name": name, "score": score}
                for name, score in plan.skill_candidates
            ],
            "high_confidence": plan.high_confidence,
            "workflow": plan.workflow,
            "initial_allowed_tools": list(plan.initial_allowed_tools),
            "required_actions": sorted(plan.required_actions),
            "needs_clarification": plan.needs_clarification,
        }
        trace.record(
            "request_understanding",
            "route_and_plan_created",
            meaning="完成请求理解，并生成受 Skill 约束的任务计划。",
            route=trace.route,
            plan=trace.plan,
            route_ms=route_ms,
            plan_build_ms=plan_build_ms,
        )

        # ── Step 2: Build route- and skill-constrained context ──
        context_start = time.perf_counter()
        messages = build_messages(
            instruction=instruction,
            history=trimmed_history,
            skill=skill,
            route_note=route_system_note(route, plan),
        )
        context_build_ms = _elapsed_ms(context_start)

        # ── Step 3: Initialize the workflow state ──
        allowed_tool_names = list(plan.initial_allowed_tools)
        completed_actions: set[str] = set()
        seen_tool_calls: set[tuple[str, str]] = set()
        force_finalize = False
        finalization_note_added = False

        # ── Meta event ──
        yield {
            "type": "meta",
            "run_id": run_id,
            "active_mode": selected_mode,
            "active_model": active_model,
            "active_temperature": active_temp,
            "active_route": route.route,
            "route_confidence": route.confidence,
            "route_reason": route.reason,
            "skill_matched": skill.name if skill else None,
            "skill_confidence": confidence if skill else 0.0,
            "skill_margin": plan.skill_margin,
            "skill_high_confidence": plan.high_confidence,
            "workflow": plan.workflow,
            "allowed_tools": list(allowed_tool_names),
            "required_actions": sorted(plan.required_actions),
        }

        tool_logs: list[str] = []
        final_reply = ""
        tool_call_history: list[dict] = []
        llm_call_metrics: list[dict[str, Any]] = []
        action_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        termination_reason = "max_turns_reached"

        try:
            # ── Step 4: Agent loop ──
            for turn in range(max_turns):
                yield {
                    "type": "thinking",
                    "turn": turn + 1,
                    "max_turns": max_turns,
                }

                # Only expose tools that are valid for the current workflow
                # stage.  Chat, clarification and finalization stages receive
                # no tools at all.
                openai_tools = (
                    []
                    if force_finalize or plan.needs_clarification
                    else self.tools.as_openai_tools(names=allowed_tool_names)
                )
                attempted_current_stage = any(
                    call.get("name") in allowed_tool_names
                    and not call.get("rejected", False)
                    for call in tool_call_history
                )
                tool_choice = "auto" if attempted_current_stage else "required"
                if openai_tools:
                    llm_purpose = "tool_decision"
                elif plan.needs_clarification:
                    llm_purpose = "clarification_response"
                elif force_finalize:
                    llm_purpose = "final_response"
                else:
                    llm_purpose = "direct_response"
                trace.record(
                    "llm_decision",
                    "turn_started",
                    meaning=f"开始第 {turn + 1} 轮 LLM 调用，用途为 {llm_purpose}。",
                    turn=turn + 1,
                    purpose=llm_purpose,
                    allowed_tools=list(allowed_tool_names),
                    tool_choice=tool_choice if openai_tools else "none",
                    force_finalize=force_finalize,
                    completed_actions=sorted(completed_actions),
                )

                stream_kwargs: dict[str, Any] = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": active_temp,
                    "stream": True,
                }
                if openai_tools:
                    stream_kwargs["tools"] = openai_tools
                    stream_kwargs["tool_choice"] = tool_choice

                input_characters = len(
                    json.dumps(messages, ensure_ascii=False, default=str)
                )
                tool_schema_characters = len(
                    json.dumps(openai_tools, ensure_ascii=False, default=str)
                )
                llm_start = time.perf_counter()
                stream = self.llm.chat.completions.create(**stream_kwargs)
                llm_service_wait_ms = _elapsed_ms(llm_start)

                # Collect streaming response
                tc_parts: dict[int, dict[str, str]] = defaultdict(
                    lambda: {"id": "", "name": "", "arguments": ""}
                )
                text_parts: list[str] = []
                first_output_ms: float | None = None
                first_text_ms: float | None = None
                stream_chunks = 0
                prompt_tokens: int | None = None
                completion_tokens: int | None = None
                total_tokens: int | None = None

                stream_iterator = iter(stream)
                while True:
                    chunk_wait_start = time.perf_counter()
                    try:
                        chunk = next(stream_iterator)
                    except StopIteration:
                        llm_service_wait_ms += _elapsed_ms(chunk_wait_start)
                        break
                    llm_service_wait_ms += _elapsed_ms(chunk_wait_start)
                    stream_chunks += 1
                    usage = getattr(chunk, "usage", None)
                    prompt_tokens = _usage_metric(usage, "prompt_tokens") or prompt_tokens
                    completion_tokens = (
                        _usage_metric(usage, "completion_tokens") or completion_tokens
                    )
                    total_tokens = _usage_metric(usage, "total_tokens") or total_tokens
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue

                    if first_output_ms is None and (
                        delta.content or delta.tool_calls
                    ):
                        first_output_ms = round(llm_service_wait_ms, 2)

                    # Text delta — split into small chunks for visible streaming.
                    # Some Ollama versions batch many tokens per chunk;
                    # splitting ensures the browser renders incrementally.
                    if delta.content:
                        if first_text_ms is None:
                            first_text_ms = round(llm_service_wait_ms, 2)
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
                llm_stream_wall_ms = _elapsed_ms(llm_start)
                llm_total_ms = round(llm_service_wait_ms, 2)
                generation_ms = (
                    round(max(0.0, llm_total_ms - first_output_ms), 2)
                    if first_output_ms is not None
                    else None
                )
                tokens_per_second = None
                if completion_tokens is not None and generation_ms and generation_ms > 0:
                    tokens_per_second = round(completion_tokens / (generation_ms / 1000), 2)
                call_metrics = {
                    "call_index": len(llm_call_metrics) + 1,
                    "turn": turn + 1,
                    "purpose": llm_purpose,
                    "total_ms": llm_total_ms,
                    "stream_wall_ms": llm_stream_wall_ms,
                    "first_output_ms": first_output_ms,
                    "first_text_ms": first_text_ms,
                    "generation_ms": generation_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "tokens_per_second": tokens_per_second,
                    "input_messages": len(messages),
                    "input_characters": input_characters,
                    "tool_schema_characters": tool_schema_characters,
                    "response_characters": len(assistant_text),
                    "proposed_tool_calls": len(tool_calls),
                    "stream_chunks": stream_chunks,
                }
                llm_call_metrics.append(call_metrics)
                trace.record(
                    "llm_decision",
                    "model_response",
                    meaning=(
                        "LLM 完成本轮工具决策。"
                        if llm_purpose == "tool_decision"
                        else "LLM 完成本轮文本回复。"
                    ),
                    turn=turn + 1,
                    purpose=llm_purpose,
                    assistant_text=assistant_text,
                    tool_calls=tool_calls,
                    metrics=call_metrics,
                )

                # No tool calls → LLM chose to answer directly
                if not tool_calls:
                    # A required workflow stage cannot be skipped by a textual
                    # claim. Retry while a not-yet-attempted tool stage exists.
                    if openai_tools and not attempted_current_stage:
                        messages.append(_assistant_msg(assistant_text))
                        messages.append({
                            "role": "system",
                            "content": (
                                "你刚才没有调用当前工作流要求的工具。禁止用文字假装已经完成。"
                                f"本阶段只允许调用：{'、'.join(allowed_tool_names)}。"
                            ),
                        })
                        yield {
                            "type": "tool_log",
                            "line": "模型未调用工具（疑似幻觉），强制重试中…",
                        }
                        yield {"type": "retry", "reason": "no_tool_call"}
                        trace.record(
                            "llm_decision",
                            "required_tool_missing",
                            meaning="LLM 没有执行当前工作流要求的工具，Agent 将重试。",
                            turn=turn + 1,
                            allowed_tools=list(allowed_tool_names),
                        )
                        continue

                    # Chat, clarification, failure explanation, or forced
                    # finalization — a textual reply is now acceptable.
                    final_reply = assistant_text
                    messages.append(_assistant_msg(assistant_text))
                    termination_reason = (
                        "final_response"
                        if assistant_text.strip()
                        else "empty_final_response"
                    )
                    break

                # Append assistant message with tool calls
                messages.append(_assistant_msg(assistant_text, tool_calls))

                # Execute at most one valid tool call per turn.  Every extra or
                # out-of-stage call receives an explicit tool error so the
                # message sequence remains valid for OpenAI-compatible APIs.
                for call_index, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    raw_arguments = fn.get("arguments", "{}")
                    rejection_reason = ""
                    rejection_code = ""
                    try:
                        tool_args = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        tool_args = {}
                        rejection_reason = "工具参数不是有效 JSON"
                        rejection_code = "invalid_json_arguments"

                    if not isinstance(tool_args, dict):
                        tool_args = {}
                        rejection_reason = "工具参数必须是 JSON 对象"
                        rejection_code = "arguments_not_object"
                    elif call_index > 0:
                        rejection_reason = "同一轮只允许调用一个工具"
                        rejection_code = "multiple_tools_in_one_turn"
                    elif tool_name not in allowed_tool_names:
                        rejection_reason = (
                            f"工具 {tool_name} 不属于当前工作流阶段；"
                            f"允许的工具：{'、'.join(allowed_tool_names) or '无'}"
                        )
                        rejection_code = "tool_outside_allowed_set"
                    elif self.tools.get(tool_name) is None:
                        rejection_reason = f"未知工具: {tool_name}"
                        rejection_code = "unknown_tool"

                    signature = (
                        tool_name,
                        json.dumps(tool_args, sort_keys=True, ensure_ascii=False),
                    )
                    if not rejection_reason and signature in seen_tool_calls:
                        rejection_reason = "本次请求内检测到相同工具和相同参数的重复调用"
                        rejection_code = "duplicate_tool_call"

                    if rejection_reason:
                        result_obj = {"status": "error", "error": rejection_reason}
                        result_json = json.dumps(result_obj, ensure_ascii=False)
                        messages.append(_tool_result_msg(tc["id"], tool_name, result_json))
                        tool_call_history.append(
                            {
                                "turn": turn + 1,
                                "call_index": call_index + 1,
                                "name": tool_name,
                                "args": tool_args,
                                "status": "error",
                                "execution_status": "failed",
                                "decision_valid": False,
                                "elapsed_ms": 0.0,
                                "completed_actions": [],
                                "rejected": True,
                                "rejection_code": rejection_code,
                                "error": rejection_reason,
                            }
                        )
                        trace.record(
                            "tool_selection",
                            "tool_call_rejected",
                            meaning="Agent 拒绝了不符合当前工作流约束的工具调用。",
                            turn=turn + 1,
                            tool=tool_name,
                            args=tool_args,
                            decision_valid=False,
                            rejection_code=rejection_code,
                            reason=rejection_reason,
                        )
                        tool_logs.append(rejection_reason)
                        yield {"type": "tool_call", "name": tool_name, "args": tool_args}
                        yield {"type": "tool_log", "line": rejection_reason}
                        yield {"type": "tool_result", "name": tool_name, "result": result_obj}
                        continue

                    seen_tool_calls.add(signature)

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

                    new_actions = completed_actions_for_result(
                        tool_name, tool_args, result_obj
                    )
                    completed_actions.update(new_actions)
                    for action in sorted(new_actions):
                        action_evidence[action].append(
                            {
                                "turn": turn + 1,
                                "tool": tool_name,
                                "raw_status": result_obj.get("status", "unknown"),
                            }
                        )
                    execution_status = _tool_effect_status(
                        result_obj.get("status"), new_actions
                    )

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
                            "turn": turn + 1,
                            "call_index": call_index + 1,
                            "name": tool_name,
                            "args": tool_args,
                            "status": result_obj.get("status", "unknown"),
                            "execution_status": execution_status,
                            "decision_valid": True,
                            "elapsed_ms": tool_elapsed,
                            "completed_actions": sorted(new_actions),
                            "rejected": False,
                            "error": result_obj.get("error"),
                        }
                    )
                    trace.record(
                        "tool_execution",
                        "tool_completed",
                        meaning=(
                            "工具执行后产生了可验证的业务动作。"
                            if new_actions
                            else "工具已返回，但没有完成当前要求的业务动作。"
                        ),
                        turn=turn + 1,
                        tool=tool_name,
                        args=tool_args,
                        raw_status=result_obj.get("status", "unknown"),
                        execution_status=execution_status,
                        decision_valid=True,
                        elapsed_ms=tool_elapsed,
                        result=result_obj,
                        new_actions=sorted(new_actions),
                    )

                    # Inject tool result into messages
                    messages.append(
                        _tool_result_msg(tc["id"], tool_name, result_json)
                    )

                    # Advance conditional workflows one stage at a time.
                    if tool_name == "search_sdr_knowledge":
                        if "answer_with_knowledge" in new_actions:
                            allowed_tool_names = []
                        else:
                            allowed_tool_names = ["web_search"]
                    elif tool_name == "web_search":
                        # Whether results were found or not, do not search in a
                        # loop. The next turn must explain the outcome.
                        allowed_tool_names = []
                        force_finalize = True

                required_actions = set(plan.required_actions)
                hardware_actions = required_actions - {"answer_with_knowledge"}
                hardware_stage_complete = hardware_actions.issubset(completed_actions)

                if route.route == "hardware" and hardware_stage_complete:
                    if (
                        "answer_with_knowledge" in required_actions
                        and "answer_with_knowledge" not in completed_actions
                    ):
                        if not force_finalize and allowed_tool_names != ["web_search"]:
                            allowed_tool_names = ["search_sdr_knowledge"]
                    else:
                        allowed_tool_names = []
                        force_finalize = True
                elif route.route == "knowledge" and required_actions.issubset(completed_actions):
                    allowed_tool_names = []
                    force_finalize = True

                if force_finalize and not finalization_note_added:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "工具阶段已经结束。下一轮禁止调用任何工具；"
                                "请只根据用户原始请求和真实工具结果生成最终中文答复。"
                            ),
                        }
                    )
                    finalization_note_added = True
                elif completed_actions and required_actions:
                    missing = sorted(required_actions - completed_actions)
                    if missing:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "当前业务尚未完成，仍缺少动作："
                                    f"{'、'.join(missing)}。只能使用当前阶段允许的工具继续，"
                                    "不要重复已经成功且参数相同的调用。"
                                ),
                            }
                        )
                trace.record(
                    "loop_control",
                    "stage_evaluated",
                    meaning=(
                        "当前必需动作已经满足，下一步生成最终回复。"
                        if required_actions.issubset(completed_actions)
                        else "当前业务尚未完成，Agent 根据缺失动作决定是否继续循环。"
                    ),
                    turn=turn + 1,
                    allowed_tools=list(allowed_tool_names),
                    required_actions=sorted(required_actions),
                    completed_actions=sorted(completed_actions),
                    missing_actions=sorted(required_actions - completed_actions),
                    actions_satisfied=required_actions.issubset(completed_actions),
                    force_finalize=force_finalize,
                )

            # ── Step 5: Classify the task outcome ──
            # Tool-level status answers "did this tool invocation run?".  The
            # task-level status below answers "did this agent run produce a
            # usable final outcome?".  Keep these concepts separate.
            successful_tool_calls = [
                tc for tc in tool_call_history if tc.get("status") == "success"
            ]
            has_final_reply = bool(final_reply.strip())
            required_actions = set(plan.required_actions)
            business_complete = (
                route.route == "chat" and has_final_reply
            ) or (
                bool(required_actions)
                and required_actions.issubset(completed_actions)
                and has_final_reply
            )

            if plan.needs_clarification and has_final_reply:
                task_status = "incomplete"
                termination_reason = "clarification_required"
            elif business_complete:
                task_status = "success"
            elif has_final_reply and not successful_tool_calls and tool_call_history:
                task_status = "failed"
                termination_reason = "all_tool_calls_failed"
            elif has_final_reply:
                task_status = "incomplete"
                termination_reason = "business_actions_missing"
            elif completed_actions or successful_tool_calls:
                task_status = "incomplete"
                termination_reason = (
                    "final_response_missing"
                    if required_actions.issubset(completed_actions)
                    else "max_turns_reached"
                )
                missing = sorted(required_actions - completed_actions)
                missing_note = f"，仍缺少：{'、'.join(missing)}" if missing else ""
                final_reply = (
                    f"已有 {len(successful_tool_calls)} 次工具调用执行成功，但任务未完整结束"
                    f"{missing_note}。请查看工具日志后重试。"
                )
            else:
                task_status = "failed"
                termination_reason = (
                    termination_reason
                    if termination_reason != "max_turns_reached"
                    else "no_successful_result"
                )
                final_reply = (
                    f"任务未完成：Agent 在 {max_turns} 轮内没有生成有效答复，"
                    "也没有成功的工具调用。请查看工具日志后重试。"
                )

            # Persist the generated fallback as the assistant's visible reply,
            # so the next user turn does not inherit an unexplained blank.
            if not has_final_reply:
                messages.append(_assistant_msg(final_reply))

            # ── Step 6: Save memory ──
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
            tools_executed = len(successful_tool_calls)
            rejected_calls = [
                call for call in tool_call_history if call.get("rejected", False)
            ]
            duplicate_call_count = sum(
                call.get("rejection_code") == "duplicate_tool_call"
                for call in rejected_calls
            )
            actual_tool_calls = [
                call for call in tool_call_history if not call.get("rejected", False)
            ]
            llm_total_ms = round(
                sum(float(call.get("total_ms") or 0) for call in llm_call_metrics), 2
            )
            llm_stream_wall_ms = round(
                sum(
                    float(call.get("stream_wall_ms") or 0)
                    for call in llm_call_metrics
                ),
                2,
            )
            stream_delivery_ms = round(
                max(0.0, llm_stream_wall_ms - llm_total_ms), 2
            )
            tool_execution_ms = round(
                sum(float(call.get("elapsed_ms") or 0) for call in actual_tool_calls),
                2,
            )
            knowledge_lookup_ms = round(
                sum(
                    float(call.get("elapsed_ms") or 0)
                    for call in actual_tool_calls
                    if call.get("name") in {"search_sdr_knowledge", "web_search"}
                ),
                2,
            )
            hardware_execution_ms = round(
                max(0.0, tool_execution_ms - knowledge_lookup_ms), 2
            )
            tool_decision_ms = round(
                sum(
                    float(call.get("total_ms") or 0)
                    for call in llm_call_metrics
                    if call.get("purpose") == "tool_decision"
                ),
                2,
            )
            final_reply_ms = round(
                sum(
                    float(call.get("total_ms") or 0)
                    for call in llm_call_metrics
                    if call.get("purpose") != "tool_decision"
                ),
                2,
            )
            final_text_calls = [
                call
                for call in llm_call_metrics
                if call.get("purpose") != "tool_decision"
                and call.get("first_text_ms") is not None
            ]
            first_token_ms = (
                final_text_calls[-1]["first_text_ms"] if final_text_calls else None
            )
            overhead_ms = round(
                max(0.0, total_ms - llm_stream_wall_ms - tool_execution_ms), 2
            )
            actions_satisfied = (
                required_actions.issubset(completed_actions)
                if required_actions
                else route.route == "chat"
            )
            completion = {
                "actions_satisfied": actions_satisfied,
                "model_final_reply_present": has_final_reply,
                "visible_reply_present": bool(final_reply.strip()),
                "task_satisfied": business_complete,
                "required_actions": sorted(required_actions),
                "completed_actions": sorted(completed_actions),
                "missing_actions": sorted(required_actions - completed_actions),
                "action_evidence": dict(action_evidence),
            }
            trace.metrics = {
                "timing": {
                    "total_ms": total_ms,
                    "pre_agent_ms": pre_agent_ms,
                    "memory_read_ms": memory_read_ms,
                    "route_ms": route_ms,
                    "plan_build_ms": plan_build_ms,
                    "context_build_ms": context_build_ms,
                    "llm_total_ms": llm_total_ms,
                    "llm_stream_wall_ms": llm_stream_wall_ms,
                    "stream_delivery_ms": stream_delivery_ms,
                    "tool_decision_ms": tool_decision_ms,
                    "final_reply_ms": final_reply_ms,
                    "tool_execution_ms": tool_execution_ms,
                    "hardware_execution_ms": hardware_execution_ms,
                    "knowledge_lookup_ms": knowledge_lookup_ms,
                    "first_token_ms": first_token_ms,
                    "agent_overhead_ms": overhead_ms,
                },
                "counts": {
                    "turns_used": turn + 1,
                    "llm_calls": len(llm_call_metrics),
                    "tool_calls_proposed": sum(
                        int(call.get("proposed_tool_calls") or 0)
                        for call in llm_call_metrics
                    ),
                    "tool_calls_executed": len(actual_tool_calls),
                    "tool_calls_rejected": len(rejected_calls),
                    "duplicate_call_count": duplicate_call_count,
                },
                "llm_calls": llm_call_metrics,
            }
            diagnosis = build_diagnosis(
                status=task_status,
                termination_reason=termination_reason,
                route=route.route,
                skill_name=skill.name if skill else None,
                tool_calls=tool_call_history,
                required_actions=required_actions,
                completed_actions=completed_actions,
            )
            trace.record(
                "completion_check",
                "task_classified",
                meaning=(
                    "所有完成条件均已满足，任务判定为 success。"
                    if task_status == "success"
                    else "任务完成条件未全部满足，已记录缺失动作和停止原因。"
                ),
                status=task_status,
                termination_reason=termination_reason,
                completion=completion,
                diagnosis=diagnosis,
            )
            trace.finish(
                status=task_status,
                termination_reason=termination_reason,
                reply=final_reply,
                total_ms=total_ms,
                tool_calls=tool_call_history,
                required_actions=sorted(required_actions),
                completed_actions=sorted(completed_actions),
                missing_actions=sorted(required_actions - completed_actions),
                completion=completion,
                diagnosis=diagnosis,
            )
            if self.trace_store is not None:
                self.trace_store.append(trace)

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
                "run_id": run_id,
                "status": task_status,
                "termination_reason": termination_reason,
                "diagnosis": diagnosis,
                "reply": final_reply,
                "hardware_logs": tool_logs,
                "updated_history": clean_history,
                "active_mode": selected_mode,
                "active_model": active_model,
                "active_temperature": active_temp,
                "active_route": route.route,
                "route_confidence": route.confidence,
                "skill_matched": skill.name if skill else None,
                "workflow": plan.workflow,
                "required_actions": sorted(required_actions),
                "completed_actions": sorted(completed_actions),
                "tools_executed": tools_executed,
                "tools_total": len(tool_call_history),
                "timing": {
                    "total_ms": total_ms,
                    "pre_agent_ms": pre_agent_ms,
                    "route": route.route,
                    "memory_read_ms": memory_read_ms,
                    "route_ms": route_ms,
                    "plan_build_ms": plan_build_ms,
                    "context_build_ms": context_build_ms,
                    "llm_total_ms": llm_total_ms,
                    "llm_stream_wall_ms": llm_stream_wall_ms,
                    "stream_delivery_ms": stream_delivery_ms,
                    "knowledge_lookup_ms": knowledge_lookup_ms,
                    "tool_decision_ms": tool_decision_ms,
                    "tool_execution_ms": tool_execution_ms,
                    "hardware_execution_ms": hardware_execution_ms,
                    "final_reply_ms": final_reply_ms,
                    "first_token_ms": first_token_ms,
                    "agent_overhead_ms": overhead_ms,
                    "tool_count": len(tool_call_history),
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
            diagnosis = {
                "stage": "exception",
                "diagnosis_code": "unhandled_exception",
                "suspected_component": "agent_loop_or_dependency",
                "retryable": True,
                "summary": str(exc),
                "suggested_files": ["src/agent/loop.py"],
            }
            total_ms = _elapsed_ms(request_start)
            trace.metrics = {
                "timing": {
                    "total_ms": total_ms,
                    "pre_agent_ms": pre_agent_ms,
                    "memory_read_ms": memory_read_ms,
                    "route_ms": route_ms,
                    "plan_build_ms": plan_build_ms,
                    "context_build_ms": context_build_ms,
                    "llm_total_ms": round(
                        sum(
                            float(call.get("total_ms") or 0)
                            for call in llm_call_metrics
                        ),
                        2,
                    ),
                    "tool_execution_ms": round(
                        sum(
                            float(call.get("elapsed_ms") or 0)
                            for call in tool_call_history
                            if not call.get("rejected", False)
                        ),
                        2,
                    ),
                },
                "counts": {
                    "llm_calls": len(llm_call_metrics),
                    "tool_calls_recorded": len(tool_call_history),
                },
                "llm_calls": llm_call_metrics,
            }
            trace.record(
                "exception",
                "unhandled_exception",
                meaning="Agent 运行出现未处理异常，任务被判定为 failed。",
                error=str(exc),
            )
            trace.finish(
                status="failed",
                termination_reason="exception",
                reply=final_reply,
                total_ms=total_ms,
                tool_calls=tool_call_history,
                required_actions=sorted(plan.required_actions),
                completed_actions=sorted(completed_actions),
                missing_actions=sorted(set(plan.required_actions) - completed_actions),
                diagnosis=diagnosis,
            )
            if self.trace_store is not None:
                self.trace_store.append(trace)
            yield {
                "type": "error",
                "run_id": run_id,
                "status": "failed",
                "message": str(exc),
                "diagnosis": diagnosis,
                "hardware_logs": tool_logs,
                "reply": final_reply,
            }
