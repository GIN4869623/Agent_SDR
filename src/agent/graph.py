"""LangGraph orchestration for the SDR agent.

The graph owns control flow only. All tool schemas, validation and execution
still cross the canonical :class:`tools.ToolRegistry` boundary shared with the
external MCP adapter.

Topology::

    START -> prepare -> call_model
                         | tool calls -> execute_tools --+
                         | retry -------------------------+
                         | final text / turn limit
                         v
                      finalize -> END
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from core.memory import ConversationMemory
from skill.skill_spec import SkillRegistry
from src.agent import loop as loop_runtime
from src.agent.context import build_messages
from src.agent.router import (
    build_task_plan,
    completed_actions_for_result,
    route_instruction,
    route_system_note,
)
from src.agent.trace import RunTrace, RunTraceStore, build_diagnosis
from src.config import settings
from tools import ToolRegistry

logger = logging.getLogger(__name__)

GraphAction = Literal["call_model", "execute_tools", "finalize"]


class AgentState(TypedDict, total=False):
    """State passed between orchestration nodes.

    No checkpointer is attached yet, so route/plan/trace objects may remain
    in-process Python objects. Durable resume can later replace these fields
    with their serializable representations without changing the tool layer.
    """

    instruction: str
    session_id: str
    selected_mode: str
    active_model: str
    active_temperature: float
    max_turns: int
    run_id: str
    request_started_at: float
    agent_started_at: float
    pre_agent_ms: float
    route: Any
    plan: Any
    trace: RunTrace
    messages: list[dict[str, Any]]
    allowed_tool_names: list[str]
    completed_actions: set[str]
    seen_tool_calls: set[tuple[str, str]]
    force_finalize: bool
    finalization_note_added: bool
    tool_logs: list[str]
    final_reply: str
    tool_call_history: list[dict[str, Any]]
    llm_call_metrics: list[dict[str, Any]]
    action_evidence: dict[str, list[dict[str, Any]]]
    pending_tool_calls: list[dict[str, Any]]
    turn: int
    next_action: GraphAction
    termination_reason: str
    memory_read_ms: float
    route_ms: float
    plan_build_ms: float
    context_build_ms: float


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _usage_metric(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _tool_effect_status(raw_status: Any, completed_actions: set[str]) -> str:
    normalized = str(raw_status or "unknown").lower()
    if normalized in {"error", "failed", "failure"}:
        return "failed"
    if normalized == "success" and completed_actions:
        return "success"
    return "incomplete"


def _assistant_msg(content: str, tool_calls: list[dict] | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_result_msg(tool_call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


class AgentLoop:
    """Compatibility facade backed by a compiled LangGraph state graph."""

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
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
        self._pending_console_session_id = ""
        self._pending_console_instruction = ""
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("call_model", self._call_model_node)
        builder.add_node("execute_tools", self._execute_tools_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "call_model")
        builder.add_conditional_edges(
            "call_model",
            self._next_action,
            {
                "call_model": "call_model",
                "execute_tools": "execute_tools",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "execute_tools",
            self._next_action,
            {"call_model": "call_model", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    async def _next_action(state: AgentState) -> GraphAction:
        # Keep the router async. In restricted runtimes LangGraph's sync
        # conditional runner may wait on an unavailable worker thread.
        return state.get("next_action", "finalize")

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
        if (
            self._pending_console_run_id is not None
            and not self.tools.is_hardware_busy()
        ):
            loop_runtime.report_to_console(
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
        """Run the graph while preserving the existing SSE event contract."""
        agent_start = time.perf_counter()
        request_start = request_started_at or agent_start
        selected_mode = self.normalize_mode(mode)
        active_run_id = run_id or str(uuid.uuid4())
        turn_limit = max_turns or settings.max_tool_turns
        trace = RunTrace(
            run_id=active_run_id,
            session_id=session_id,
            instruction=instruction,
            mode=selected_mode,
            model=self.resolve_model(selected_mode),
            temperature=self.resolve_temperature(selected_mode, temperature),
        )
        initial_state: AgentState = {
            "instruction": instruction,
            "session_id": session_id,
            "selected_mode": selected_mode,
            "active_model": self.resolve_model(selected_mode),
            "active_temperature": self.resolve_temperature(selected_mode, temperature),
            "max_turns": turn_limit,
            "run_id": active_run_id,
            "request_started_at": request_start,
            "agent_started_at": agent_start,
            "pre_agent_ms": round(max(0.0, (agent_start - request_start) * 1000), 2),
            "trace": trace,
            "messages": [],
            "allowed_tool_names": [],
            "completed_actions": set(),
            "seen_tool_calls": set(),
            "force_finalize": False,
            "finalization_note_added": False,
            "tool_logs": [],
            "final_reply": "",
            "tool_call_history": [],
            "llm_call_metrics": [],
            "action_evidence": {},
            "pending_tool_calls": [],
            "turn": 0,
            "next_action": "call_model",
            "termination_reason": "max_turns_reached",
            "memory_read_ms": 0.0,
            "route_ms": 0.0,
            "plan_build_ms": 0.0,
            "context_build_ms": 0.0,
        }
        streamed_logs: list[str] = []
        streamed_reply: list[str] = []
        try:
            async for chunk in self.graph.astream(
                initial_state,
                config={"recursion_limit": max(25, turn_limit * 3 + 5)},
                stream_mode="custom",
                version="v2",
            ):
                event = (
                    chunk.get("data")
                    if isinstance(chunk, dict) and chunk.get("type") == "custom"
                    else chunk
                )
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "tool_log" and event.get("line"):
                    streamed_logs.append(str(event["line"]))
                elif event.get("type") == "delta" and event.get("text"):
                    streamed_reply.append(str(event["text"]))
                yield event
        except Exception as exc:
            self._flush_pending_completion()
            loop_runtime.report_to_console(
                agent_id="usrp",
                run_id=active_run_id,
                session_id=session_id,
                phase="failed",
                task=instruction,
                tool_name="",
                detail=str(exc),
            )
            diagnosis = {
                "stage": "exception",
                "diagnosis_code": "unhandled_exception",
                "suspected_component": "langgraph_or_dependency",
                "retryable": True,
                "summary": str(exc),
                "suggested_files": ["src/agent/graph.py"],
            }
            total_ms = _elapsed_ms(request_start)
            trace.record(
                "exception",
                "unhandled_exception",
                meaning="LangGraph 运行出现未处理异常，任务被判定为 failed。",
                error=str(exc),
            )
            trace.finish(
                status="failed",
                termination_reason="exception",
                reply="".join(streamed_reply),
                total_ms=total_ms,
                tool_calls=[],
                required_actions=[],
                completed_actions=[],
                missing_actions=[],
                diagnosis=diagnosis,
            )
            if self.trace_store is not None:
                self.trace_store.append(trace)
            yield {
                "type": "error",
                "run_id": active_run_id,
                "status": "failed",
                "message": str(exc),
                "diagnosis": diagnosis,
                "hardware_logs": streamed_logs,
                "reply": "".join(streamed_reply),
            }

    async def _prepare_node(
        self, state: AgentState, writer: StreamWriter
    ) -> dict[str, Any]:
        memory_start = time.perf_counter()
        history = self.memory.get_history(state["session_id"])
        history_limit = (
            settings.fast_history_messages
            if state["selected_mode"] == "fast"
            else settings.general_history_messages
        )
        trimmed_history = history[-history_limit:] if history_limit > 0 else []
        memory_read_ms = _elapsed_ms(memory_start)

        route_start = time.perf_counter()
        route = route_instruction(state["instruction"], trimmed_history)
        route_ms = _elapsed_ms(route_start)
        plan_start = time.perf_counter()
        plan = build_task_plan(
            state["instruction"], route, self.skills, trimmed_history
        )
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

        trace = state["trace"]
        trace.request = {
            "raw_instruction": state["instruction"],
            "history_messages_available": len(history),
            "history_messages_used": len(trimmed_history),
            "max_turns": state["max_turns"],
            "pre_agent_ms": state["pre_agent_ms"],
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

        context_start = time.perf_counter()
        messages = build_messages(
            instruction=state["instruction"],
            history=trimmed_history,
            skill=skill,
            route_note=route_system_note(route, plan),
        )
        context_build_ms = _elapsed_ms(context_start)
        allowed = list(plan.initial_allowed_tools)
        writer(
            {
                "type": "meta",
                "run_id": state["run_id"],
                "active_mode": state["selected_mode"],
                "active_model": state["active_model"],
                "active_temperature": state["active_temperature"],
                "active_route": route.route,
                "route_confidence": route.confidence,
                "route_reason": route.reason,
                "skill_matched": skill.name if skill else None,
                "skill_confidence": confidence if skill else 0.0,
                "skill_margin": plan.skill_margin,
                "skill_high_confidence": plan.high_confidence,
                "workflow": plan.workflow,
                "allowed_tools": allowed,
                "required_actions": sorted(plan.required_actions),
                "orchestrator": "langgraph",
            }
        )
        return {
            "route": route,
            "plan": plan,
            "messages": messages,
            "allowed_tool_names": allowed,
            "memory_read_ms": memory_read_ms,
            "route_ms": route_ms,
            "plan_build_ms": plan_build_ms,
            "context_build_ms": context_build_ms,
            "next_action": "call_model",
        }

    async def _call_model_node(
        self, state: AgentState, writer: StreamWriter
    ) -> dict[str, Any]:
        turn = int(state.get("turn", 0))
        max_turns = int(state["max_turns"])
        if turn >= max_turns:
            return {"next_action": "finalize"}

        writer({"type": "thinking", "turn": turn + 1, "max_turns": max_turns})
        plan = state["plan"]
        allowed = list(state["allowed_tool_names"])
        force_finalize = bool(state["force_finalize"])
        tool_history = list(state["tool_call_history"])
        openai_tools = (
            []
            if force_finalize or plan.needs_clarification
            else self.tools.as_openai_tools(names=allowed)
        )
        attempted_stage = any(
            call.get("name") in allowed and not call.get("rejected", False)
            for call in tool_history
        )
        tool_choice = "auto" if attempted_stage else "required"
        if openai_tools:
            purpose = "tool_decision"
        elif plan.needs_clarification:
            purpose = "clarification_response"
        elif force_finalize:
            purpose = "final_response"
        else:
            purpose = "direct_response"

        trace = state["trace"]
        completed = set(state["completed_actions"])
        trace.record(
            "llm_decision",
            "turn_started",
            meaning=f"开始第 {turn + 1} 轮 LLM 调用，用途为 {purpose}。",
            turn=turn + 1,
            purpose=purpose,
            allowed_tools=allowed,
            tool_choice=tool_choice if openai_tools else "none",
            force_finalize=force_finalize,
            completed_actions=sorted(completed),
        )
        messages = list(state["messages"])
        stream_kwargs: dict[str, Any] = {
            "model": state["active_model"],
            "messages": messages,
            "temperature": state["active_temperature"],
            "stream": True,
        }
        if openai_tools:
            stream_kwargs["tools"] = openai_tools
            stream_kwargs["tool_choice"] = tool_choice

        input_characters = len(json.dumps(messages, ensure_ascii=False, default=str))
        schema_characters = len(
            json.dumps(openai_tools, ensure_ascii=False, default=str)
        )
        llm_start = time.perf_counter()
        stream = self.llm.chat.completions.create(**stream_kwargs)
        if inspect.isawaitable(stream):
            stream = await stream
        initial_wait_ms = _elapsed_ms(llm_start)
        service_wait_ms = initial_wait_ms
        stream_chunks = 0
        first_output_ms: float | None = None
        first_text_ms: float | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = defaultdict(
            lambda: {"id": "", "name": "", "arguments": ""}
        )

        async def consume(chunk: Any, waited_ms: float) -> None:
            nonlocal service_wait_ms, stream_chunks, first_output_ms, first_text_ms
            nonlocal prompt_tokens, completion_tokens, total_tokens
            service_wait_ms += waited_ms
            stream_chunks += 1
            usage = getattr(chunk, "usage", None)
            prompt_tokens = _usage_metric(usage, "prompt_tokens") or prompt_tokens
            completion_tokens = (
                _usage_metric(usage, "completion_tokens") or completion_tokens
            )
            total_tokens = _usage_metric(usage, "total_tokens") or total_tokens
            if not getattr(chunk, "choices", None):
                return
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                return
            content = getattr(delta, "content", None)
            calls = getattr(delta, "tool_calls", None)
            if first_output_ms is None and (content or calls):
                first_output_ms = round(service_wait_ms, 2)
            if content:
                if first_text_ms is None:
                    first_text_ms = round(service_wait_ms, 2)
                text = str(content)
                text_parts.append(text)
                for index in range(0, len(text), 4):
                    writer({"type": "delta", "text": text[index : index + 4]})
            if calls:
                for call in calls:
                    index = int(getattr(call, "index", 0) or 0)
                    if getattr(call, "id", None):
                        tool_parts[index]["id"] = call.id or ""
                    function = getattr(call, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            tool_parts[index]["name"] = function.name or ""
                        if getattr(function, "arguments", None):
                            tool_parts[index]["arguments"] += function.arguments

        if hasattr(stream, "__aiter__"):
            iterator = stream.__aiter__()
            while True:
                wait_start = time.perf_counter()
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    service_wait_ms += _elapsed_ms(wait_start)
                    break
                await consume(chunk, _elapsed_ms(wait_start))
        else:
            iterator = iter(stream)
            while True:
                wait_start = time.perf_counter()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    service_wait_ms += _elapsed_ms(wait_start)
                    break
                await consume(chunk, _elapsed_ms(wait_start))

        tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_parts):
            part = tool_parts[index]
            if not part["name"] and not part["arguments"]:
                continue
            tool_calls.append(
                {
                    "id": part["id"] or f"call_{part['name']}_{index}",
                    "type": "function",
                    "function": {
                        "name": part["name"],
                        "arguments": part["arguments"],
                    },
                }
            )
        assistant_text = "".join(text_parts)
        stream_wall_ms = _elapsed_ms(llm_start)
        llm_total_ms = round(service_wait_ms, 2)
        generation_ms = (
            round(max(0.0, llm_total_ms - first_output_ms), 2)
            if first_output_ms is not None
            else None
        )
        tokens_per_second = None
        if completion_tokens is not None and generation_ms and generation_ms > 0:
            tokens_per_second = round(completion_tokens / (generation_ms / 1000), 2)
        metrics = list(state["llm_call_metrics"])
        call_metrics = {
            "call_index": len(metrics) + 1,
            "turn": turn + 1,
            "purpose": purpose,
            "total_ms": llm_total_ms,
            "stream_wall_ms": stream_wall_ms,
            "first_output_ms": first_output_ms,
            "first_text_ms": first_text_ms,
            "generation_ms": generation_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "input_messages": len(messages),
            "input_characters": input_characters,
            "tool_schema_characters": schema_characters,
            "response_characters": len(assistant_text),
            "proposed_tool_calls": len(tool_calls),
            "stream_chunks": stream_chunks,
        }
        metrics.append(call_metrics)
        trace.record(
            "llm_decision",
            "model_response",
            meaning=(
                "LLM 完成本轮工具决策。"
                if purpose == "tool_decision"
                else "LLM 完成本轮文本回复。"
            ),
            turn=turn + 1,
            purpose=purpose,
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            metrics=call_metrics,
        )
        next_turn = turn + 1
        if not tool_calls:
            if openai_tools and not attempted_stage:
                messages.append(_assistant_msg(assistant_text))
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "你刚才没有调用当前工作流要求的工具。禁止用文字假装已经完成。"
                            f"本阶段只允许调用：{'、'.join(allowed)}。"
                        ),
                    }
                )
                writer(
                    {
                        "type": "tool_log",
                        "line": "模型未调用工具（疑似幻觉），强制重试中…",
                    }
                )
                writer({"type": "retry", "reason": "no_tool_call"})
                trace.record(
                    "llm_decision",
                    "required_tool_missing",
                    meaning="LLM 没有执行当前工作流要求的工具，Agent 将重试。",
                    turn=next_turn,
                    allowed_tools=allowed,
                )
                return {
                    "messages": messages,
                    "llm_call_metrics": metrics,
                    "turn": next_turn,
                    "next_action": (
                        "call_model" if next_turn < max_turns else "finalize"
                    ),
                }
            messages.append(_assistant_msg(assistant_text))
            return {
                "messages": messages,
                "llm_call_metrics": metrics,
                "final_reply": assistant_text,
                "turn": next_turn,
                "termination_reason": (
                    "final_response" if assistant_text.strip() else "empty_final_response"
                ),
                "next_action": "finalize",
            }

        messages.append(_assistant_msg(assistant_text, tool_calls))
        return {
            "messages": messages,
            "llm_call_metrics": metrics,
            "pending_tool_calls": tool_calls,
            "turn": next_turn,
            "next_action": "execute_tools",
        }

    async def _execute_tools_node(
        self, state: AgentState, writer: StreamWriter
    ) -> dict[str, Any]:
        messages = list(state["messages"])
        tool_logs = list(state["tool_logs"])
        history = list(state["tool_call_history"])
        completed = set(state["completed_actions"])
        seen = set(state["seen_tool_calls"])
        evidence = {
            action: list(items) for action, items in state["action_evidence"].items()
        }
        allowed = list(state["allowed_tool_names"])
        force_finalize = bool(state["force_finalize"])
        note_added = bool(state["finalization_note_added"])
        trace = state["trace"]
        turn = int(state["turn"])

        for call_index, tool_call in enumerate(state.get("pending_tool_calls", [])):
            function = tool_call.get("function", {})
            tool_name = str(function.get("name", ""))
            raw_arguments = function.get("arguments", "{}")
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
            elif tool_name not in allowed:
                rejection_reason = (
                    f"工具 {tool_name} 不属于当前工作流阶段；"
                    f"允许的工具：{'、'.join(allowed) or '无'}"
                )
                rejection_code = "tool_outside_allowed_set"
            elif self.tools.get(tool_name) is None:
                rejection_reason = f"未知工具: {tool_name}"
                rejection_code = "unknown_tool"
            signature = (
                tool_name,
                json.dumps(tool_args, sort_keys=True, ensure_ascii=False),
            )
            if not rejection_reason and signature in seen:
                rejection_reason = "本次请求内检测到相同工具和相同参数的重复调用"
                rejection_code = "duplicate_tool_call"

            if rejection_reason:
                result_obj = {"status": "error", "error": rejection_reason}
                result_json = json.dumps(result_obj, ensure_ascii=False)
                messages.append(
                    _tool_result_msg(tool_call["id"], tool_name, result_json)
                )
                history.append(
                    {
                        "turn": turn,
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
                    turn=turn,
                    tool=tool_name,
                    args=tool_args,
                    decision_valid=False,
                    rejection_code=rejection_code,
                    reason=rejection_reason,
                )
                tool_logs.append(rejection_reason)
                writer({"type": "tool_call", "name": tool_name, "args": tool_args})
                writer({"type": "tool_log", "line": rejection_reason})
                writer({"type": "tool_result", "name": tool_name, "result": result_obj})
                continue

            seen.add(signature)
            log_line = f"触发技能 {tool_name}，参数: {tool_args}"
            tool_logs.append(log_line)
            writer({"type": "tool_call", "name": tool_name, "args": tool_args})
            writer({"type": "tool_log", "line": log_line})
            tool_run_id = str(uuid.uuid4())
            loop_runtime.report_to_console(
                agent_id=loop_runtime._agent_id_for_tool(tool_name),
                run_id=tool_run_id,
                session_id=state["session_id"],
                phase="tool_started",
                task=state["instruction"],
                tool_name=tool_name,
            )
            tool_start = time.perf_counter()
            result_json = await self.tools.call(tool_name, tool_args)
            tool_elapsed = _elapsed_ms(tool_start)
            try:
                result_obj = json.loads(result_json)
            except json.JSONDecodeError:
                result_obj = {"status": "unknown", "raw": result_json}
            new_actions = completed_actions_for_result(
                tool_name, tool_args, result_obj
            )
            completed.update(new_actions)
            for action in sorted(new_actions):
                evidence.setdefault(action, []).append(
                    {
                        "turn": turn,
                        "tool": tool_name,
                        "raw_status": result_obj.get("status", "unknown"),
                    }
                )
            execution_status = _tool_effect_status(
                result_obj.get("status"), new_actions
            )
            result_log = (
                f"技能 {tool_name} 执行耗时 {tool_elapsed:.0f} ms，"
                f"结果 {result_obj.get('status', 'unknown')}。"
            )
            tool_logs.append(result_log)
            writer({"type": "tool_log", "line": result_log})
            writer(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "result": result_obj,
                    "elapsed_ms": tool_elapsed,
                }
            )
            loop_runtime.report_to_console(
                agent_id=loop_runtime._agent_id_for_tool(tool_name),
                run_id=tool_run_id,
                session_id=state["session_id"],
                phase="tool_completed",
                task=state["instruction"],
                tool_name=tool_name,
                detail=(
                    f"耗时 {tool_elapsed:.0f}ms, "
                    f"状态 {result_obj.get('status', 'unknown')}"
                ),
            )
            self._flush_pending_completion()
            history.append(
                {
                    "turn": turn,
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
                turn=turn,
                tool=tool_name,
                args=tool_args,
                raw_status=result_obj.get("status", "unknown"),
                execution_status=execution_status,
                decision_valid=True,
                elapsed_ms=tool_elapsed,
                result=result_obj,
                new_actions=sorted(new_actions),
            )
            messages.append(
                _tool_result_msg(tool_call["id"], tool_name, result_json)
            )
            if tool_name == "search_sdr_knowledge":
                allowed = [] if "answer_with_knowledge" in new_actions else ["web_search"]
            elif tool_name == "web_search":
                allowed = []
                force_finalize = True

        plan = state["plan"]
        route = state["route"]
        required = set(plan.required_actions)
        hardware_actions = required - {"answer_with_knowledge"}
        if route.route == "hardware" and hardware_actions.issubset(completed):
            if (
                "answer_with_knowledge" in required
                and "answer_with_knowledge" not in completed
            ):
                if not force_finalize and allowed != ["web_search"]:
                    allowed = ["search_sdr_knowledge"]
            else:
                allowed = []
                force_finalize = True
        elif route.route == "knowledge" and required.issubset(completed):
            allowed = []
            force_finalize = True

        if force_finalize and not note_added:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "工具阶段已经结束。下一轮禁止调用任何工具；"
                        "请只根据用户原始请求和真实工具结果生成最终中文答复。"
                    ),
                }
            )
            note_added = True
        elif completed and required:
            missing = sorted(required - completed)
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
                if required.issubset(completed)
                else "当前业务尚未完成，Agent 根据缺失动作决定是否继续循环。"
            ),
            turn=turn,
            allowed_tools=allowed,
            required_actions=sorted(required),
            completed_actions=sorted(completed),
            missing_actions=sorted(required - completed),
            actions_satisfied=required.issubset(completed),
            force_finalize=force_finalize,
        )
        return {
            "messages": messages,
            "tool_logs": tool_logs,
            "tool_call_history": history,
            "completed_actions": completed,
            "seen_tool_calls": seen,
            "action_evidence": evidence,
            "allowed_tool_names": allowed,
            "force_finalize": force_finalize,
            "finalization_note_added": note_added,
            "pending_tool_calls": [],
            "next_action": (
                "call_model" if int(state["turn"]) < int(state["max_turns"])
                else "finalize"
            ),
        }

    async def _finalize_node(
        self, state: AgentState, writer: StreamWriter
    ) -> dict[str, Any]:
        plan = state["plan"]
        route = state["route"]
        trace = state["trace"]
        messages = list(state["messages"])
        history = list(state["tool_call_history"])
        completed = set(state["completed_actions"])
        required = set(plan.required_actions)
        final_reply = state["final_reply"]
        termination_reason = state["termination_reason"]
        successful_calls = [call for call in history if call.get("status") == "success"]
        has_final_reply = bool(final_reply.strip())
        business_complete = (
            route.route == "chat" and has_final_reply
        ) or (
            bool(required) and required.issubset(completed) and has_final_reply
        )
        if plan.needs_clarification and has_final_reply:
            task_status = "incomplete"
            termination_reason = "clarification_required"
        elif business_complete:
            task_status = "success"
        elif has_final_reply and not successful_calls and history:
            task_status = "failed"
            termination_reason = "all_tool_calls_failed"
        elif has_final_reply:
            task_status = "incomplete"
            termination_reason = "business_actions_missing"
        elif completed or successful_calls:
            task_status = "incomplete"
            termination_reason = (
                "final_response_missing"
                if required.issubset(completed)
                else "max_turns_reached"
            )
            missing = sorted(required - completed)
            missing_note = f"，仍缺少：{'、'.join(missing)}" if missing else ""
            final_reply = (
                f"已有 {len(successful_calls)} 次工具调用执行成功，但任务未完整结束"
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
                f"任务未完成：Agent 在 {state['max_turns']} 轮内没有生成有效答复，"
                "也没有成功的工具调用。请查看工具日志后重试。"
            )
        if not has_final_reply:
            messages.append(_assistant_msg(final_reply))

        clean_history: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str):
                if role == "assistant" and message.get("tool_calls"):
                    continue
                if content.strip():
                    clean_history.append({"role": role, "content": content})
        self.memory.replace(state["session_id"], clean_history)

        total_ms = _elapsed_ms(state["request_started_at"])
        rejected = [call for call in history if call.get("rejected", False)]
        actual_calls = [call for call in history if not call.get("rejected", False)]
        metrics = state["llm_call_metrics"]
        llm_total_ms = round(sum(float(call.get("total_ms") or 0) for call in metrics), 2)
        llm_wall_ms = round(
            sum(float(call.get("stream_wall_ms") or 0) for call in metrics), 2
        )
        stream_delivery_ms = round(max(0.0, llm_wall_ms - llm_total_ms), 2)
        tool_execution_ms = round(
            sum(float(call.get("elapsed_ms") or 0) for call in actual_calls), 2
        )
        knowledge_lookup_ms = round(
            sum(
                float(call.get("elapsed_ms") or 0)
                for call in actual_calls
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
                for call in metrics
                if call.get("purpose") == "tool_decision"
            ),
            2,
        )
        final_reply_ms = round(
            sum(
                float(call.get("total_ms") or 0)
                for call in metrics
                if call.get("purpose") != "tool_decision"
            ),
            2,
        )
        final_text_calls = [
            call
            for call in metrics
            if call.get("purpose") != "tool_decision"
            and call.get("first_text_ms") is not None
        ]
        first_token_ms = (
            final_text_calls[-1]["first_text_ms"] if final_text_calls else None
        )
        overhead_ms = round(max(0.0, total_ms - llm_wall_ms - tool_execution_ms), 2)
        actions_satisfied = (
            required.issubset(completed) if required else route.route == "chat"
        )
        completion = {
            "actions_satisfied": actions_satisfied,
            "model_final_reply_present": has_final_reply,
            "visible_reply_present": bool(final_reply.strip()),
            "task_satisfied": business_complete,
            "required_actions": sorted(required),
            "completed_actions": sorted(completed),
            "missing_actions": sorted(required - completed),
            "action_evidence": state["action_evidence"],
        }
        trace.metrics = {
            "timing": {
                "total_ms": total_ms,
                "pre_agent_ms": state["pre_agent_ms"],
                "memory_read_ms": state["memory_read_ms"],
                "route_ms": state["route_ms"],
                "plan_build_ms": state["plan_build_ms"],
                "context_build_ms": state["context_build_ms"],
                "llm_total_ms": llm_total_ms,
                "llm_stream_wall_ms": llm_wall_ms,
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
                "turns_used": state["turn"],
                "llm_calls": len(metrics),
                "tool_calls_proposed": sum(
                    int(call.get("proposed_tool_calls") or 0) for call in metrics
                ),
                "tool_calls_executed": len(actual_calls),
                "tool_calls_rejected": len(rejected),
                "duplicate_call_count": sum(
                    call.get("rejection_code") == "duplicate_tool_call"
                    for call in rejected
                ),
            },
            "llm_calls": metrics,
        }
        diagnosis = build_diagnosis(
            status=task_status,
            termination_reason=termination_reason,
            route=route.route,
            skill_name=plan.skill.name if plan.skill else None,
            tool_calls=history,
            required_actions=required,
            completed_actions=completed,
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
            tool_calls=history,
            required_actions=sorted(required),
            completed_actions=sorted(completed),
            missing_actions=sorted(required - completed),
            completion=completion,
            diagnosis=diagnosis,
        )
        if self.trace_store is not None:
            self.trace_store.append(trace)

        tools_executed = len(successful_calls)
        if self.tools.is_hardware_busy():
            self._pending_console_run_id = state["run_id"]
            self._pending_console_session_id = state["session_id"]
            self._pending_console_instruction = state["instruction"]
            logger.info(
                "Deferred console completion for run %s — USRP still active",
                state["run_id"],
            )
        else:
            loop_runtime.report_to_console(
                agent_id="usrp",
                run_id=state["run_id"],
                session_id=state["session_id"],
                phase="completed",
                task=state["instruction"],
                tool_name="",
                detail=(
                    f"执行 {tools_executed}/{len(history)} 个工具, "
                    f"总耗时 {total_ms:.0f}ms"
                ),
            )
        writer(
            {
                "type": "done",
                "run_id": state["run_id"],
                "status": task_status,
                "termination_reason": termination_reason,
                "diagnosis": diagnosis,
                "reply": final_reply,
                "hardware_logs": state["tool_logs"],
                "updated_history": clean_history,
                "active_mode": state["selected_mode"],
                "active_model": state["active_model"],
                "active_temperature": state["active_temperature"],
                "active_route": route.route,
                "route_confidence": route.confidence,
                "skill_matched": plan.skill.name if plan.skill else None,
                "workflow": plan.workflow,
                "required_actions": sorted(required),
                "completed_actions": sorted(completed),
                "tools_executed": tools_executed,
                "tools_total": len(history),
                "orchestrator": "langgraph",
                "timing": {
                    "total_ms": total_ms,
                    "pre_agent_ms": state["pre_agent_ms"],
                    "route": route.route,
                    "memory_read_ms": state["memory_read_ms"],
                    "route_ms": state["route_ms"],
                    "plan_build_ms": state["plan_build_ms"],
                    "context_build_ms": state["context_build_ms"],
                    "llm_total_ms": llm_total_ms,
                    "llm_stream_wall_ms": llm_wall_ms,
                    "stream_delivery_ms": stream_delivery_ms,
                    "knowledge_lookup_ms": knowledge_lookup_ms,
                    "tool_decision_ms": tool_decision_ms,
                    "tool_execution_ms": tool_execution_ms,
                    "hardware_execution_ms": hardware_execution_ms,
                    "final_reply_ms": final_reply_ms,
                    "first_token_ms": first_token_ms,
                    "agent_overhead_ms": overhead_ms,
                    "tool_count": len(history),
                    "tool_calls": history,
                    "turns": state["turn"],
                },
            }
        )
        return {
            "messages": messages,
            "final_reply": final_reply,
            "termination_reason": termination_reason,
        }


__all__ = ["AgentLoop", "AgentState"]
