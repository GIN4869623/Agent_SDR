"""Persistent, structured work logs for Agent runs.

Each completed run is stored as one JSON object per line.  The log is meant
for diagnosis rather than model context: large tool payloads are compacted so
waveform arrays and long retrieval results cannot make the file explode.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_value(
    value: Any,
    *,
    max_text: int = 2000,
    max_items: int = 20,
    depth: int = 0,
) -> Any:
    """Return a JSON-safe, bounded representation of a diagnostic value."""
    if depth >= 8:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= max_text:
            return value
        return value[:max_text] + f"…<truncated {len(value) - max_text} chars>"
    if isinstance(value, dict):
        items = list(value.items())
        compacted = {
            str(key): compact_value(
                item, max_text=max_text, max_items=max_items, depth=depth + 1
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            compacted["_truncated_keys"] = len(items) - max_items
        return compacted
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        compacted = [
            compact_value(
                item, max_text=max_text, max_items=max_items, depth=depth + 1
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            compacted.append(f"<truncated {len(items) - max_items} items>")
        return compacted
    if hasattr(value, "tolist"):
        try:
            return compact_value(
                value.tolist(), max_text=max_text, max_items=max_items, depth=depth
            )
        except Exception:
            pass
    return compact_value(
        str(value), max_text=max_text, max_items=max_items, depth=depth + 1
    )


def build_diagnosis(
    *,
    status: str,
    termination_reason: str,
    route: str,
    skill_name: str | None,
    tool_calls: list[dict[str, Any]],
    required_actions: set[str],
    completed_actions: set[str],
) -> dict[str, Any]:
    """Map an outcome to the most likely layer and useful source files."""
    rejected = [call for call in tool_calls if call.get("rejected")]
    errors = " ".join(
        str(call.get("error", "")) for call in tool_calls if call.get("error")
    )
    missing = sorted(required_actions - completed_actions)

    if status == "success":
        return {
            "stage": "completed",
            "diagnosis_code": "task_completed",
            "suspected_component": None,
            "retryable": False,
            "summary": "路由、工作流、工具执行和最终回复均满足当前完成条件。",
            "suggested_files": [],
        }
    if termination_reason == "clarification_required":
        return {
            "stage": "skill_selection",
            "diagnosis_code": "skill_requires_clarification",
            "suspected_component": "router_or_skill_rules",
            "retryable": False,
            "summary": "硬件意图已识别，但没有足够信息确定具体 Skill。",
            "suggested_files": ["src/agent/router.py", "skill/skills/*.md"],
        }
    if rejected and "JSON" in errors:
        return {
            "stage": "argument_validation",
            "diagnosis_code": "invalid_tool_arguments",
            "suspected_component": "llm_or_tool_schema",
            "retryable": True,
            "summary": "LLM 生成的工具参数格式无效。",
            "suggested_files": ["skill/skills/*.md", "tools/", "src/mcp/adapter.py"],
        }
    if rejected and "当前工作流" in errors:
        return {
            "stage": "tool_selection",
            "diagnosis_code": "tool_outside_workflow",
            "suspected_component": "llm_or_skill_constraints",
            "retryable": True,
            "summary": "LLM 选择了当前 Skill 工作流不允许的工具。",
            "suggested_files": ["src/agent/router.py", "skill/skills/*.md"],
        }
    if rejected and "重复调用" in errors:
        return {
            "stage": "loop_control",
            "diagnosis_code": "duplicate_tool_call",
            "suspected_component": "llm_or_agent_loop",
            "retryable": False,
            "summary": "同一请求内出现了相同工具和参数的重复调用。",
            "suggested_files": ["src/agent/loop.py", "skill/skills/*.md"],
        }
    if termination_reason == "all_tool_calls_failed":
        return {
            "stage": "tool_execution",
            "diagnosis_code": "all_tool_calls_failed",
            "suspected_component": "tool_mcp_or_hardware",
            "retryable": True,
            "summary": "工具已选定，但所有实际调用均执行失败。",
            "suggested_files": ["tools/", "hardware/"],
        }
    if missing:
        return {
            "stage": "completion_check",
            "diagnosis_code": "required_actions_missing",
            "suspected_component": "llm_or_completion_rules",
            "retryable": True,
            "summary": f"任务缺少业务动作：{'、'.join(missing)}。",
            "suggested_files": ["src/agent/router.py", "skill/skills/*.md"],
        }
    if termination_reason in {"final_response_missing", "empty_final_response"}:
        return {
            "stage": "final_response",
            "diagnosis_code": "final_response_missing",
            "suspected_component": "llm_finalization_or_agent_loop",
            "retryable": True,
            "summary": "业务动作可能已经完成，但 LLM 没有生成有效最终回复。",
            "suggested_files": ["src/agent/loop.py", "src/prompts.py"],
        }
    return {
        "stage": "agent_loop",
        "diagnosis_code": termination_reason,
        "suspected_component": "agent_loop",
        "retryable": True,
        "summary": f"任务在 {route} 路由中以 {termination_reason} 结束。",
        "suggested_files": ["src/agent/loop.py", "src/agent/router.py"],
        "skill": skill_name,
    }


@dataclass
class RunTrace:
    run_id: str
    session_id: str
    instruction: str
    mode: str
    model: str
    temperature: float
    started_at: str = field(default_factory=_utc_now)
    request: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    final: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        event: str,
        *,
        meaning: str | None = None,
        **details: Any,
    ) -> None:
        item = {
            "at": _utc_now(),
            "stage": stage,
            "event": event,
            "details": compact_value(details),
        }
        if meaning:
            item["meaning"] = meaning
        self.events.append(item)

    def finish(self, **final: Any) -> None:
        self.final = {"finished_at": _utc_now(), **compact_value(final)}

    def to_dict(self) -> dict[str, Any]:
        return compact_value(
            {
                "schema_version": 2,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "instruction": self.instruction,
                "mode": self.mode,
                "model": self.model,
                "temperature": self.temperature,
                "started_at": self.started_at,
                "request": self.request,
                "route": self.route,
                "plan": self.plan,
                "events": self.events,
                "metrics": self.metrics,
                "final": self.final,
            },
            max_text=4000,
            max_items=200,
        )


class RunTraceStore:
    """Thread-safe rolling JSONL storage.

    The newest ``max_records`` runs are retained. Logging or retention failures
    never fail an Agent task.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        max_records: int = 1000,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.max_records = max(1, int(max_records))
        cleanup_batch = max(1, min(100, self.max_records // 10))
        self.trim_to_records = max(1, self.max_records - cleanup_batch)
        self._lock = Lock()
        self._record_count: int | None = None

    def _count_records_locked(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _trim_locked(self) -> None:
        """Atomically discard oldest lines while the caller holds the lock."""
        with self.path.open("r", encoding="utf-8") as handle:
            newest = deque(
                (line for line in handle if line.strip()),
                maxlen=self.trim_to_records,
            )

        temporary_path = self.path.with_name(f".{self.path.name}.retention.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                handle.writelines(newest)
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._record_count = len(newest)

    def append(self, trace: RunTrace) -> None:
        if not self.enabled:
            return
        try:
            line = json.dumps(trace.to_dict(), ensure_ascii=False) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._record_count is None:
                    self._record_count = self._count_records_locked()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                self._record_count += 1
                if self._record_count > self.max_records:
                    self._trim_locked()
        except Exception:
            logger.warning("Failed to persist agent run trace", exc_info=True)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled or not self.path.exists():
            return []
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                lines = deque(handle, maxlen=safe_limit)
        records: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not self.enabled or not self.path.exists():
            return None
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("run_id") == run_id:
                        return record
        return None
