"""Evaluate a real local LLM against the current Agent Loop with safe mock tools.

The script uses the configured OpenAI-compatible endpoint and the production
AgentLoop/router/Skills/tool schemas.  Tool execution is intercepted, so no
USRP, VLC, web search or other external side effect can occur.

Examples:
    .venv/bin/python scripts/evaluate_model_compatibility.py --mode fast
    .venv/bin/python scripts/evaluate_model_compatibility.py --mode both
    .venv/bin/python scripts/evaluate_model_compatibility.py --model qwen3.5:35b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.memory import ConversationMemory
from skill.skills import build_all_skills
from src.agent.loop import AgentLoop
from src.agent.trace import RunTraceStore
from src.config import settings
from tools import ToolRegistry
from tools.registry import create_tool_registry


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    title: str
    instruction: str
    expected_tools: tuple[str, ...]
    expected_status: str
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_args: tuple[dict[str, Any], ...] = ()
    reply_markers: tuple[str, ...] = ()


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="chat_no_tool",
        title="普通对话不调用工具",
        instruction="你好，请用一句话介绍你自己。",
        expected_tools=(),
        expected_status="success",
    ),
    EvalCase(
        case_id="scan_frequency_units",
        title="扫频工具与单位换算",
        instruction="请扫描 2.45 GHz 附近 2 MHz 带宽的底噪。",
        expected_tools=("perform_physical_scan",),
        expected_status="success",
        expected_args=({"center_freq_hz": 2_450_000_000.0, "bandwidth_hz": 2_000_000.0},),
        tool_results={
            "perform_physical_scan": {
                "status": "success",
                "center_freq_hz": 2_450_000_000.0,
                "bandwidth_hz": 2_000_000.0,
                "rx_power_db": -73.25,
                "message": "评测模拟扫描完成，标记 S73。",
            }
        },
        reply_markers=("S73", "-73.25"),
    ),
    EvalCase(
        case_id="fixed_qpsk_parameters",
        title="中文文本与QPSK参数提取",
        instruction="在 2.45 GHz 用 QPSK 发送“兼容性测试”，收发增益都设为 25 dB。",
        expected_tools=("text_fsk_send_and_receive",),
        expected_status="success",
        expected_args=(
            {
                "text": "兼容性测试",
                "modulation_scheme": "QPSK",
                "center_freq_hz": 2_450_000_000.0,
                "tx_gain": 25.0,
                "rx_gain": 25.0,
            },
        ),
        tool_results={
            "text_fsk_send_and_receive": {
                "status": "success",
                "decoded_ok": True,
                "decoded_text": "兼容性测试",
                "message": "评测模拟收发完成，标记 TX25。",
            }
        },
        reply_markers=("TX25", "兼容性测试"),
    ),
    EvalCase(
        case_id="adaptive_transmit",
        title="自适应调制意图",
        instruction="请在 2.42 GHz 根据信道 SNR 自动选择调制方式发送“链路评测”。",
        expected_tools=("adaptive_modulation_transmit",),
        expected_status="success",
        expected_args=({"text": "链路评测", "center_freq_hz": 2_420_000_000.0},),
        tool_results={
            "adaptive_modulation_transmit": {
                "status": "success",
                "probe_snr_db": 12.5,
                "selected_modulation": "QPSK",
                "decoded_ok": True,
                "message": "评测模拟自适应发送完成，标记 AMC12。",
            }
        },
        reply_markers=("AMC12", "12.5"),
    ),
    EvalCase(
        case_id="cognitive_tone",
        title="认知选频复合参数",
        instruction="寻找 2.4 GHz 附近最干净的频点，并启动 80 kHz 单音频谱可视化，不要发送文本。",
        expected_tools=("auto_optimal_transmit",),
        expected_status="success",
        expected_args=(
            {
                "text": "",
                "center_freq_hz": 2_400_000_000.0,
                "enable_tone": True,
                "tone_freq_hz": 80_000.0,
            },
        ),
        tool_results={
            "auto_optimal_transmit": {
                "status": "success",
                "selected_center_freq_hz": 2_407_500_000.0,
                "tone_result": {"status": "success"},
                "message": "评测模拟认知选频完成，标记 COG80。",
            }
        },
        reply_markers=("COG80", "2.4075"),
    ),
    EvalCase(
        case_id="knowledge_grounding",
        title="本地知识检索与结果忠实度",
        instruction="什么是 QPSK？请根据知识库回答。",
        expected_tools=("search_sdr_knowledge",),
        expected_status="success",
        expected_args=({"query": {"$contains": "QPSK"}},),
        tool_results={
            "search_sdr_knowledge": {
                "status": "success",
                "results": [
                    {
                        "source": "eval_fixture.md",
                        "snippet": "QPSK 每个符号携带 2 比特。评测标记 K42。",
                        "score": 0.99,
                    }
                ],
            }
        },
        reply_markers=("K42", "2 比特"),
    ),
    EvalCase(
        case_id="knowledge_web_fallback",
        title="本地未命中后的两阶段回退",
        instruction="请查询罕见术语 Zeta-SDR-17 的含义。",
        expected_tools=("search_sdr_knowledge", "web_search"),
        expected_status="success",
        expected_args=(
            {"query": {"$contains": "Zeta-SDR-17"}},
            {"query": {"$contains": "Zeta-SDR-17"}},
        ),
        tool_results={
            "search_sdr_knowledge": {"status": "success", "results": []},
            "web_search": {
                "status": "success",
                "results": [
                    {
                        "title": "评测模拟网络资料",
                        "url": "https://example.invalid/eval",
                        "snippet": "这是模拟回退结果，评测标记 W17。",
                    }
                ],
            },
        },
        reply_markers=("W17",),
    ),
    EvalCase(
        case_id="tool_error_truthfulness",
        title="工具失败后如实回答",
        instruction="请扫描 2.4 GHz 底噪。",
        expected_tools=("perform_physical_scan",),
        expected_status="failed",
        tool_results={
            "perform_physical_scan": {
                "status": "error",
                "error": "评测模拟：USRP 连接失败，标记 ERR9。",
            }
        },
        reply_markers=("ERR9", "失败"),
    ),
)


class _SafeController:
    """Shape-only dependency for obtaining production MCP tool schemas."""

    def has_background_task(self) -> bool:
        return False

    def __getattr__(self, name: str):
        raise RuntimeError(f"兼容性评测禁止执行真实硬件方法：{name}")


class _SafeKnowledgeBase:
    def search(self, *_args, **_kwargs):
        raise RuntimeError("兼容性评测禁止执行真实知识检索")

    def query(self, *_args, **_kwargs):
        raise RuntimeError("兼容性评测禁止执行真实知识检索")


class _SafeVideoController:
    def __getattr__(self, name: str):
        raise RuntimeError(f"兼容性评测禁止执行真实视频方法：{name}")


def _production_schema_registry() -> ToolRegistry:
    """Create current production schemas without constructing real hardware."""
    return create_tool_registry(
        _SafeController(),
        _SafeKnowledgeBase(),
        _SafeVideoController(),
    )


class MockToolRegistry:
    """Expose production schemas while returning deterministic mock results."""

    def __init__(self, schema_registry: ToolRegistry, case: EvalCase) -> None:
        self.schema_registry = schema_registry
        self.case = case
        self.calls: list[dict[str, Any]] = []

    def as_openai_tools(self, category=None, names=None):
        return self.schema_registry.as_openai_tools(category=category, names=names)

    def get(self, name: str):
        return self.schema_registry.get(name)

    async def call(self, name: str, args: dict) -> str:
        self.calls.append({"name": name, "args": dict(args)})
        result = self.case.tool_results.get(
            name,
            {"status": "error", "error": f"评测案例没有为 {name} 配置模拟结果"},
        )
        return json.dumps(result, ensure_ascii=False)

    def is_hardware_busy(self) -> bool:
        return False


def _normalized_text(value: Any) -> str:
    return "".join(str(value).lower().split())


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and "$contains" in expected:
        return _normalized_text(expected["$contains"]) in _normalized_text(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        tolerance = max(1e-6, abs(float(expected)) * 1e-6)
        return abs(float(actual) - float(expected)) <= tolerance
    if isinstance(expected, str):
        return _normalized_text(actual) == _normalized_text(expected)
    return actual == expected


def _score_case(
    case: EvalCase,
    done: dict[str, Any] | None,
    calls: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    if error or done is None:
        return {
            "score": 0.0,
            "passed": False,
            "checks": {"protocol": False},
            "error": error or "Agent Loop 没有产生 done 事件",
        }

    actual_tools = tuple(call["name"] for call in calls)
    tool_sequence_ok = actual_tools == case.expected_tools

    arg_checks: list[dict[str, Any]] = []
    for index, expected_subset in enumerate(case.expected_args):
        actual_args = calls[index]["args"] if index < len(calls) else {}
        fields = {
            key: _values_match(actual_args.get(key), expected)
            for key, expected in expected_subset.items()
        }
        arg_checks.append({"expected": expected_subset, "actual": actual_args, "fields": fields})
    args_ok = all(all(item["fields"].values()) for item in arg_checks)

    status_ok = done.get("status") == case.expected_status
    reply = str(done.get("reply", ""))
    reply_present = bool(reply.strip())
    marker_ok = not case.reply_markers or any(marker in reply for marker in case.reply_markers)

    score = (
        35.0 * tool_sequence_ok
        + 30.0 * args_ok
        + 20.0 * status_ok
        + 5.0 * reply_present
        + 10.0 * marker_ok
    )
    checks = {
        "protocol": True,
        "tool_sequence": tool_sequence_ok,
        "arguments": args_ok,
        "task_status": status_ok,
        "reply_present": reply_present,
        "result_grounding": marker_ok,
    }
    return {
        "score": score,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_tools": list(case.expected_tools),
        "actual_tools": list(actual_tools),
        "argument_checks": arg_checks,
        "expected_status": case.expected_status,
        "actual_status": done.get("status"),
        "termination_reason": done.get("termination_reason"),
        "diagnosis": done.get("diagnosis"),
        "reply": reply,
        "timing": done.get("timing"),
    }


async def _run_case(
    client: OpenAI,
    schema_registry: ToolRegistry,
    case: EvalCase,
    mode: str,
    temperature: float,
) -> dict[str, Any]:
    tools = MockToolRegistry(schema_registry, case)
    with tempfile.TemporaryDirectory() as temp_dir:
        trace_store = RunTraceStore(Path(temp_dir) / "trace.jsonl")
        agent = AgentLoop(
            llm_client=client,
            tool_registry=tools,  # type: ignore[arg-type]
            skill_registry=build_all_skills(),
            memory=ConversationMemory(),
            trace_store=trace_store,
        )
        done: dict[str, Any] | None = None
        error: str | None = None
        started = time.perf_counter()
        try:
            with patch("src.agent.loop.report_to_console"):
                async for event in agent.run_stream(
                    instruction=case.instruction,
                    session_id=f"eval-{mode}-{case.case_id}",
                    mode=mode,
                    temperature=temperature,
                ):
                    if event.get("type") == "done":
                        done = event
                    elif event.get("type") == "error":
                        error = str(event.get("message", "未知 Agent 错误"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        result = _score_case(case, done, tools.calls, error)
        result.update(
            {
                "case_id": case.case_id,
                "title": case.title,
                "instruction": case.instruction,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return result


def _grade(score: float, pass_rate: float, protocol_ok: bool) -> str:
    if not protocol_ok:
        return "协议不兼容"
    if score >= 90 and pass_rate >= 0.95:
        return "高度兼容"
    if score >= 80 and pass_rate >= 0.8:
        return "基本兼容"
    if score >= 70:
        return "有限兼容，需要优化提示词或 Schema"
    return "兼容性不足"


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.model:
        modes = ("custom",)
        model_by_mode = {"custom": args.model}
    else:
        modes = ("think", "fast") if args.mode == "both" else (args.mode,)
        model_by_mode = {
            "think": settings.model_think,
            "fast": settings.model_fast,
        }

    selected_cases = [case for case in CASES if not args.case or case.case_id in args.case]
    unknown_cases = set(args.case or ()) - {case.case_id for case in CASES}
    if unknown_cases:
        raise SystemExit(f"未知案例：{', '.join(sorted(unknown_cases))}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    schema_registry = _production_schema_registry()
    report: dict[str, Any] = {
        "schema_version": 1,
        "base_url": args.base_url,
        "safe_mock_tools": True,
        "repetitions": args.repeat,
        "models": [],
    }

    for requested_mode in modes:
        model = model_by_mode[requested_mode]
        loop_mode = "think" if requested_mode == "custom" else requested_mode
        original_model = settings.model_think if loop_mode == "think" else settings.model_fast
        if loop_mode == "think":
            settings.model_think = model
        else:
            settings.model_fast = model

        results: list[dict[str, Any]] = []
        try:
            for repetition in range(1, args.repeat + 1):
                for case in selected_cases:
                    result = await _run_case(
                        client,
                        schema_registry,
                        case,
                        loop_mode,
                        args.temperature,
                    )
                    result["repetition"] = repetition
                    results.append(result)
                    state = "PASS" if result["passed"] else "FAIL"
                    print(
                        f"[{state}] {model} | {case.case_id} | "
                        f"{result['score']:.0f}分 | {result['elapsed_ms']:.0f}ms",
                        flush=True,
                    )
        finally:
            if loop_mode == "think":
                settings.model_think = original_model
            else:
                settings.model_fast = original_model

        protocol_ok = all(item["checks"].get("protocol", False) for item in results)
        score = round(sum(item["score"] for item in results) / len(results), 2) if results else 0.0
        pass_rate = round(sum(item["passed"] for item in results) / len(results), 4) if results else 0.0
        elapsed_values = [float(item["elapsed_ms"]) for item in results]
        report["models"].append(
            {
                "mode": requested_mode,
                "model": model,
                "protocol_compatible": protocol_ok,
                "score": score,
                "pass_rate": pass_rate,
                "grade": _grade(score, pass_rate, protocol_ok),
                "latency_ms": {
                    "average": round(statistics.fmean(elapsed_values), 2) if elapsed_values else None,
                    "median": round(statistics.median(elapsed_values), 2) if elapsed_values else None,
                    "minimum": round(min(elapsed_values), 2) if elapsed_values else None,
                    "maximum": round(max(elapsed_values), 2) if elapsed_values else None,
                },
                "results": results,
            }
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="安全评测本地 LLM 与当前 Agent Loop 的兼容性")
    parser.add_argument("--mode", choices=("think", "fast", "both"), default="fast")
    parser.add_argument("--model", help="评测指定模型；设置后忽略 --mode 对应的模型名")
    parser.add_argument("--base-url", default=settings.ollama_base_url)
    parser.add_argument("--api-key", default=settings.api_key)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--case", action="append", help="只运行指定 case_id，可重复传入")
    parser.add_argument("--output", type=Path, help="将完整 JSON 报告写入指定路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat 必须大于等于 1")
    report = asyncio.run(evaluate(args))

    print("\n兼容性汇总")
    for item in report["models"]:
        print(
            f"- {item['model']} ({item['mode']}): {item['score']:.2f}分，"
            f"通过率 {item['pass_rate'] * 100:.1f}%，{item['grade']}，"
            f"平均 {item['latency_ms']['average'] / 1000:.2f}s/案例"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整报告：{args.output}")


if __name__ == "__main__":
    main()
