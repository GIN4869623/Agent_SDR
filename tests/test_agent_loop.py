from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import httpx
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from skill.skill_spec import SkillRegistry
from skill.skills import _load_skills
from src.agent.loop import AgentLoop
from src.agent.trace import RunTraceStore
from src.config import Settings
from src.llm import create_async_llm_client


def text_chunk(text: str):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=None))])


def tool_chunk(name: str, args: dict):
    call = NS(
        index=0,
        id=f"call-{name}",
        function=NS(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )
    return NS(choices=[NS(delta=NS(content=None, tool_calls=[call]))])


class FakeCompletions:
    def __init__(self, turns: list[list[object]]) -> None:
        self.turns = turns
        self.index = 0
        self.requests = []

    def create(self, **_kwargs):
        self.requests.append(_kwargs)
        turn = self.turns[self.index]
        self.index += 1
        return iter(turn)


class FakeLLM:
    def __init__(self, turns: list[list[object]]) -> None:
        self.chat = NS(completions=FakeCompletions(turns))


class GatedAsyncStream:
    """Hold the second chunk so tests can observe the first delta in flight."""

    def __init__(self) -> None:
        self.index = 0
        self.release = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index == 0:
            self.index += 1
            return text_chunk("流")
        if self.index == 1:
            await self.release.wait()
            self.index += 1
            return text_chunk("式")
        raise StopAsyncIteration


class FakeAsyncCompletions:
    def __init__(self, stream: GatedAsyncStream) -> None:
        self.stream = stream

    async def create(self, **_kwargs):
        return self.stream


class FakeAsyncLLM:
    def __init__(self, stream: GatedAsyncStream) -> None:
        self.chat = NS(completions=FakeAsyncCompletions(stream))


class FakeTools:
    def __init__(self, results: dict[str, dict]) -> None:
        self.results = results

    def as_openai_tools(self, category=None, names=None):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in (names or [])
        ]

    def get(self, _name: str):
        return object()

    async def call(self, name: str, _args: dict) -> str:
        return json.dumps(self.results[name], ensure_ascii=False)

    def is_hardware_busy(self) -> bool:
        return False


class FakeMemory:
    def __init__(self) -> None:
        self.value: list[dict[str, str]] = []

    def get_history(self, _session_id: str):
        return list(self.value)

    def replace(self, _session_id: str, value):
        self.value = list(value)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skills = SkillRegistry(list(_load_skills().values()))

    async def run_agent(self, instruction, turns, results, trace_store=None):
        agent = AgentLoop(
            FakeLLM(turns), FakeTools(results), self.skills, FakeMemory(), trace_store
        )
        with patch("src.agent.loop.report_to_console"):
            events = [
                event
                async for event in agent.run_stream(
                    instruction, "session", max_turns=4
                )
            ]
        return next(event for event in events if event["type"] == "done")

    def test_agent_is_backed_by_expected_langgraph_topology(self) -> None:
        agent = AgentLoop(FakeLLM([]), FakeTools({}), self.skills, FakeMemory())
        graph = agent.graph.get_graph()
        self.assertEqual(
            set(graph.nodes),
            {"__start__", "prepare", "call_model", "execute_tools", "finalize", "__end__"},
        )
        self.assertIn(("prepare", "call_model"), {(e.source, e.target) for e in graph.edges})

    async def test_async_model_yields_delta_before_response_finishes(self) -> None:
        model_stream = GatedAsyncStream()
        agent = AgentLoop(
            FakeAsyncLLM(model_stream), FakeTools({}), self.skills, FakeMemory()
        )
        events = agent.run_stream("你好", "stream-session")

        with patch("src.agent.loop.report_to_console"):
            while True:
                event = await asyncio.wait_for(anext(events), timeout=1)
                if event.get("type") == "delta":
                    break

            self.assertEqual(event["text"], "流")
            self.assertFalse(model_stream.release.is_set())

            model_stream.release.set()
            remaining = [item async for item in events]

        deltas = [item["text"] for item in remaining if item.get("type") == "delta"]
        self.assertEqual(deltas, ["式"])
        done = next(item for item in remaining if item.get("type") == "done")
        self.assertEqual(done["reply"], "流式")

    async def test_hardware_success_requires_tool_effect_and_final_reply(self) -> None:
        done = await self.run_agent(
            "帮我扫描 2.4 GHz 底噪",
            [
                [tool_chunk("perform_physical_scan", {})],
                [text_chunk("扫描完成。")],
            ],
            {"perform_physical_scan": {"status": "success"}},
        )
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["completed_actions"], ["scan_spectrum"])
        self.assertTrue(done["run_id"])
        self.assertEqual(done["diagnosis"]["stage"], "completed")

    async def test_deepseek_thinking_stream_preserves_reasoning_across_tool_and_retry(self) -> None:
        config = Settings(llm_provider="deepseek", deepseek_api_key="test-key")
        requests = []
        turns = [
            [{"reasoning_content": "准备"}, {"reasoning_content": "扫描"}],
            [{"reasoning_content": "执行扫描"}, {"tool_calls": [{
                "index": 0, "id": "scan-1", "type": "function",
                "function": {"name": "perform_physical_scan", "arguments": "{}"},
            }]}],
            [{"reasoning_content": "扫描完成"}, {"content": "扫描完成。"}],
        ]

        def respond(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            deltas = turns[len(requests)]
            requests.append(payload)
            chunks = [{
                "id": "response-1", "object": "chat.completion.chunk", "created": 1,
                "model": config.deepseek_model_think,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            } for delta in deltas]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

        memory = FakeMemory()
        async with create_async_llm_client(
            config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            agent = AgentLoop(
                client, FakeTools({"perform_physical_scan": {"status": "success"}}),
                self.skills, memory, config=config,
            )
            with patch("src.agent.loop.report_to_console"):
                events = [event async for event in agent.run_stream("帮我扫描 2.4 GHz 底噪", "cloud")]

        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertEqual(request["thinking"], {"type": "enabled"})
            self.assertNotIn("temperature", request)
            self.assertEqual(request["model"], config.deepseek_model_think)
            if "tools" in request:
                self.assertEqual(request["tool_choice"], "auto")
        retry_assistants = [msg for msg in requests[1]["messages"] if msg["role"] == "assistant"]
        self.assertEqual(retry_assistants[0]["reasoning_content"], "准备扫描")
        tool_assistant = next(msg for msg in requests[2]["messages"] if msg.get("tool_calls"))
        self.assertEqual(tool_assistant["reasoning_content"], "执行扫描")
        self.assertTrue(any(msg["role"] == "tool" and msg["tool_call_id"] == "scan-1" for msg in requests[2]["messages"]))
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["reply"], "扫描完成。")
        self.assertEqual(memory.value[-1]["reasoning_content"], "扫描完成")
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "delta"), "扫描完成。")

    async def test_deepseek_fast_disables_thinking_and_keeps_temperature(self) -> None:
        config = Settings(llm_provider="deepseek", deepseek_api_key="test-key")
        llm = FakeLLM([[text_chunk("你好。")]])
        agent = AgentLoop(llm, FakeTools({}), self.skills, FakeMemory(), config=config)
        with patch("src.agent.loop.report_to_console"):
            events = [event async for event in agent.run_stream("你好", "cloud-fast", mode="fast", temperature=0.4)]
        request = llm.chat.completions.requests[0]
        self.assertEqual(request["model"], config.deepseek_model_fast)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request["temperature"], 0.4)
        self.assertTrue(any(event["type"] == "done" for event in events))

    async def test_local_requests_do_not_receive_deepseek_options(self) -> None:
        config = Settings(llm_provider="local", model_think="local-think")
        llm = FakeLLM([[tool_chunk("perform_physical_scan", {})], [text_chunk("完成。")]])
        agent = AgentLoop(llm, FakeTools({"perform_physical_scan": {"status": "success"}}), self.skills, FakeMemory(), config=config)
        with patch("src.agent.loop.report_to_console"):
            events = [event async for event in agent.run_stream("帮我扫描 2.4 GHz 底噪", "local")]
        request = llm.chat.completions.requests[0]
        self.assertEqual(request["model"], "local-think")
        self.assertEqual(request["tool_choice"], "required")
        self.assertIn("temperature", request)
        self.assertNotIn("extra_body", request)
        self.assertTrue(any(event.get("status") == "success" for event in events))

    async def test_knowledge_fallback_opens_web_only_after_local_empty(self) -> None:
        done = await self.run_agent(
            "什么是罕见 SDR 术语",
            [
                [tool_chunk("search_sdr_knowledge", {"query": "罕见 SDR 术语"})],
                [tool_chunk("web_search", {"query": "罕见 SDR 术语"})],
                [text_chunk("根据网络资料回答。")],
            ],
            {
                "search_sdr_knowledge": {"status": "success", "results": []},
                "web_search": {"status": "success", "results": [{"title": "资料"}]},
            },
        )
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["completed_actions"], ["answer_with_knowledge"])

    async def test_duplicate_call_cannot_fake_business_completion(self) -> None:
        done = await self.run_agent(
            "找干净频点并发送“你好”",
            [
                [tool_chunk("auto_optimal_transmit", {})],
                [tool_chunk("auto_optimal_transmit", {})],
                [text_chunk("未能发送文本。")],
            ],
            {
                "auto_optimal_transmit": {
                    "status": "success",
                    "tx_result": {"status": "success"},
                }
            },
        )
        self.assertEqual(done["status"], "incomplete")
        self.assertIn("transmit_text", done["required_actions"])
        self.assertNotIn("transmit_text", done["completed_actions"])
        self.assertEqual(done["diagnosis"]["stage"], "loop_control")

    async def test_composite_tool_checks_nested_transmit_result(self) -> None:
        done = await self.run_agent(
            "找干净频点并发送你好",
            [
                [tool_chunk("auto_optimal_transmit", {"text": "你好"})],
                [text_chunk("已找到频点，但发送失败。")],
            ],
            {
                "auto_optimal_transmit": {
                    "status": "success",
                    "tx_result": {"status": "error", "error": "发送失败"},
                }
            },
        )
        self.assertEqual(done["status"], "incomplete")
        self.assertIn("select_clean_frequency", done["completed_actions"])
        self.assertNotIn("transmit_text", done["completed_actions"])
        self.assertEqual(done["diagnosis"]["stage"], "completion_check")

    async def test_run_trace_is_persisted_with_route_plan_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunTraceStore(Path(directory) / "runs.jsonl")
            done = await self.run_agent(
                "帮我扫描 2.4 GHz 底噪",
                [
                    [tool_chunk("perform_physical_scan", {})],
                    [text_chunk("扫描完成。")],
                ],
                {"perform_physical_scan": {"status": "success"}},
                trace_store=store,
            )
            record = store.get(done["run_id"])
            self.assertIsNotNone(record)
            self.assertEqual(record["route"]["name"], "hardware")
            self.assertEqual(record["route"]["business_intent"], "perform_physical_scan")
            self.assertEqual(record["plan"]["skill"], "perform_physical_scan")
            self.assertEqual(
                record["plan"]["skill_candidates"][0]["name"],
                "perform_physical_scan",
            )
            self.assertEqual(record["final"]["status"], "success")
            self.assertEqual(record["schema_version"], 2)
            self.assertEqual(record["request"]["max_turns"], 4)
            self.assertEqual(record["metrics"]["counts"]["llm_calls"], 2)
            self.assertEqual(record["metrics"]["counts"]["tool_calls_executed"], 1)
            self.assertIn("route_ms", record["metrics"]["timing"])
            self.assertIn("first_output_ms", record["metrics"]["llm_calls"][0])
            completion = record["final"]["completion"]
            self.assertTrue(completion["actions_satisfied"])
            self.assertTrue(completion["task_satisfied"])
            self.assertEqual(
                completion["action_evidence"]["scan_spectrum"][0]["tool"],
                "perform_physical_scan",
            )
            self.assertTrue(all(event.get("meaning") for event in record["events"]))


if __name__ == "__main__":
    unittest.main()
