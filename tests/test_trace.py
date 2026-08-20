from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent.trace import RunTrace, RunTraceStore, compact_value


class TraceTests(unittest.TestCase):
    @staticmethod
    def make_trace(run_id: str) -> RunTrace:
        trace = RunTrace(
            run_id=run_id,
            session_id="session-1",
            instruction="扫描频谱",
            mode="think",
            model="test-model",
            temperature=0.0,
        )
        trace.finish(status="success")
        return trace

    def test_compact_value_bounds_large_payloads(self) -> None:
        compacted = compact_value(
            {"samples": list(range(100)), "text": "x" * 3000},
            max_items=5,
            max_text=20,
        )
        self.assertEqual(len(compacted["samples"]), 6)
        self.assertIn("truncated", compacted["samples"][-1])
        self.assertIn("truncated", compacted["text"])

    def test_jsonl_store_lists_and_gets_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_runs.jsonl"
            store = RunTraceStore(path)
            trace = RunTrace(
                run_id="run-1",
                session_id="session-1",
                instruction="扫描频谱",
                mode="think",
                model="test-model",
                temperature=0.0,
            )
            trace.route = {"name": "hardware"}
            trace.record("routing", "route_selected", route="hardware")
            trace.finish(status="success")
            store.append(trace)

            self.assertEqual(store.list_recent(1)[0]["run_id"], "run-1")
            self.assertEqual(store.get("run-1")["final"]["status"], "success")

    def test_store_keeps_only_newest_configured_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_runs.jsonl"
            store = RunTraceStore(path, max_records=2)

            store.append(self.make_trace("run-1"))
            store.append(self.make_trace("run-2"))
            store.append(self.make_trace("run-3"))

            self.assertIsNone(store.get("run-1"))
            self.assertEqual(
                [record["run_id"] for record in store.list_recent(10)],
                ["run-3"],
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
