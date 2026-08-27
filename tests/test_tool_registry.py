from __future__ import annotations

import json
import unittest

from src.mcp import create_sdr_mcp
from tools.registry import create_tool_registry


class FakeController:
    def __init__(self) -> None:
        self.scan_calls: list[dict] = []

    def has_background_task(self) -> bool:
        return False

    def perform_physical_scan(self, **kwargs):
        self.scan_calls.append(kwargs)
        return {"status": "success", "samples": 1}

    def tone_loopback_visualize(self, **kwargs):
        return {"status": "success", **kwargs}

    def stop_hardware_task(self):
        return {"status": "success"}

    def text_fsk_send_and_receive(self, **kwargs):
        return {"status": "success", **kwargs}

    def adaptive_modulation_transmit(self, **kwargs):
        return {"status": "success", **kwargs}

    def auto_optimal_transmit(self, **kwargs):
        return {"status": "success", **kwargs}

    def get_hardware_snapshot(self):
        return {"connected": False}

    def get_visualization_snapshot(self):
        return {"active": False}


class FakeKnowledgeBase:
    def search(self, _query: str, top_k: int = 3):
        return []

    def query(self, _query: str):
        return ""


class FakeVideoController:
    def start(self):
        return {"status": "success"}

    def stop(self):
        return {"status": "success"}

    def get_state(self):
        return {"running": False}


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.controller = FakeController()
        self.knowledge = FakeKnowledgeBase()
        self.registry = create_tool_registry(
            self.controller,
            self.knowledge,
            FakeVideoController(),
        )

    def test_registry_is_the_schema_source_of_truth(self) -> None:
        names = {tool.name for tool in self.registry.all()}
        self.assertEqual(
            names,
            {
                "perform_physical_scan",
                "tone_loopback_visualize",
                "stop_hardware_task",
                "text_fsk_send_and_receive",
                "adaptive_modulation_transmit",
                "auto_optimal_transmit",
                "search_sdr_knowledge",
                "web_search",
                "start_video_stream",
                "stop_video_stream",
                "video_stream_status",
            },
        )
        send_tool = self.registry.get("text_fsk_send_and_receive")
        assert send_tool is not None
        self.assertIn("text", send_tool.parameters["required"])
        self.assertFalse(send_tool.parameters["additionalProperties"])
        self.assertEqual(send_tool.risk, "transmit")
        self.assertFalse(send_tool.retry_safe)

    async def test_valid_scan_uses_validated_defaults(self) -> None:
        result = json.loads(await self.registry.call("perform_physical_scan", {}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(self.controller.scan_calls), 1)
        self.assertEqual(self.controller.scan_calls[0]["center_freq_hz"], 2.4e9)
        self.assertEqual(self.controller.scan_calls[0]["bandwidth_hz"], 1e6)

    async def test_out_of_range_frequency_is_rejected_before_hardware(self) -> None:
        result = await self.registry.execute(
            "perform_physical_scan", {"center_freq_hz": 100e6}
        )
        self.assertEqual(result["errorType"], "INVALID_TOOL_ARGUMENTS")
        self.assertEqual(self.controller.scan_calls, [])

    async def test_unknown_arguments_are_rejected(self) -> None:
        result = await self.registry.execute(
            "perform_physical_scan", {"shell_command": "uhd_usrp_probe"}
        )
        self.assertEqual(result["errorType"], "INVALID_TOOL_ARGUMENTS")
        self.assertEqual(self.controller.scan_calls, [])

    async def test_skill_allowlist_is_enforced_at_execution_boundary(self) -> None:
        result = await self.registry.execute(
            "tone_loopback_visualize",
            {},
            allowed_tools={"perform_physical_scan"},
        )
        self.assertEqual(result["errorType"], "TOOL_NOT_ALLOWED")

    def test_mcp_adapter_builds_catalogue_from_existing_registry(self) -> None:
        mcp = create_sdr_mcp(
            tool_registry=self.registry,
            controller=self.controller,
            knowledge_base=self.knowledge,
        )
        mcp_tools = {tool.name: tool for tool in mcp._tool_manager.list_tools()}
        self.assertEqual(set(mcp_tools), {tool.name for tool in self.registry.all()})
        mcp_frequency = mcp_tools["perform_physical_scan"].parameters["properties"][
            "center_freq_hz"
        ]
        registry_frequency = self.registry.get(
            "perform_physical_scan"
        ).parameters["properties"]["center_freq_hz"]
        self.assertEqual(mcp_frequency["minimum"], registry_frequency["minimum"])
        self.assertEqual(mcp_frequency["maximum"], registry_frequency["maximum"])


if __name__ == "__main__":
    unittest.main()
