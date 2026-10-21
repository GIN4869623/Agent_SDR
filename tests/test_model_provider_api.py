from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx

from src.config import Settings


class ModelProviderAPITests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Import the real HTTP app without connecting to hardware or indexing data.
        with patch("hardware.sdr_controller.SDRController._connect"), patch(
            "core.rag.LocalKnowledgeBase", return_value=NS(document_count=0)
        ):
            cls.server = importlib.import_module("src.server")

    async def asyncSetUp(self) -> None:
        self.config = Settings(
            llm_provider="local", model_think="local-think", model_fast="local-fast",
            deepseek_api_key="cloud-secret", deepseek_model_think="cloud-think",
            deepseek_model_fast="cloud-fast",
        )
        self.calls = []

        def client_factory(config):
            def create(**kwargs):
                self.calls.append((config.llm_provider, kwargs))
                return iter([NS(choices=[NS(delta=NS(content="你好。", tool_calls=None))])])
            return NS(chat=NS(completions=NS(create=create)))

        for patcher in [
            patch.object(self.server, "settings", self.config),
            patch.object(self.server, "agent_loops", {}),
            patch.object(self.server, "create_async_llm_client", side_effect=client_factory),
            patch.object(self.server, "_post_console_event", new_callable=AsyncMock),
            patch("src.agent.loop.report_to_console"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.app), base_url="http://test")
        self.addAsyncCleanup(self.http.aclose)

    async def test_switching_requests_uses_selected_provider_and_keeps_default(self) -> None:
        for provider, expected in [("local", "local-think"), ("deepseek", "cloud-think"), ("local", "local-think")]:
            response = await self.http.post("/api/chat", json={
                "instruction": "你好", "session_id": "provider-switch", "provider": provider,
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["active_model"], expected)
        self.assertEqual([call[0] for call in self.calls], ["local", "deepseek", "local"])
        self.assertEqual(self.config.llm_provider, "local")

    async def test_streaming_endpoint_routes_to_cloud_fast(self) -> None:
        response = await self.http.post("/api/chat/stream", json={
            "instruction": "你好", "session_id": "cloud-stream", "provider": "deepseek", "mode": "fast",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('"active_model": "cloud-fast"', response.text)
        self.assertEqual(self.calls[0][0], "deepseek")
        self.assertEqual(self.calls[0][1]["extra_body"]["thinking"]["type"], "disabled")

    async def test_missing_key_returns_actionable_error_without_calling_model(self) -> None:
        self.config.deepseek_api_key = Settings(deepseek_api_key="").deepseek_api_key
        # Validate the configuration before the mocked client factory is reached.
        original = self.server.create_async_llm_client.side_effect
        def validate_then_create(config):
            config.validate_llm_config()
            return original(config)
        self.server.create_async_llm_client.side_effect = validate_then_create
        for path in ["/api/chat", "/api/chat/stream"]:
            response = await self.http.post(path, json={
                "instruction": "你好", "session_id": "missing-key", "provider": "deepseek",
            })
            self.assertEqual(response.status_code, 400)
            self.assertIn("DEEPSEEK_API_KEY", response.json()["detail"])
        self.assertEqual(self.calls, [])
        response = await self.http.post("/api/chat", json={"instruction": "你好", "session_id": "local-fallback"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_model"], "local-think")

    async def test_health_only_exposes_readiness_and_model_names(self) -> None:
        response = await self.http.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["providers"]["deepseek"]["configured"])
        self.assertNotIn("cloud-secret", response.text)

    async def test_unknown_provider_is_rejected(self) -> None:
        response = await self.http.post("/api/chat", json={
            "instruction": "你好", "session_id": "invalid", "provider": "invalid",
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
