from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.config import Settings
from src.llm import llm_client_options, provider_catalog, provider_config


class ModelConfigTests(unittest.TestCase):
    def test_local_path_keeps_existing_endpoint_key_and_models(self) -> None:
        config = Settings(
            llm_provider="local", ollama_base_url="http://localhost:11434/v1",
            api_key="local-key", model_think="local-think", model_fast="local-fast",
            deepseek_api_key="cloud-key",
        )
        self.assertEqual(llm_client_options(config), {
            "base_url": "http://localhost:11434/v1", "api_key": "local-key",
        })
        self.assertEqual(config.model_map, {"think": "local-think", "fast": "local-fast"})

    def test_deepseek_uses_only_its_own_credentials_and_models(self) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "cloud-secret",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL_THINK": "cloud-think", "DEEPSEEK_MODEL_FAST": "cloud-fast",
        }):
            config = Settings(api_key="local-key")
        self.assertEqual(llm_client_options(config), {
            "base_url": "https://api.deepseek.com", "api_key": "cloud-secret",
        })
        self.assertEqual(config.model_map, {"think": "cloud-think", "fast": "cloud-fast"})
        self.assertNotIn("cloud-secret", repr(config))
        self.assertNotIn("cloud-secret", config.model_dump_json())

    def test_cloud_configuration_cannot_fall_back_to_local_credentials(self) -> None:
        for field in ["deepseek_api_key", "deepseek_base_url", "deepseek_model_think", "deepseek_model_fast"]:
            with self.subTest(field=field):
                values = {
                    "llm_provider": "deepseek", "deepseek_api_key": "cloud-key",
                    "deepseek_base_url": "https://api.deepseek.com",
                    "deepseek_model_think": "cloud-think", "deepseek_model_fast": "cloud-fast",
                    field: " ",
                }
                with self.assertRaisesRegex(ValueError, field.upper()):
                    llm_client_options(Settings(**values))

    def test_provider_typos_fail_instead_of_silently_selecting_local(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "deepseak"}):
            with self.assertRaises(ValidationError):
                Settings()

    def test_request_provider_does_not_mutate_default_configuration(self) -> None:
        config = Settings(llm_provider="local", deepseek_api_key="cloud-key")
        cloud = provider_config("deepseek", config)
        self.assertEqual(config.llm_provider, "local")
        self.assertEqual(cloud.llm_provider, "deepseek")
        self.assertEqual(llm_client_options(cloud)["api_key"], "cloud-key")
        self.assertEqual(provider_config(None, config).llm_provider, "local")

    def test_catalog_reports_missing_key_without_exposing_credentials(self) -> None:
        config = Settings(llm_provider="local", api_key="local-secret", deepseek_api_key="")
        catalog = provider_catalog(config)
        self.assertTrue(catalog["local"]["configured"])
        self.assertFalse(catalog["deepseek"]["configured"])
        self.assertIn("DEEPSEEK_API_KEY", catalog["deepseek"]["message"])
        config.deepseek_api_key = Settings(deepseek_api_key="cloud-secret").deepseek_api_key
        ready = provider_catalog(config)
        self.assertTrue(ready["deepseek"]["configured"])
        self.assertNotIn("cloud-secret", str(ready))
        self.assertNotIn("local-secret", str(ready))


if __name__ == "__main__":
    unittest.main()
