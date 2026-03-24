from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from shared.llm import LLMClientRegistry, ResolvedLLMConfig, _is_free_model_name


class ProxyAPIFreeConfigTests(unittest.TestCase):
    def test_default_resolved_config_uses_proxyapi_free_envs(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROXYAPI_API_KEY": "test-key",
                "PROXYAPI_OPENROUTER_BASE_URL": "https://api.proxyapi.ru/openrouter/v1",
                "FREE_MODEL": "openrouter/free",
            },
            clear=False,
        ):
            cfg = LLMClientRegistry({}).resolve(agent_section="rewrite")
        self.assertIsInstance(cfg, ResolvedLLMConfig)
        self.assertEqual(cfg.api_key_env, "PROXYAPI_API_KEY")
        self.assertEqual(cfg.base_url, "https://api.proxyapi.ru/openrouter/v1")
        self.assertEqual(cfg.model, "openrouter/free")

    def test_specific_free_model_is_supported(self) -> None:
        model = "meta-llama/llama-3.3-70b-instruct:free"
        with patch.dict(
            os.environ,
            {
                "PROXYAPI_API_KEY": "test-key",
                "PROXYAPI_OPENROUTER_BASE_URL": "https://api.proxyapi.ru/openrouter/v1",
                "FREE_MODEL": model,
            },
            clear=False,
        ):
            cfg = LLMClientRegistry({}).resolve(agent_section="rewrite")
        self.assertEqual(cfg.model, model)
        self.assertTrue(_is_free_model_name(model))

    def test_model_mode_free_ignores_legacy_specific_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROXYAPI_API_KEY": "test-key",
                "PROXYAPI_MODEL_MODE": "free",
                "FREE_MODEL": "openrouter/free",
                "REWRITE_AGENT_PROXYAPI_MODEL": "qwen/qwen3-8b",
            },
            clear=False,
        ):
            cfg = LLMClientRegistry({}).resolve(agent_section="rewrite")
        self.assertEqual(cfg.model, "openrouter/free")

    def test_model_mode_specific_uses_fixed_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROXYAPI_API_KEY": "test-key",
                "PROXYAPI_MODEL_MODE": "specific",
                "PROXYAPI_MODEL": "qwen/qwen3-8b",
                "FREE_MODEL": "openrouter/free",
            },
            clear=False,
        ):
            cfg = LLMClientRegistry({}).resolve(agent_section="rewrite")
        self.assertEqual(cfg.model, "qwen/qwen3-8b")


if __name__ == "__main__":
    unittest.main()
