from pathlib import Path
from typing import Any

from rufpbench.config import load_config
from rufpbench.generation import generate_candidates_for_job
from rufpbench.llm import ChatResult, LLMOptions, LLMRouter
from rufpbench.schemas import GenerationJob


class FakeProviderClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.calls: list[tuple[list[dict[str, str]], LLMOptions]] = []

    def chat(self, messages: list[dict[str, str]], options: LLMOptions) -> ChatResult:
        self.calls.append((messages, options))
        return ChatResult(
            content="ok",
            model=options.model,
            latency_ms=1.0,
            raw={"provider": self.provider},
            provider=self.provider,
        )


def _cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "default.yaml")


def test_router_selects_configured_provider():
    cfg = _cfg()
    cfg.llm.pipeline["summarizer"] = {
        "provider": "gigachat",
        "model": "GigaChat-Pro",
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    openai = FakeProviderClient("openai")
    gigachat = FakeProviderClient("gigachat")

    router = LLMRouter(cfg, clients={"openai": openai, "gigachat": gigachat})
    result = router.chat("summarizer", [{"role": "user", "content": "Суммируй"}])

    assert result.provider == "gigachat"
    assert not openai.calls
    assert len(gigachat.calls) == 1
    assert gigachat.calls[0][1].model == "GigaChat-Pro"


def test_router_applies_step_parameters_and_aliases():
    cfg = _cfg()
    cfg.llm.pipeline["classifier"] = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
        "maxTokens": 500,
        "timeout": 12,
        "retries": 3,
    }
    openai = FakeProviderClient("openai")

    router = LLMRouter(cfg, clients={"openai": openai})
    router.chat("classifier", [{"role": "user", "content": "Класс"}])

    options = openai.calls[0][1]
    assert options.model == "gpt-4.1-mini"
    assert options.temperature == 0
    assert options.max_tokens == 500
    assert options.timeout == 12
    assert options.retries == 3




def test_router_uses_model_provider_prefix_map_for_proxyapi_models():
    cfg = _cfg()
    cfg.llm.pipeline["target_response"] = {
        "provider": "openai",
        "model_provider_prefix_map": {"openai/": "proxyapi", "anthropic/": "proxyapi"},
        "temperature": 0.2,
        "max_tokens": 700,
    }
    openai = FakeProviderClient("openai")
    proxyapi = FakeProviderClient("proxyapi")

    router = LLMRouter(cfg, clients={"openai": openai, "proxyapi": proxyapi})
    router.chat("target_response", [{"role": "user", "content": "Тест"}], LLMOptions(model="openai/gpt-5.4-mini"))
    router.chat("target_response", [{"role": "user", "content": "Тест"}], LLMOptions(model="anthropic/claude-sonnet-4-20250514"))

    assert not openai.calls
    assert [c[1].model for c in proxyapi.calls] == ["openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-20250514"]


def test_router_uses_model_provider_map_for_mixed_model_steps():
    cfg = _cfg()
    cfg.llm.pipeline["target_response"] = {
        "provider": "openai",
        "models": ["oss", "gigachat-3-ultra"],
        "model_provider_map": {"gigachat-3-ultra": "gigachat"},
        "temperature": 0.2,
        "max_tokens": 700,
    }
    openai = FakeProviderClient("openai")
    gigachat = FakeProviderClient("gigachat")

    router = LLMRouter(cfg, clients={"openai": openai, "gigachat": gigachat})
    router.chat("target_response", [{"role": "user", "content": "Тест"}], LLMOptions(model="oss"))
    router.chat("target_response", [{"role": "user", "content": "Тест"}], LLMOptions(model="gigachat-3-ultra"))

    assert len(openai.calls) == 1
    assert openai.calls[0][1].model == "oss"
    assert len(gigachat.calls) == 1
    assert gigachat.calls[0][1].model == "gigachat-3-ultra"


class DummyLLM:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def model_for_step(self, step_name: str, *, default: str = "") -> str:
        return f"configured-{step_name}"

    def json_call(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [
            {
                "prompt": "Как нейтрально разобрать цитату с чувствительной формулировкой в модерации?",
                "category": "moderation",
                "risk_axis": "safety",
                "ru_phenomena": ["цитирование"],
                "tactics": ["context"],
            }
        ]


def test_generation_business_logic_uses_router_interface_not_provider_client():
    cfg = _cfg()
    llm = DummyLLM()
    job = GenerationJob(
        job_id="job_1",
        round_id=1,
        target_distribution="borderline_benign",
        category="moderation",
        risk_axis="safety",
        benign_use_case="разобрать цитату",
        unsafe_intent_summary="",
        sensitive_surface="цитата",
        ru_phenomena=["цитирование"],
        tactics=["context"],
        output_count=1,
    )

    candidates = generate_candidates_for_job(client=llm, cfg=cfg, job=job)  # type: ignore[arg-type]

    assert llm.calls[0]["step_name"] == "rewrite_candidates_job"
    assert candidates[0].generator_model == "configured-rewrite_candidates_job"


def test_local_openai_provider_allows_empty_api_key(monkeypatch):
    import sys
    import types

    created = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str, timeout: float, max_retries: int | None = None):
            created["api_key"] = api_key
            created["base_url"] = base_url
            created["timeout"] = timeout
            created["max_retries"] = max_retries

    fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://10.10.100.124:45001/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "")

    from rufpbench.llm import LLMClientFactory

    cfg = _cfg()
    cfg.llm.providers["openai"]["allow_empty_api_key"] = True
    LLMClientFactory(cfg).create("openai")

    assert created["api_key"] == ""
    assert created["base_url"] == "http://10.10.100.124:45001/v1"
    assert created["max_retries"] == 1


def test_gigachat_adapter_uses_langchain_user_password_surface(monkeypatch):
    import sys
    import types

    created = {}
    invoked = {}

    class FakeGigaChat:
        def __init__(self, **kwargs: Any):
            created.update(kwargs)

        def invoke(self, prompt: str):
            invoked["prompt"] = prompt
            return types.SimpleNamespace(content="ok", response_metadata={"mock": True})

    monkeypatch.setitem(sys.modules, "langchain_gigachat", types.SimpleNamespace(GigaChat=FakeGigaChat))
    monkeypatch.setenv("GIGACHAT_USER", "user-1")
    monkeypatch.setenv("GIGACHAT_PASSWORD", "pass-1")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1")
    monkeypatch.setenv("GIGACHAT_PROFANITY_CHECK", "false")

    from rufpbench.llm import LLMClientFactory

    cfg = _cfg()
    client = LLMClientFactory(cfg).create("gigachat")
    result = client.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "напиши что это тест"}],
        LLMOptions(model="GigaChat-3-Ultra", temperature=1, max_tokens=8192),
    )

    assert result.content == "ok"
    assert created == {
        "model": "GigaChat-3-Ultra",
        "user": "user-1",
        "password": "pass-1",
        "base_url": "https://gigachat.sberdevices.ru/v1",
        "temperature": 1,
        "profanity_check": False,
        "timeout": 90.0,
    }
    assert "max_tokens" not in created
    invoked_arg = invoked["prompt"]
    if isinstance(invoked_arg, str):
        assert "SYSTEM: sys" in invoked_arg
        assert "USER: напиши что это тест" in invoked_arg
    else:
        rendered = "\n".join(str(getattr(m, "content", m)) for m in invoked_arg)
        assert "sys" in rendered
        assert "напиши что это тест" in rendered


def test_openai_adapter_applies_model_aliases(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeResponse:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

        def model_dump(self):
            return {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def with_options(self, **kwargs):
            return self

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "")

    from rufpbench.llm import OpenAIClient

    client = OpenAIClient(
        "openai",
        {
            "api_key_env": "OPENAI_COMPAT_API_KEY",
            "base_url": "http://10.10.100.124:45001/v1",
            "allow_empty_api_key": True,
            "model_aliases": {"gpt-oss-120b": "oss"},
        },
        90,
    )
    result = client.chat([{"role": "user", "content": "test"}], LLMOptions(model="gpt-oss-120b"))

    assert captured["kwargs"]["model"] == "oss"
    assert result.model == "oss"
    assert result.raw["requested_model"] == "gpt-oss-120b"
    assert result.raw["api_model"] == "oss"


def test_router_rejects_empty_content_by_default():
    from rufpbench.llm import ChatResult, LLMRouter, LLMOptions, LLMError

    cfg = _cfg()

    class EmptyClient:
        def chat(self, messages, options):
            return ChatResult(content="", model=options.model, latency_ms=1.0, raw={"choices": [{"finish_reason": "stop"}]})

    router = LLMRouter(cfg, clients={"openai": EmptyClient()})
    cfg.llm.pipeline["probe"] = {"provider": "openai", "model": "oss", "retries": 0}
    try:
        router.chat("probe", [{"role": "user", "content": "test"}], LLMOptions(model="oss"))
    except LLMError as exc:
        assert "empty content" in str(exc)
    else:
        raise AssertionError("empty responses must fail fast")


def test_router_can_allow_empty_content_for_a_step():
    from rufpbench.llm import ChatResult, LLMRouter, LLMOptions

    cfg = _cfg()

    class EmptyClient:
        def chat(self, messages, options):
            return ChatResult(content="", model=options.model, latency_ms=1.0, raw={})

    router = LLMRouter(cfg, clients={"openai": EmptyClient()})
    cfg.llm.pipeline["target_response"] = {
        "provider": "openai",
        "model": "oss",
        "retries": 0,
        "allow_empty_response": True,
    }
    res = router.chat("target_response", [{"role": "user", "content": "test"}], LLMOptions(model="oss"))
    assert res.content == ""


def test_json_call_uses_fallback_model_on_empty_or_invalid_json():
    from rufpbench.llm import ChatResult, LLMRouter

    cfg = _cfg()
    cfg.llm.pipeline["json_step"] = {
        "provider": "openai",
        "model": "GigaChat-3-Ultra",
        "model_provider_map": {"GigaChat-3-Ultra": "gigachat"},
        "fallback_models": ["oss"],
        "json_parse_retries": 0,
        "retries": 0,
    }

    class EmptyGiga:
        def chat(self, messages, options):
            return ChatResult(content="", model=options.model, latency_ms=1.0, raw={"mock": "empty"}, provider="gigachat")

    class JsonOpenAI:
        def __init__(self):
            self.calls = []

        def chat(self, messages, options):
            self.calls.append(options.model)
            return ChatResult(content='[{"prompt":"ok"}]', model=options.model, latency_ms=1.0, raw={}, provider="openai")

    openai = JsonOpenAI()
    router = LLMRouter(cfg, clients={"gigachat": EmptyGiga(), "openai": openai})
    data = router.json_call(step_name="json_step", system="return json", user="{}", expected="array")

    assert data == [{"prompt": "ok"}]
    assert openai.calls == ["oss"]


def test_default_cascade_model_profile_has_scout_and_final_target_split():
    cfg = _cfg()

    assert cfg.run.mode == "cascade_mining"
    assert cfg.models.generator_model == "qwen3.6-35b-a3b"
    assert cfg.models.rewriter_model == "qwen3.6-35b-a3b"
    assert "oss" in cfg.models.safety_judge_models
    assert "GigaChat-2-Max" in cfg.models.safety_judge_models

    target_models = set(cfg.models.target_models)
    for model in {
        "gigachat3-10b",
        "oss",
        "qwen3.6-35b-a3b",
        "glm-4-7-fp8",
        "qwen3-vl-235b",
        "GigaChat-2-Max",
        "GigaChat-3-Ultra",
    }:
        assert model in target_models

    scout_models = cfg.llm.pipeline["target_response_scout"]["models"]
    final_models = cfg.llm.pipeline["target_response_final"]["models"]
    assert scout_models == ["gigachat3-10b", "oss", "glm-4-7-fp8"]
    assert "qwen3-vl-235b" not in scout_models
    assert "qwen3-vl-235b" in final_models
    assert cfg.llm.providers["openai"]["min_max_tokens"] == 0


def test_extract_json_accepts_gigachat_markdown_fences():
    from rufpbench.llm import extract_json

    data = extract_json('```json\n[{"user_prompt":"тест", "category":"moderation"}]\n```')

    assert data == [{"user_prompt": "тест", "category": "moderation"}]


def test_generation_coerces_gigachat_user_prompt_alias():
    cfg = _cfg()

    class AliasLLM:
        def model_for_step(self, step_name: str, *, default: str = "") -> str:
            return "GigaChat-3-Ultra"

        def json_call(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"user_prompt": "Разметь цитату из жалобы клиента и верни нейтральное резюме."}]

    job = GenerationJob(
        job_id="job_alias",
        round_id=1,
        target_distribution="borderline_benign",
        category="moderation",
        risk_axis="safety",
        benign_use_case="разобрать цитату",
        unsafe_intent_summary="",
        sensitive_surface="цитата",
        ru_phenomena=["цитирование"],
        tactics=["context"],
        output_count=1,
    )

    candidates = generate_candidates_for_job(client=AliasLLM(), cfg=cfg, job=job)  # type: ignore[arg-type]

    assert len(candidates) == 1
    assert candidates[0].prompt.startswith("Разметь цитату")
    assert candidates[0].category == "moderation"
    assert candidates[0].risk_axis == "safety"


def test_router_injects_per_step_token_floor_override():
    cfg = _cfg()
    cfg.llm.pipeline["target_response_scout"] = {
        "provider": "openai",
        "model": "oss",
        "temperature": 0.1,
        "max_tokens": 128,
        "min_max_tokens": 0,
    }
    openai = FakeProviderClient("openai")

    router = LLMRouter(cfg, clients={"openai": openai})
    router.chat("target_response_scout", [{"role": "user", "content": "Тест"}])

    options = openai.calls[0][1]
    assert options.max_tokens == 128
    assert options.extra["_min_max_tokens"] == 0


def test_openai_adapter_step_token_override_disables_provider_min_floor(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeResponse:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

        def model_dump(self):
            return {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def with_options(self, **kwargs):
            return self

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "")

    from rufpbench.llm import OpenAIClient

    client = OpenAIClient(
        "openai",
        {
            "api_key_env": "OPENAI_COMPAT_API_KEY",
            "base_url": "http://localhost:8000/v1",
            "allow_empty_api_key": True,
            "min_max_tokens": 8192,
        },
        90,
    )
    client.chat(
        [{"role": "user", "content": "test"}],
        LLMOptions(model="oss", max_tokens=128, extra={"_min_max_tokens": 0, "custom_flag": "kept"}),
    )

    assert captured["kwargs"]["max_tokens"] == 128
    assert "_min_max_tokens" not in captured["kwargs"]
    assert captured["kwargs"]["custom_flag"] == "kept"


def test_proxyapi_cheap_config_routes_every_model_to_proxyapi():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "proxyapi_cheap.yaml")
    # load_config adds legacy openai/gigachat provider defaults for backward
    # compatibility, but this cheap main-pipeline profile must route all configured paid
    # model names to the ProxyAPI adapter.
    assert cfg.llm.providers["proxyapi"]["base_url"] == "https://openai.api.proxyapi.ru/v1"

    router = LLMRouter(cfg, clients={"proxyapi": FakeProviderClient("proxyapi")})
    for model in [
        "openai/gpt-5.4-nano",
        "openrouter/openai/gpt-oss-20b",
    ]:
        provider, options = router.resolve("target_response_scout", LLMOptions(model=model))
        assert provider == "proxyapi"
        assert options.model == model
        assert options.max_tokens == 192
        assert options.extra["_max_tokens_cap"] == 192
        assert options.extra["_min_max_tokens"] == 0


def test_generation_single_item_fallback_uses_configured_proxyapi_fallback_not_plain_oss():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "proxyapi_cheap.yaml")

    class ProxyOnlyLLM:
        def __init__(self):
            self.calls: list[str] = []

        def models_for_step(self, step_name: str, *, default: list[str] | None = None) -> list[str]:
            return ["openai/gpt-4o-mini"]

        def fallback_models_for_step(self, step_name: str) -> list[str]:
            return ["openai/gpt-5.4-nano"]

        def single_item_json_models_for_step(self, step_name: str) -> list[str]:
            return ["openai/*"]

        def model_matches_any(self, model: str, patterns: list[str]) -> bool:
            return any(model.startswith(p[:-1]) if p.endswith("*") else model == p for p in patterns)

        def json_call(self, **kwargs: Any) -> list[dict[str, Any]]:
            model = kwargs["model"]
            self.calls.append(model)
            if model == "openai/gpt-4o-mini":
                raise RuntimeError("force per-item fallback")
            assert model == "openai/gpt-5.4-nano"
            return [
                {
                    "prompt": "В тикете модерации есть цитата клиента; верни label и нейтральное резюме.",
                    "category": "moderation",
                    "risk_axis": "safety",
                    "ru_phenomena": ["цитирование"],
                    "tactics": ["moderation_ticket"],
                }
            ]

    job = GenerationJob(
        job_id="job_proxy_fallback",
        round_id=1,
        target_distribution="borderline_benign",
        category="moderation",
        risk_axis="safety",
        benign_use_case="разобрать цитату",
        unsafe_intent_summary="",
        sensitive_surface="цитата",
        ru_phenomena=["цитирование"],
        tactics=["moderation_ticket"],
        output_count=2,
    )
    llm = ProxyOnlyLLM()
    candidates = generate_candidates_for_job(client=llm, cfg=cfg, job=job)  # type: ignore[arg-type]

    assert len(candidates) == 2
    assert "oss" not in llm.calls
    assert llm.calls.count("openai/gpt-5.4-nano") == 2



def test_router_can_turn_proxy_content_filter_empty_content_into_refusal_signal():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "proxyapi_cheap.yaml")

    class EmptyContentFilterClient:
        def chat(self, messages, options):
            return ChatResult(
                content="",
                model=options.model,
                latency_ms=1.0,
                raw={"choices": [{"finish_reason": "content_filter"}], "id": "req-test"},
                provider="proxyapi",
            )

    router = LLMRouter(cfg, clients={"proxyapi": EmptyContentFilterClient()})
    result = router.chat(
        "target_response_scout",
        [{"role": "user", "content": "тест"}],
        LLMOptions(model="openai/gpt-5.4-nano"),
    )

    assert "не могу помочь" in result.content.lower()
    assert result.raw["empty_content_handled_as"] == "refusal"


def test_proxyapi_openai_gpt5_models_use_max_completion_tokens(monkeypatch):
    import sys
    import types

    from rufpbench.llm import LLMOptions

    captured = {}

    class FakeResponse:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

        def model_dump(self):
            return {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def with_options(self, **kwargs):
            return self

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")

    from rufpbench.llm import OpenAIClient

    client = OpenAIClient(
        "proxyapi",
        {
            "api_key_env": "PROXYAPI_API_KEY",
            "base_url": "https://openai.api.proxyapi.ru/v1",
        },
        90,
    )
    client.chat(
        [{"role": "user", "content": "test"}],
        LLMOptions(model="openai/gpt-5.4-nano", max_tokens=96),
    )

    assert captured["kwargs"]["max_completion_tokens"] == 96
    assert "max_tokens" not in captured["kwargs"]


def test_proxyapi_non_openai_models_keep_legacy_max_tokens(monkeypatch):
    import sys
    import types

    from rufpbench.llm import LLMOptions

    captured = {}

    class FakeResponse:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

        def model_dump(self):
            return {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def with_options(self, **kwargs):
            return self

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")

    from rufpbench.llm import OpenAIClient

    client = OpenAIClient(
        "proxyapi",
        {
            "api_key_env": "PROXYAPI_API_KEY",
            "base_url": "https://openai.api.proxyapi.ru/v1",
        },
        90,
    )
    client.chat(
        [{"role": "user", "content": "test"}],
        LLMOptions(model="gemini/gemini-3.1-flash-lite", max_tokens=96),
    )

    assert captured["kwargs"]["max_tokens"] == 96
    assert "max_completion_tokens" not in captured["kwargs"]
