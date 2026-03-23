from __future__ import annotations

"""OpenAI-compatible chat client (ProxyAPI and similar) for agents that need an LLM."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import os
import time

import requests

from shared.config import as_config_dict
from shared.logging_utils import get_logger

_log = get_logger("shared.llm")

# region agent log
_DEBUG_LLM_LOG_PATH = Path("/Users/dekovaleva/PythonProjects/ru_fp_bench/.cursor/debug-5d942c.log")
_DEBUG_LLM_SESSION = "5d942c"


def _dbg_llm(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "llm-client",
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_LLM_SESSION,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LLM_LOG_PATH.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# Defaults favor OpenRouter direct (free test route); PROXYAPI_* env / yaml still override via resolve().
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
# Legacy names kept for callers/tests that import them
DEFAULT_PROXYAPI_BASE_URL = DEFAULT_OPENROUTER_BASE_URL
DEFAULT_PROXYAPI_MODEL = DEFAULT_OPENROUTER_MODEL
# RewriteAgent → ProxyAPI OpenRouter (paid); defaults if env/yaml omit values
DEFAULT_REWRITE_PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openrouter/v1"
DEFAULT_REWRITE_PROXYAPI_MODEL = "qwen/qwen3-8b"
DEFAULT_DATA_COLLECTION_PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openrouter/v1"
DEFAULT_DATA_COLLECTION_PROXYAPI_MODEL = "qwen/qwen3-8b"


def _first_nonempty_str(*candidates: object) -> str | None:
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if s:
            return s
    return None


def _default_api_key_env_name() -> str:
    """Prefer OPENROUTER_API_KEY when set; else PROXYAPI (backward compatible)."""
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "OPENROUTER_API_KEY"
    return "PROXYAPI_API_KEY"


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect HTTP 429 / OpenRouter free-tier rate limits from SDK or requests."""
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    # openai.APIStatusError
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("code") == 429:
        return True
    name = type(exc).__name__
    if "RateLimit" in name or "TooManyRequests" in name:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg


@dataclass
class ResolvedLLMConfig:
    profile_name: str = "default"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    model: str = DEFAULT_OPENROUTER_MODEL
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_sleep: float = 2.0
    backoff_on_429_s: float = 15.0
    max_429_retries: int = 8
    temperature: float = 0.2
    max_tokens: int = 800
    enabled: bool = True
    extra_headers: dict[str, str] | None = None

    @property
    def api_key(self) -> str:
        primary = os.getenv(self.api_key_env, "").strip()
        if primary:
            return primary
        # Dedicated env names: do not cross-fallback to generic PROXYAPI / OPENROUTER keys.
        if self.api_key_env in (
            "OPENROUTER_API_KEY",
            "REWRITE_AGENT_PROXYAPI_API_KEY",
            "DATA_COLLECTION_PROXYAPI_API_KEY",
        ):
            return ""
        # Soft migration: allow the other common key if the configured env is empty
        if self.api_key_env != "PROXYAPI_API_KEY":
            fb = os.getenv("PROXYAPI_API_KEY", "").strip()
            if fb:
                return fb
        if self.api_key_env != "OPENROUTER_API_KEY":
            fb = os.getenv("OPENROUTER_API_KEY", "").strip()
            if fb:
                return fb
        return ""

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)


class ProxyAPIModelClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        backoff_on_429_s: float = 15.0,
        max_429_retries: int = 8,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for ProxyAPIModelClient")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.backoff_on_429_s = float(backoff_on_429_s)
        self.max_429_retries = int(max_429_retries)
        self.extra_headers = extra_headers or {}
        self._sdk_client = None

        if OpenAI is not None:  # pragma: no cover - optional path
            try:
                self._sdk_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    default_headers=self.extra_headers or None,
                )
            except Exception:
                self._sdk_client = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: Optional[str] = None,
        request_timeout: Optional[float] = None,
        **extra: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(extra)

        eff_timeout = float(request_timeout) if request_timeout is not None else self.timeout

        # region agent log
        _msg_chars = sum(len(m.get("content", "")) for m in messages)
        _dbg_llm(
            "H-D",
            "ProxyAPIModelClient.chat",
            "routing",
            {
                "use_sdk": self._sdk_client is not None,
                "model": payload.get("model"),
                "timeout_default": self.timeout,
                "effective_timeout": eff_timeout,
                "timeout_override": request_timeout is not None,
                "messages_chars": _msg_chars,
            },
        )
        # endregion

        if self._sdk_client is not None:  # pragma: no cover - optional path
            return self._chat_via_sdk(payload, timeout_s=eff_timeout)
        return self._chat_via_requests(payload, timeout_s=eff_timeout)

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: Optional[str] = None,
        request_timeout: Optional[float] = None,
        **extra: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            request_timeout=request_timeout,
            **extra,
        )

    def _chat_via_sdk(self, payload: dict[str, Any], *, timeout_s: float) -> str:  # pragma: no cover
        last_exc: Optional[Exception] = None
        n_429 = 0
        n_other = 0
        max_rounds = self.max_429_retries + self.max_retries + 5
        for _round in range(max_rounds):
            try:
                # region agent log
                _t0 = time.monotonic()
                _dbg_llm("H-D", "_chat_via_sdk:before_create", "sdk request", {"round": _round})
                # endregion
                try:
                    resp = self._sdk_client.chat.completions.create(**payload, timeout=timeout_s)
                except TypeError:
                    resp = self._sdk_client.chat.completions.create(**payload)
                # region agent log
                _dbg_llm(
                    "H-A",
                    "_chat_via_sdk:after_create",
                    "sdk ok",
                    {"round": _round, "elapsed_ms": round((time.monotonic() - _t0) * 1000, 2)},
                )
                # endregion
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc):
                    if n_429 >= self.max_429_retries:
                        break
                    n_429 += 1
                    _log.warning(
                        "LLM 429 rate limit; sleeping %.1fs (%s/%s)",
                        self.backoff_on_429_s,
                        n_429,
                        self.max_429_retries,
                    )
                    time.sleep(self.backoff_on_429_s)
                    continue
                n_other += 1
                if n_other >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * n_other)
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")

    def _chat_via_requests(self, payload: dict[str, Any], *, timeout_s: float) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        last_exc: Optional[Exception] = None
        n_429 = 0
        n_other = 0
        max_rounds = self.max_429_retries + self.max_retries + 5
        for _round in range(max_rounds):
            try:
                # region agent log
                _t0 = time.monotonic()
                _dbg_llm("H-A", "_chat_via_requests:before_post", "http post", {"round": _round, "url": url})
                # endregion
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
                # region agent log
                _dbg_llm(
                    "H-A",
                    "_chat_via_requests:after_post",
                    "http response",
                    {
                        "round": _round,
                        "status": resp.status_code,
                        "elapsed_ms": round((time.monotonic() - _t0) * 1000, 2),
                    },
                )
                # endregion
                if resp.status_code == 429:
                    if n_429 >= self.max_429_retries:
                        last_exc = RuntimeError(f"HTTP 429: {resp.text[:500]}")
                        break
                    n_429 += 1
                    _log.warning(
                        "LLM 429 rate limit; sleeping %.1fs (%s/%s)",
                        self.backoff_on_429_s,
                        n_429,
                        self.max_429_retries,
                    )
                    time.sleep(self.backoff_on_429_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code == 429:
                    if n_429 >= self.max_429_retries:
                        break
                    n_429 += 1
                    _log.warning(
                        "LLM 429 rate limit; sleeping %.1fs (%s/%s)",
                        self.backoff_on_429_s,
                        n_429,
                        self.max_429_retries,
                    )
                    time.sleep(self.backoff_on_429_s)
                    continue
                n_other += 1
                if n_other >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * n_other)
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc):
                    if n_429 >= self.max_429_retries:
                        break
                    n_429 += 1
                    _log.warning(
                        "LLM 429 rate limit; sleeping %.1fs (%s/%s)",
                        self.backoff_on_429_s,
                        n_429,
                        self.max_429_retries,
                    )
                    time.sleep(self.backoff_on_429_s)
                    continue
                n_other += 1
                if n_other >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * n_other)
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")


class LLMClientRegistry:
    def __init__(self, project_config: str | Path | dict[str, Any] | None = None) -> None:
        self.project_config = as_config_dict(project_config)
        self._cache: dict[tuple[str, str], Optional[ProxyAPIModelClient]] = {}

    def resolve(
        self, *, agent_section: Optional[str] = None, explicit_profile: Optional[str] = None
    ) -> ResolvedLLMConfig:
        raw = self.project_config or {}
        global_llm = dict(raw.get("llm") or {})
        profiles = dict(raw.get("llm_profiles") or {})
        section_cfg = (
            dict(raw.get(agent_section) or {})
            if agent_section and isinstance(raw.get(agent_section), dict)
            else {}
        )

        profile_name = (
            explicit_profile
            or section_cfg.get("llm_profile")
            or raw.get("llm_profile")
            or "default"
        )
        profile_cfg = dict(profiles.get(profile_name) or {})

        section_llm = dict(section_cfg.get("llm") or {})
        section_runtime = dict(section_cfg.get("runtime") or {})

        merged: dict[str, Any] = {}
        merged.update(global_llm)
        merged.update(profile_cfg)
        merged.update(section_llm)

        enabled = bool(section_cfg.get("llm_enabled", merged.get("enabled", True)))

        extra_headers = merged.get("extra_headers")
        if isinstance(extra_headers, dict):
            eh = {str(k): str(v) for k, v in extra_headers.items() if str(v).strip()}
        else:
            eh = None

        base_url = _first_nonempty_str(
            merged.get("base_url"),
            os.getenv("DATA_COLLECTION_PROXYAPI_BASE_URL") if agent_section == "collection" else None,
            os.getenv("REWRITE_AGENT_PROXYAPI_BASE_URL") if agent_section == "rewrite" else None,
            os.getenv("OPENROUTER_BASE_URL"),
            os.getenv("PROXYAPI_BASE_URL"),
        )
        if not base_url:
            if agent_section == "rewrite":
                base_url = DEFAULT_REWRITE_PROXYAPI_BASE_URL
            elif agent_section == "collection":
                base_url = DEFAULT_DATA_COLLECTION_PROXYAPI_BASE_URL
            else:
                base_url = DEFAULT_OPENROUTER_BASE_URL

        # On OpenRouter direct host (openrouter.ai), do not fall back to PROXYAPI_MODEL.
        model_candidates: list[object] = [
            merged.get("model"),
            os.getenv("DATA_COLLECTION_PROXYAPI_MODEL") if agent_section == "collection" else None,
            os.getenv("REWRITE_AGENT_PROXYAPI_MODEL") if agent_section == "rewrite" else None,
            os.getenv("OPENROUTER_MODEL"),
        ]
        if "openrouter.ai" not in (base_url or ""):
            model_candidates.append(os.getenv("PROXYAPI_MODEL"))
        model = _first_nonempty_str(*model_candidates)
        if not model:
            if agent_section == "rewrite" and "proxyapi.ru" in (base_url or ""):
                model = DEFAULT_REWRITE_PROXYAPI_MODEL
            elif agent_section == "collection" and "proxyapi.ru" in (base_url or ""):
                model = DEFAULT_DATA_COLLECTION_PROXYAPI_MODEL
            else:
                model = DEFAULT_OPENROUTER_MODEL

        api_key_env = _first_nonempty_str(
            merged.get("api_key_env"),
            os.getenv("PROXYAPI_API_KEY_ENV"),
        ) or (
            "DATA_COLLECTION_PROXYAPI_API_KEY"
            if agent_section == "collection"
            else (
                "REWRITE_AGENT_PROXYAPI_API_KEY"
                if agent_section == "rewrite"
                else _default_api_key_env_name()
            )
        )

        backoff_429 = merged.get("backoff_on_429_s")
        if backoff_429 is None:
            backoff_429 = section_runtime.get("backoff_on_429_s", 15.0)
        max429 = merged.get("max_429_retries")
        if max429 is None:
            max429 = section_runtime.get("max_429_retries", 8)

        return ResolvedLLMConfig(
            profile_name=str(profile_name),
            api_key_env=str(api_key_env),
            base_url=str(base_url),
            model=str(model),
            timeout_s=float(merged.get("timeout_s", 120.0)),
            max_retries=int(merged.get("max_retries", 3)),
            retry_sleep=float(merged.get("retry_sleep", 2.0)),
            backoff_on_429_s=float(backoff_429),
            max_429_retries=int(max429),
            temperature=float(merged.get("temperature", 0.2)),
            max_tokens=int(merged.get("max_tokens", 800)),
            enabled=enabled,
            extra_headers=eh,
        )

    def get_client(
        self, *, agent_section: Optional[str] = None, explicit_profile: Optional[str] = None
    ) -> Optional[ProxyAPIModelClient]:
        resolved = self.resolve(agent_section=agent_section, explicit_profile=explicit_profile)
        cache_key = (agent_section or "__none__", resolved.profile_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not resolved.available:
            self._cache[cache_key] = None
            return None

        client = ProxyAPIModelClient(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            model=resolved.model,
            timeout=resolved.timeout_s,
            max_retries=resolved.max_retries,
            retry_sleep=resolved.retry_sleep,
            backoff_on_429_s=resolved.backoff_on_429_s,
            max_429_retries=resolved.max_429_retries,
            extra_headers=resolved.extra_headers,
        )
        self._cache[cache_key] = client
        return client


class LLMEnabledMixin:
    def _init_llm(
        self,
        *,
        project_config: str | Path | dict[str, Any] | None,
        agent_section: Optional[str],
        default_profile: str = "default",
    ) -> None:
        self._llm_project_config = as_config_dict(project_config)
        self._llm_agent_section = agent_section
        self.llm_registry = LLMClientRegistry(self._llm_project_config)

        provisional = self.llm_registry.resolve(agent_section=agent_section, explicit_profile=None)
        if provisional.profile_name == "default" and default_profile and default_profile != "default":
            effective_profile = default_profile
        else:
            effective_profile = provisional.profile_name or default_profile or "default"

        self._llm_default_profile = effective_profile
        resolved = self.llm_registry.resolve(
            agent_section=agent_section, explicit_profile=effective_profile
        )
        self.llm_profile = resolved.profile_name
        self.llm_enabled = resolved.available
        self.llm_config = resolved

        if not resolved.available and agent_section == "rewrite":
            _log.warning(
                "LLM rewrite: API key missing for api_key_env=%s (set in .env; see .env.example).",
                resolved.api_key_env,
            )
        elif not resolved.available and agent_section == "collection":
            _log.warning(
                "LLM collection: API key missing for api_key_env=%s (code EDA still works; set key for LLM summary).",
                resolved.api_key_env,
            )
        elif not resolved.available and resolved.api_key_env == "OPENROUTER_API_KEY":
            _log.warning(
                "LLM: OPENROUTER_API_KEY is missing or empty; not falling back to PROXYAPI_API_KEY. "
                "Set OPENROUTER_API_KEY in .env for rewrite.llm (see .env.example)."
            )

        if resolved.available and agent_section == "rewrite":
            route = (
                "ProxyAPI OpenRouter"
                if "proxyapi.ru" in (resolved.base_url or "")
                else "OpenAI-compatible"
            )
            _log.info(
                "LLM rewrite: %s route (base_url=%s model=%s api_key_env=%s)",
                route,
                resolved.base_url,
                resolved.model,
                resolved.api_key_env,
            )
        elif resolved.available and agent_section == "collection":
            route = (
                "ProxyAPI OpenRouter"
                if "proxyapi.ru" in (resolved.base_url or "")
                else "OpenAI-compatible"
            )
            _log.info(
                "LLM collection: %s route (base_url=%s model=%s api_key_env=%s)",
                route,
                resolved.base_url,
                resolved.model,
                resolved.api_key_env,
            )
        elif resolved.available and (
            resolved.model.strip() == "openrouter/free" or resolved.model.endswith("/openrouter/free")
        ):
            _log.warning(
                "OpenRouter free route detected; using low-throughput safe mode (model=%s base_url=%s)",
                resolved.model,
                resolved.base_url,
            )
        elif resolved.available and "openrouter.ai" in (resolved.base_url or ""):
            _log.info(
                "LLM: OpenRouter-compatible endpoint (model=%s base_url=%s)",
                resolved.model,
                resolved.base_url,
            )

    def get_llm_client(self, profile: Optional[str] = None) -> Optional[ProxyAPIModelClient]:
        explicit = profile or self.llm_profile or self._llm_default_profile
        return self.llm_registry.get_client(
            agent_section=self._llm_agent_section,
            explicit_profile=explicit,
        )

    def llm_generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        profile: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        **extra: Any,
    ) -> str:
        client = self.get_llm_client(profile=profile)
        if client is None:
            cfg = self.llm_registry.resolve(
                agent_section=self._llm_agent_section,
                explicit_profile=profile or self.llm_profile or self._llm_default_profile,
            )
            raise RuntimeError(
                f"LLM client is not available for profile '{cfg.profile_name}'. "
                f"Set env var '{cfg.api_key_env}' or disable LLM mode."
            )
        resolved = self.llm_registry.resolve(
            agent_section=self._llm_agent_section,
            explicit_profile=profile or self.llm_profile or self._llm_default_profile,
        )
        return client.generate(
            prompt,
            system=system,
            temperature=temperature if temperature is not None else resolved.temperature,
            max_tokens=max_tokens if max_tokens is not None else resolved.max_tokens,
            request_timeout=request_timeout,
            **extra,
        )
