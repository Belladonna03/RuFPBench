from __future__ import annotations

import importlib
import json
import os
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Wikimedia/Wiktionary API requires a descriptive User-Agent (see https://meta.wikimedia.org/wiki/User-Agent_policy).
_DEFAULT_WIKIMEDIA_APP = "RuFPBench"
_WIKIMEDIA_UA_VERSION = "0.1"

from .collection_eda import eda as collection_eda_compute
from .collection_eda import save_eda as collection_save_eda
from shared.collection_merge_sample import stratified_sample_by_source
from shared.config import as_config_dict
from shared.llm import LLMEnabledMixin
from shared.logging_utils import get_logger
from shared.mediawiki_wikitext import apply_wikitext_mode
from shared.paths import ProjectPaths
from shared.schemas import COLLECTION_COLUMNS
from shared.utils import dumps_meta, make_uid, utc_now_iso

_log = get_logger("agents.data_collection")

_EDA_LLM_SYSTEM_PROMPT = """You are analyzing a raw text dataset for a Russian-language ML project on benign-borderline prompt classification.

Project context:
- This is not a final clean benchmark yet.
- This is a raw donor pool collected from multiple heterogeneous sources.
- The downstream pipeline includes: collection -> quality filtering -> rewrite -> annotation -> human review -> active learning.
- Some sources are useful as native lexical seeds.
- Some sources are useful only as unsafe donor material for rewrite.
- Some sources may be noisy and only partially useful.

Your task:
Write a high-quality analytical markdown report about the collected raw dataset based on:
1. computed EDA statistics
2. source-level distributions
3. label distributions
4. seed-role distributions
5. several sample rows from each source

Important:
- Do NOT write generic boilerplate like "the dataset is diverse" unless you immediately explain why that matters here.
- Do NOT just restate numbers mechanically.
- Do NOT treat raw donor labels and project labels as equivalent if they are mixed.
- Do NOT pretend this is already a clean final dataset if it is not.
- Focus on dataset quality, source roles, project usefulness, and next-step implications.

You must reason in terms of the project's actual structure:
- native_ru_seed = lexical / idiomatic / phraseological seed layer
- unsafe_donor_ru = unsafe donor pool for rewrite
- safe_seed_en = English structural donor prompts
- plain_benign_control = benign control layer
- noisy_ru_seed = noisy lexical support layer

The report must answer these questions:

1. What exactly was collected?
Explain what kinds of sources are present and what roles they play in the project.
Do not just list source names — explain their function.

2. Which sources look most useful?
Identify the strongest sources for:
- native lexical seeds
- unsafe donor material
- benign controls
- translation/adaptation donors
Be specific.

3. Which sources look noisy or problematic?
Point out sources that are:
- too noisy
- too long
- too heterogeneous
- not prompt-like
- weakly aligned with the final task
Use concrete reasoning, not vague statements.

4. What are the main data quality problems?
You must explicitly check for and discuss:
- dirty or inconsistent label space
- source imbalance
- extreme text-length outliers
- HTML/noise artifacts
- mixed granularity of texts (single lexical items vs long forum posts vs definitions)
- risk of using donor data as if it were final benchmark data

5. What should be done next?
Give practical recommendations for:
- filtering / cleaning
- label normalization
- source prioritization
- rewrite usage
- what should remain donor-only
- what can serve as a strong base for borderline generation

Writing style requirements:
- Write in Russian.
- Be concise but concrete.
- Write like a data analyst / ML researcher, not like a generic assistant.
- Prefer strong source-aware statements over generic praise.
- Avoid filler.
- Avoid vague phrases such as "dataset is informative" unless backed by a reason.
- Use short sections with meaningful headings.
- If something is suspicious or broken, say so directly.
- If label schema looks inconsistent, say so directly.
- If a source is useful only as donor material and not as final data, say so directly.

Output format:
Produce markdown with the following sections:

# Отчёт по EDA raw-датасета

## Краткое описание собранных данных

## Наиболее полезные источники

## Проблемные источники

## Основные проблемы качества данных

## Рекомендации

Additional guidance:
- If you see raw labels like 0/1 mixed with semantic labels, explicitly call this a schema problem.
- If one source dominates the dataset, explicitly discuss why that is risky.
- If lexical wiki sources are the strongest material for borderline generation, say so.
- If toxic/slang sources should be donor-only, say so.
- If the dataset is suitable as a donor pool but not as a final benchmark, say so explicitly.

Grounding: the user message contains JSON with precomputed EDA metrics and sample texts per source. Do not invent or recalculate statistics; rely only on that JSON and the samples."""


def _nullable_str_scalar(v: Any) -> Any:
    """Missing -> pd.NA; else str (for parquet-safe nullable strings, no literal 'nan' for floats)."""
    if v is None:
        return pd.NA
    try:
        if pd.isna(v):
            return pd.NA
    except (ValueError, TypeError):
        return pd.NA
    return str(v)


def normalize_collection_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify string-like columns so pyarrow can write parquet (no object columns mixing int/str).
    """
    out = df.copy()
    if "label" in out.columns:
        before = out["label"].dtype
        out["label"] = out["label"].map(_nullable_str_scalar).astype(pd.StringDtype())
        _log.info("column label dtype before=%s after=%s", before, out["label"].dtype)

    for col in ("uid", "text", "source", "collected_at", "language", "source_type", "url", "meta", "audio", "image"):
        if col not in out.columns or col == "label":
            continue
        if out[col].dtype != object:
            continue
        before = out[col].dtype
        out[col] = out[col].map(_nullable_str_scalar).astype(pd.StringDtype())
        _log.debug("column %s dtype before=%s after=%s", col, before, out[col].dtype)

    return out


def _get_nested(obj: Any, path: list[str]) -> Any:
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


def _http_user_agent(collection_cfg: dict[str, Any]) -> str:
    """User-Agent for HTTP API sources (Wikimedia blocks generic clients without it)."""
    full_override = os.environ.get("RUFBENCH_HTTP_USER_AGENT", "").strip()
    if full_override:
        return full_override
    explicit = (collection_cfg.get("user_agent") or "").strip()
    if explicit:
        return explicit
    app = (os.environ.get("WIKIMEDIA_USER_AGENT_APP") or _DEFAULT_WIKIMEDIA_APP).strip() or _DEFAULT_WIKIMEDIA_APP
    contact = os.environ.get("WIKIMEDIA_CONTACT_EMAIL", "").strip()
    if contact:
        return f"{app}/{_WIKIMEDIA_UA_VERSION} (contact: {contact})"
    return f"{app}/{_WIKIMEDIA_UA_VERSION}"


def _api_request_headers(collection_cfg: dict[str, Any]) -> dict[str, str]:
    return {"User-Agent": _http_user_agent(collection_cfg)}


def _is_mediawiki_categorymembers_source(src: dict[str, Any]) -> bool:
    """api + query categorymembers → paginated MediaWiki API path (not single-shot JSON)."""
    mode = str(src.get("api_mode", "")).lower()
    if mode in ("categorymembers", "mediawiki_categorymembers"):
        return True
    p = src.get("params") or {}
    return p.get("action") == "query" and p.get("list") == "categorymembers"


def _coerce_row_cap(v: Any) -> int | None:
    """Turn config value into a positive row cap, or None = unlimited."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("max_rows must be an integer or null, not a boolean")
    if isinstance(v, (int, float)):
        n = int(v)
        if n <= 0:
            return None
        return n
    raise TypeError(f"max_rows must be an integer or null, got {type(v).__name__}")


def _effective_max_rows(src: dict[str, Any], global_cap: Any) -> int | None:
    """
    Priority: ``sources[].max_rows`` (if key present) else ``collection.max_rows_per_source``.
    ``null`` / omitted cap / non-positive values → no limit for that resolution step.
    """
    if "max_rows" in src:
        return _coerce_row_cap(src.get("max_rows"))
    return _coerce_row_cap(global_cap)


def _default_source_name(value: str, fallback: str) -> str:
    parsed = urlparse(value)
    if parsed.netloc:
        parts = [parsed.netloc]
        if parsed.path and parsed.path != "/":
            tail = parsed.path.strip("/").split("/")[-1]
            if tail:
                parts.append(tail)
        return "::".join(parts)
    return fallback


class DataCollectionAgent(LLMEnabledMixin):
    """Collects unified-schema rows from HF datasets, JSON APIs, and MediaWiki API (category + parse)."""

    def __init__(self, config: str | Path | dict[str, Any] | None = None):
        self.config_path: Path | None = Path(config) if isinstance(config, (str, Path)) else None
        self.cfg = as_config_dict(config or {})
        self.collection_cfg = self.cfg.get("collection") or {}
        root = Path(__file__).resolve().parents[1]
        self.paths = ProjectPaths.from_config(root, self.cfg)
        self.project_name = (self.cfg.get("project") or {}).get("name", "RuFPBench")
        self._init_llm(project_config=self.cfg, agent_section="collection", default_profile="default")

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a public dataset source into the unified dataframe schema.

        The default ``source='hf'`` works out of the box for common text datasets
        such as IMDb (``text`` / ``label`` columns). ``source='kaggle'`` is
        accepted for API compatibility but requires a local export path.
        """
        dataset_source = str(source).lower().strip()
        if dataset_source == "hf":
            src = {
                "type": "hf_dataset",
                "name": name,
                "split": kwargs.get("split", "train"),
                "text_column": kwargs.get("text_column", "text"),
                "label_column": kwargs.get("label_column", "label"),
                "config_name": kwargs.get("config_name"),
                "language": kwargs.get("language"),
                "constant_label": kwargs.get("constant_label"),
                "unified_label": kwargs.get("unified_label"),
                "raw_label_name": kwargs.get("raw_label_name"),
                "filters": kwargs.get("filters") or [],
                "meta": kwargs.get("meta") or {},
                "max_rows": kwargs.get("max_rows"),
            }
            df = self._collect_hf(src)
            max_rows = _coerce_row_cap(kwargs.get("max_rows"))
            if max_rows is not None and len(df) > max_rows:
                df = df.head(max_rows).reset_index(drop=True)
            return df

        if dataset_source == "kaggle":
            csv_path = kwargs.get("csv_path")
            if not csv_path:
                raise ValueError("load_dataset(..., source='kaggle') requires csv_path for the exported dataset")
            df = pd.read_csv(csv_path)
            text_column = kwargs.get("text_column", "text")
            label_column = kwargs.get("label_column", "label")
            source_name = kwargs.get("name") or Path(str(csv_path)).stem or name
            rows: list[dict[str, Any]] = []
            for i, row_in in df.iterrows():
                text = row_in.get(text_column)
                if pd.isna(text) or not str(text).strip():
                    continue
                row = self._empty_row()
                row.update(
                    {
                        "uid": make_uid(source_name, str(text).strip(), str(i)),
                        "text": str(text).strip(),
                        "label": None if label_column not in df.columns else row_in.get(label_column),
                        "source": source_name,
                        "collected_at": utc_now_iso(),
                        "language": kwargs.get("language"),
                        "source_type": "kaggle_dataset",
                        "url": kwargs.get("url"),
                        "meta": dumps_meta(kwargs.get("meta") or {}),
                    }
                )
                rows.append(row)
            return normalize_collection_schema(pd.DataFrame(rows))

        raise ValueError(f"Unsupported dataset source: {source!r}")

    def fetch_api(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch JSON API rows into the unified dataframe schema."""
        src = {
            "type": "api",
            "name": kwargs.get("name") or _default_source_name(endpoint, "api_source"),
            "endpoint": endpoint,
            "params": params or {},
            "records_path": kwargs.get("records_path", []),
            "text_field": kwargs.get("text_field", "text"),
            "label": kwargs.get("label"),
            "language": kwargs.get("language"),
            "meta": kwargs.get("meta") or {},
            "api_mode": kwargs.get("api_mode"),
            "max_rows": kwargs.get("max_rows"),
        }
        df = self._collect_api(src)
        max_rows = _coerce_row_cap(kwargs.get("max_rows"))
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows).reset_index(drop=True)
        return df

    def scrape(
        self,
        url: str,
        selector: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Scrape text items from an HTML page using a CSS selector.

        Each matched element becomes one row in the unified collection schema.
        """
        timeout = float(kwargs.get("request_timeout_s", self.collection_cfg.get("request_timeout_s", 30)))
        headers = _api_request_headers(self.collection_cfg)
        source_name = kwargs.get("name") or _default_source_name(url, "scrape_source")
        _log.info("scrape request source=%s url=%s selector=%s timeout_s=%s", source_name, url, selector, timeout)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            raise RuntimeError(f"scrape source {source_name!r} HTTP {code} for URL {url!r}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"scrape source {source_name!r} request failed for URL {url!r}: {e}") from e

        soup = importlib.import_module("bs4").BeautifulSoup(resp.text, "html.parser")
        nodes = soup.select(selector)
        attr = kwargs.get("attr")
        meta_extra = kwargs.get("meta") or {}
        rows: list[dict[str, Any]] = []
        for i, node in enumerate(nodes):
            if attr:
                text_s = str(node.get(attr, "")).strip()
            else:
                text_s = node.get_text(" ", strip=True)
            if not text_s:
                continue
            row = self._empty_row()
            meta = dict(meta_extra)
            meta["css_selector"] = selector
            meta["scrape_index"] = i
            row.update(
                {
                    "uid": make_uid(source_name, text_s, str(i)),
                    "text": text_s,
                    "label": kwargs.get("label"),
                    "source": source_name,
                    "collected_at": utc_now_iso(),
                    "language": kwargs.get("language"),
                    "source_type": "scrape",
                    "url": url,
                    "meta": dumps_meta(meta),
                }
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        max_rows = _coerce_row_cap(kwargs.get("max_rows"))
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows).reset_index(drop=True)
        df = normalize_collection_schema(df)
        per = bool(self.collection_cfg.get("save_per_source", True))
        if per and len(df):
            out_dir = Path(self.collection_cfg.get("output_dir", self.paths.raw_dir))
            safe = source_name.replace("/", "_")
            df.to_parquet(out_dir / f"{safe}.parquet", index=False)
        return df

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge source dataframes into the standard schema used by the project."""
        if not sources:
            return normalize_collection_schema(pd.DataFrame(columns=COLLECTION_COLUMNS))
        merged = pd.concat(sources, ignore_index=True)
        for col in COLLECTION_COLUMNS:
            if col not in merged.columns:
                merged[col] = None
        merged = merged[COLLECTION_COLUMNS + [c for c in merged.columns if c not in COLLECTION_COLUMNS]]
        merged = normalize_collection_schema(merged)
        source_col = "source"
        merged_cap = _coerce_row_cap(self.collection_cfg.get("max_merged_rows"))
        _MERGED_SAMPLE_RS = 42
        if merged_cap is not None and not merged.empty:
            n_before = len(merged)
            u_before = int(merged[source_col].nunique()) if source_col in merged.columns else 0
            merged = stratified_sample_by_source(
                merged,
                merged_cap,
                source_col=source_col,
                random_state=_MERGED_SAMPLE_RS,
            )
            n_after = len(merged)
            u_after = int(merged[source_col].nunique()) if source_col in merged.columns else 0
            strat = "stratified_by_source" if source_col in merged.columns else "uniform_random"
            _log.info(
                "merged dataset sampled down before=%s after=%s strategy=%s random_state=%s "
                "unique_sources_before=%s unique_sources_after=%s",
                n_before,
                n_after,
                strat,
                _MERGED_SAMPLE_RS,
                u_before,
                u_after,
            )
        return merged

    def _collect_source(self, src: dict[str, Any]) -> pd.DataFrame:
        stype = str(src.get("type", "")).lower().strip()
        if stype == "hf_dataset":
            payload = dict(src)
            payload.pop("type", None)
            name = payload.pop("name")
            return self.load_dataset(name, source="hf", **payload)
        if stype == "kaggle_dataset":
            payload = dict(src)
            payload.pop("type", None)
            name = payload.pop("name")
            return self.load_dataset(name, source="kaggle", **payload)
        if stype == "api":
            payload = dict(src)
            payload.pop("type", None)
            endpoint = payload.pop("endpoint")
            params = payload.pop("params", None)
            return self.fetch_api(endpoint, params, **payload)
        if stype == "scrape":
            payload = dict(src)
            payload.pop("type", None)
            url = payload.pop("url")
            selector = payload.pop("selector")
            return self.scrape(url, selector, **payload)
        if stype == "mediawiki_page":
            return self._collect_mediawiki_page(src)
        raise ValueError(f"Unsupported source type: {stype}")

    def run(self, sources: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        self.paths.ensure()
        out_dir = Path(self.collection_cfg.get("output_dir", self.paths.raw_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        source_specs = sources if sources is not None else (self.collection_cfg.get("sources") or [])
        if not source_specs:
            raise ValueError("No sources provided: pass run(sources=[...]) or configure collection.sources")

        _log.info("sources count=%s output_dir=%s", len(source_specs), out_dir)
        frames: list[pd.DataFrame] = []
        for idx, src in enumerate(source_specs):
            stype = src.get("type")
            label = src.get("name", stype)
            _log.info("source[%s] type=%s name=%s", idx, stype, label)
            frames.append(self._collect_source(src))

        global_cap = self.collection_cfg.get("max_rows_per_source")
        trimmed: list[pd.DataFrame] = []
        for idx, (src, f) in enumerate(zip(source_specs, frames)):
            name = src.get("name", src.get("type"))
            eff = _effective_max_rows(src, global_cap)
            n = len(f)
            _log.info(
                "source[%s] name=%s loaded_rows=%s effective_max_rows=%s",
                idx,
                name,
                n,
                eff if eff is not None else "null (unlimited)",
            )
            if eff is None:
                _log.info("source[%s] trim skipped reason=no_row_limit", idx)
                trimmed.append(f)
                continue
            if n > eff:
                f = f.sample(n=eff, random_state=42).reset_index(drop=True)
                _log.info("source[%s] trimmed_rows=%s (cap=%s)", idx, len(f), eff)
            else:
                _log.info("source[%s] trimmed_rows=%s (under cap)", idx, len(f))
            trimmed.append(f)
        frames = trimmed

        merged = self.merge(frames)
        source_col = "source"
        u_merged = int(merged[source_col].nunique()) if source_col in merged.columns else 0
        _log.info("merge done total_rows=%s unique_sources=%s", len(merged), u_merged)

        save_merged = bool(self.collection_cfg.get("save_merged", True))
        if save_merged:
            merged_path = out_dir / "merged_raw.parquet"
            merged.to_parquet(merged_path, index=False)
            merged.to_csv(out_dir / "merged_raw.csv", index=False)
            _log.info("saved merged_raw parquet=%s csv=%s", merged_path, out_dir / "merged_raw.csv")

        self._run_collection_eda(merged)

        return merged

    def _run_collection_eda(self, merged: pd.DataFrame) -> None:
        eda_cfg = self.collection_cfg.get("eda") or {}
        if not bool(eda_cfg.get("enabled", True)):
            _log.info("collection.eda.enabled is false; skipping EDA")
            return
        out_dir = Path(eda_cfg.get("output_dir", "reports/collection_eda"))
        try:
            report = collection_save_eda(
                merged,
                out_dir,
                top_content_words_k=int(eda_cfg.get("top_content_words_k", 20)),
                content_word_min_len=int(eda_cfg.get("content_word_min_len", 3)),
                include_english_stopwords=bool(eda_cfg.get("include_english_stopwords", True)),
            )
            _log.info("collection code-based EDA completed output_dir=%s", out_dir)
        except Exception as e:
            _log.exception("collection EDA (code) failed: %s", e)
            return

        if not bool(eda_cfg.get("llm_summary_enabled", True)):
            return
        llm_cfg = self.collection_cfg.get("llm") or {}
        if not bool(llm_cfg.get("enabled", True)):
            _log.info("collection.llm.enabled is false; skipping LLM EDA summary")
            return

        llm_path = out_dir / "eda_llm_summary.md"
        try:
            self.summarize_eda(merged, report, llm_path)
        except Exception as e:
            _log.warning(
                "code-based EDA completed; LLM summary failed (collection stage continues): %s",
                e,
                exc_info=True,
            )

    def eda(self, df: pd.DataFrame) -> dict[str, Any]:
        """Code-only EDA statistics (no LLM)."""
        ec = self.collection_cfg.get("eda") or {}
        return collection_eda_compute(
            df,
            top_content_words_k=int(ec.get("top_content_words_k", 20)),
            content_word_min_len=int(ec.get("content_word_min_len", 3)),
            include_english_stopwords=bool(ec.get("include_english_stopwords", True)),
        )

    def save_eda(self, df: pd.DataFrame, out_dir: str | Path) -> dict[str, Any]:
        """Save EDA tables/plots and ``eda_summary.md`` under ``out_dir``."""
        ec = self.collection_cfg.get("eda") or {}
        return collection_save_eda(
            df,
            out_dir,
            top_content_words_k=int(ec.get("top_content_words_k", 20)),
            content_word_min_len=int(ec.get("content_word_min_len", 3)),
            include_english_stopwords=bool(ec.get("include_english_stopwords", True)),
        )

    def summarize_eda(
        self,
        df: pd.DataFrame,
        eda_report: dict[str, Any],
        out_path: str | Path,
    ) -> str:
        """
        LLM interpretation of precomputed ``eda_report``; writes ``eda_llm_summary.md``.
        Does not recompute statistics. Safe to call when LLM is disabled (returns "").
        """
        out_path = Path(out_path)
        llm_cfg = self.collection_cfg.get("llm") or {}
        if not bool(llm_cfg.get("enabled", True)):
            return ""
        if not self.llm_enabled:
            _log.warning("LLM EDA summary skipped: client not available (set %s)", self.llm_config.api_key_env)
            return ""

        _log.info(
            "LLM collection EDA (invoke): base_url=%s model=%s api_key_env=%s",
            self.llm_config.base_url,
            self.llm_config.model,
            self.llm_config.api_key_env,
        )

        # region agent log
        _DBG_LOG = "/Users/dekovaleva/PythonProjects/ru_fp_bench/.cursor/debug-5d942c.log"

        def _dbg_eda(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
            try:
                with open(_DBG_LOG, "a", encoding="utf-8") as _df:
                    _df.write(
                        json.dumps(
                            {
                                "sessionId": "5d942c",
                                "timestamp": int(time.time() * 1000),
                                "hypothesisId": hypothesis_id,
                                "location": location,
                                "message": message,
                                "data": data,
                                "runId": "pre-fix",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass

        _dbg_eda("H-B", "summarize_eda:post_invoke_log", "after invoke log", {})
        # endregion

        t_json = time.monotonic()
        report_json = json.dumps(eda_report, ensure_ascii=False, indent=2, default=str)
        if len(report_json) > 14_000:
            report_json = report_json[:14_000] + "\n…(truncated)…"
        json_ms = (time.monotonic() - t_json) * 1000

        t_samples = time.monotonic()
        samples_md = self._eda_samples_markdown(df)
        samples_ms = (time.monotonic() - t_samples) * 1000

        # region agent log
        _dbg_eda(
            "H-B",
            "summarize_eda:payload_built",
            "json and samples ready",
            {"json_dump_ms": round(json_ms, 2), "samples_ms": round(samples_ms, 2), "report_json_len": len(report_json)},
        )
        # endregion

        system = _EDA_LLM_SYSTEM_PROMPT
        user = (
            "## EDA report (JSON)\n\n```json\n"
            + report_json
            + "\n```\n\n## Примеры по источникам\n\n"
            + samples_md
        )

        eda_section = self.collection_cfg.get("eda") or {}
        raw_eda_timeout = eda_section.get("llm_request_timeout_s")
        if raw_eda_timeout is not None:
            eda_request_timeout = float(raw_eda_timeout)
        else:
            eda_request_timeout = max(float(self.llm_config.timeout_s), 240.0)

        # region agent log
        _dbg_eda(
            "H-A",
            "summarize_eda:before_llm_generate",
            "sizes",
            {
                "system_len": len(system),
                "user_len": len(user),
                "total_chars": len(system) + len(user),
                "eda_request_timeout": eda_request_timeout,
            },
        )
        # endregion

        try:
            t_llm = time.monotonic()
            text = self.llm_generate(user, system=system, request_timeout=eda_request_timeout).strip()
            # region agent log
            _dbg_eda(
                "H-A",
                "summarize_eda:after_llm_generate",
                "llm returned",
                {"elapsed_ms": round((time.monotonic() - t_llm) * 1000, 2), "response_text_len": len(text)},
            )
            # endregion
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
            _log.info("LLM collection EDA summary written %s", out_path)
            return text
        except Exception as e:
            # region agent log
            _dbg_eda(
                "H-A",
                "summarize_eda:llm_exception",
                "llm failed",
                {"error_type": type(e).__name__, "error_msg": str(e)[:500]},
            )
            # endregion
            _log.warning("LLM EDA summary call failed: %s", e, exc_info=True)
            return ""

    def _eda_samples_markdown(self, df: pd.DataFrame, max_per_source: int = 2, clip: int = 500) -> str:
        if df.empty or "source" not in df.columns or "text" not in df.columns:
            return "(no samples)"
        lines: list[str] = []
        for src, sub in df.groupby(df["source"].astype(str)):
            lines.append(f"### {src} (n={len(sub)})")
            for _, row in sub.head(max_per_source).iterrows():
                t = str(row.get("text", ""))[:clip].replace("\n", " ")
                lines.append(f"- {t!r}")
        return "\n".join(lines) if lines else "(no samples)"

    def _empty_row(self) -> dict[str, Any]:
        return {c: None for c in COLLECTION_COLUMNS}

    def _collect_hf(self, src: dict[str, Any]) -> pd.DataFrame:
        from datasets import load_dataset

        name = src["name"]
        split = src.get("split", "train")
        text_column = src["text_column"]
        language = src.get("language")
        source_name = src.get("name", name).split("/")[-1]
        label_column = src.get("label_column")
        constant_label = src.get("constant_label")
        unified_label = src.get("unified_label")
        raw_label_name = src.get("raw_label_name")
        meta_extra = src.get("meta") or {}

        cfg_name = src.get("config_name")
        if cfg_name:
            ds = load_dataset(name, cfg_name, split=split)
        else:
            ds = load_dataset(name, split=split)

        for filt in src.get("filters") or []:
            col = filt["column"]
            op = filt.get("op", "eq")
            val = filt.get("value")
            if op == "in" and isinstance(val, (list, tuple)):

                def _keep(ex: dict[str, Any], c: str = col, allowed: set[Any] = set(val)) -> bool:
                    return ex.get(c) in allowed

                ds = ds.filter(_keep)
            elif op == "eq":

                def _eq(ex: dict[str, Any], c: str = col, v: Any = val) -> bool:
                    return ex.get(c) == v

                ds = ds.filter(_eq)
            else:
                raise ValueError(f"Unsupported filter op: {op}")

        if unified_label is not None:
            _log.info(
                "hf_dataset name=%s using unified_label=%s raw_label_name=%s",
                name,
                unified_label,
                raw_label_name,
            )

        rows: list[dict[str, Any]] = []
        for i, ex in enumerate(ds):
            text = ex.get(text_column)
            if text is None or (isinstance(text, str) and not text.strip()):
                continue
            text_s = str(text).strip()
            raw_hf_label = ex.get(label_column) if label_column else None
            if unified_label is not None:
                label = str(unified_label)
            elif label_column:
                label = raw_hf_label
            else:
                label = constant_label
            uid = make_uid(source_name, text_s, str(i))
            url = ex.get("url") or ex.get("link")
            row = self._empty_row()
            meta = dict(meta_extra)
            meta["hf_index"] = i
            if unified_label is not None and label_column:
                meta["raw_label"] = raw_hf_label
            if unified_label is not None and raw_label_name:
                meta["raw_label_name"] = str(raw_label_name)
            row.update(
                {
                    "uid": uid,
                    "text": text_s,
                    "label": label,
                    "source": source_name,
                    "collected_at": utc_now_iso(),
                    "language": language,
                    "source_type": "hf_dataset",
                    "url": str(url) if url else None,
                    "meta": dumps_meta(meta),
                },
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        per = bool(self.collection_cfg.get("save_per_source", True))
        if per and len(df):
            out_dir = Path(self.collection_cfg.get("output_dir", self.paths.raw_dir))
            safe = source_name.replace("/", "_")
            df.to_parquet(out_dir / f"{safe}.parquet", index=False)
        return df

    def _collect_api(self, src: dict[str, Any]) -> pd.DataFrame:
        if _is_mediawiki_categorymembers_source(src):
            return self._collect_mediawiki_categorymembers(src)
        return self._collect_api_json_once(src)

    def _fetch_mediawiki_categorymembers_all(self, src: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = src["endpoint"]
        base_params = dict(src.get("params") or {})
        base_params.setdefault("format", "json")
        timeout = float(self.collection_cfg.get("request_timeout_s", 30))
        headers = _api_request_headers(self.collection_cfg)
        source_name = src.get("name", "api_source")

        all_members: list[dict[str, Any]] = []
        continue_params: dict[str, Any] = {}
        page_n = 0
        while True:
            params = {**base_params, **continue_params}
            _log.info(
                "mediawiki api_mode=categorymembers source=%s request_index=%s",
                source_name,
                page_n,
            )
            try:
                resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
                resp.raise_for_status()
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                raise RuntimeError(
                    f"API source {source_name!r} HTTP {code} for URL {endpoint!r}"
                ) from e
            except requests.RequestException as e:
                raise RuntimeError(
                    f"API source {source_name!r} request failed for URL {endpoint!r}: {e}"
                ) from e
            data = resp.json()
            batch = data.get("query", {}).get("categorymembers")
            if batch is None:
                raise ValueError(
                    f"MediaWiki categorymembers: missing query.categorymembers for {source_name!r}; check params"
                )
            if not isinstance(batch, list):
                raise ValueError("query.categorymembers is not a list")
            all_members.extend(batch)
            cont = data.get("continue")
            if not cont:
                break
            continue_params = dict(cont)
            page_n += 1
            if page_n > 5000:
                raise RuntimeError(f"categorymembers pagination exceeded safety limit for {source_name!r}")

        _log.info(
            "mediawiki categorymembers source=%s total_members=%s",
            source_name,
            len(all_members),
        )
        return all_members

    def _collect_mediawiki_categorymembers(self, src: dict[str, Any]) -> pd.DataFrame:
        records = self._fetch_mediawiki_categorymembers_all(src)
        return self._dataframe_from_api_record_list(src, records, source_type="api")

    def _collect_api_json_once(self, src: dict[str, Any]) -> pd.DataFrame:
        endpoint = src["endpoint"]
        params = dict(src.get("params") or {})
        records_path: list[str] = src["records_path"]
        source_name = src.get("name", "api_source")
        timeout = float(self.collection_cfg.get("request_timeout_s", 30))
        headers = _api_request_headers(self.collection_cfg)

        _log.info(
            "api request source=%s endpoint=%s timeout_s=%s (single-shot JSON)",
            source_name,
            endpoint,
            timeout,
        )
        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            raise RuntimeError(
                f"API source {source_name!r} HTTP {code} for URL {endpoint!r}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(
                f"API source {source_name!r} request failed for URL {endpoint!r}: {e}"
            ) from e
        data = resp.json()
        records = _get_nested(data, records_path)
        if not isinstance(records, list):
            raise ValueError(f"API records at {records_path} is not a list")
        return self._dataframe_from_api_record_list(src, records, source_type="api")

    def _dataframe_from_api_record_list(
        self,
        src: dict[str, Any],
        records: list[Any],
        *,
        source_type: str,
    ) -> pd.DataFrame:
        text_field = src["text_field"]
        label = src.get("label")
        language = src.get("language")
        source_name = src.get("name", "api_source")
        meta_extra = src.get("meta") or {}
        endpoint = src["endpoint"]

        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                text_s = str(rec).strip()
            else:
                tv = rec.get(text_field)
                text_s = str(tv).strip() if tv is not None else ""
            if not text_s:
                continue
            uid = make_uid(source_name, text_s, str(i))
            row = self._empty_row()
            meta = dict(meta_extra)
            meta["api_index"] = i
            row.update(
                {
                    "uid": uid,
                    "text": text_s,
                    "label": label,
                    "source": source_name,
                    "collected_at": utc_now_iso(),
                    "language": language,
                    "source_type": source_type,
                    "url": endpoint,
                    "meta": dumps_meta(meta),
                },
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        per = bool(self.collection_cfg.get("save_per_source", True))
        if per and len(df):
            out_dir = Path(self.collection_cfg.get("output_dir", self.paths.raw_dir))
            safe = str(source_name).replace("/", "_")
            df.to_parquet(out_dir / f"{safe}.parquet", index=False)
        return df

    def _collect_mediawiki_page(self, src: dict[str, Any]) -> pd.DataFrame:
        endpoint = src["endpoint"]
        params = dict(src.get("params") or {})
        params.setdefault("action", "parse")
        params.setdefault("format", "json")
        if "prop" not in params:
            params["prop"] = "wikitext"

        parse_mode = str(src.get("parse_mode", "mediawiki_wikitext_list"))
        wikitext_cfg = dict(src.get("wikitext_parser_config") or {})

        label = src.get("label")
        language = src.get("language")
        source_name = str(src.get("name", "mediawiki_page"))
        meta_extra = dict(src.get("meta") or {})
        timeout = float(self.collection_cfg.get("request_timeout_s", 30))
        headers = _api_request_headers(self.collection_cfg)

        _log.info(
            "mediawiki_page source=%s endpoint=%s parse_mode=%s params_keys=%s",
            source_name,
            endpoint,
            parse_mode,
            sorted(params.keys()),
        )

        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            raise RuntimeError(
                f"mediawiki_page {source_name!r} HTTP {code} for URL {endpoint!r}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(
                f"mediawiki_page {source_name!r} request failed for URL {endpoint!r}: {e}"
            ) from e

        data = resp.json()
        parse_block = data.get("parse") or {}
        wikitext = ""
        wt = parse_block.get("wikitext")
        if isinstance(wt, dict):
            wikitext = str(wt.get("*") or "")
        elif isinstance(wt, str):
            wikitext = wt

        if not wikitext.strip():
            raise RuntimeError(
                f"mediawiki_page {source_name!r}: empty wikitext (check title/page param and API errors: {data.get('error', {})})"
            )

        texts = apply_wikitext_mode(parse_mode, wikitext, **wikitext_cfg)
        _log.info("mediawiki_page source=%s parse_mode=%s extracted_lines=%s", source_name, parse_mode, len(texts))

        rows: list[dict[str, Any]] = []
        for i, text_s in enumerate(texts):
            text_s = str(text_s).strip()
            if not text_s:
                continue
            uid = make_uid(source_name, text_s, str(i))
            row = self._empty_row()
            meta = dict(meta_extra)
            meta["mediawiki_parse_mode"] = parse_mode
            meta["wikitext_line_index"] = i
            row.update(
                {
                    "uid": uid,
                    "text": text_s,
                    "label": label,
                    "source": source_name,
                    "collected_at": utc_now_iso(),
                    "language": language,
                    "source_type": "mediawiki_page",
                    "url": endpoint,
                    "meta": dumps_meta(meta),
                },
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        per = bool(self.collection_cfg.get("save_per_source", True))
        if per and len(df):
            out_dir = Path(self.collection_cfg.get("output_dir", self.paths.raw_dir))
            safe = source_name.replace("/", "_")
            df.to_parquet(out_dir / f"{safe}.parquet", index=False)
        return df


__all__ = ["DataCollectionAgent", "normalize_collection_schema"]
