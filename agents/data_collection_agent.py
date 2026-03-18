"""
DataCollectionAgent for course "Сбор и обработка данных" (Assignment 1).

This agent collects raw data from multiple sources and returns a unified pandas.DataFrame.

Supported source types:
- hf_dataset: Hugging Face datasets via `datasets.load_dataset`
- kaggle_dataset: Kaggle datasets via Kaggle API (optional; requires credentials)
- scrape: HTML scraping via requests + BeautifulSoup (CSS selector)
- api: REST API fetch via requests (JSON → records extraction)

Output schema (fixed columns):
- text: str | None
- audio: str | None            # path or url (optional)
- image: str | None            # path or url (optional)
- label: str | int | None      # weak label at collection stage is OK
- source: str                  # source name
- collected_at: str            # ISO timestamp UTC
- language: str | None         # e.g. "ru"/"en"
- source_type: str             # hf_dataset/kaggle_dataset/scrape/api
- url: str | None              # origin url (for scrape/api)
- meta: str | None             # JSON-encoded metadata
- uid: str                     # stable id (sha1 over text+source+seed_role)

The agent is intentionally "collection-only": transformation steps (translation, rewrite) should
live in later pipeline stages (or in an optional appendix step after collection).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import hashlib
import json
import subprocess

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

try:
    from datasets import load_dataset as hf_load_dataset  # type: ignore
except Exception:  # pragma: no cover
    hf_load_dataset = None  # type: ignore


SourceType = Literal["hf_dataset", "kaggle_dataset", "scrape", "api"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]


def extract_records(obj: Any, records_path: Optional[list[str]] = None) -> list[Any]:
    """
    Extract list-like records from a JSON object.

    - If records_path is None:
        - if obj is list -> obj
        - if obj is dict -> [obj]
    - If records_path is provided, traverses dict keys and expects a list at the end.
    """
    if records_path is None:
        if isinstance(obj, list):
            return obj
        return [obj]

    cur = obj
    for key in records_path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            raise ValueError(f"records_path traversal failed at key='{key}'. Got type={type(cur)}")
    if isinstance(cur, list):
        return cur
    # sometimes API returns dict of dicts; allow dict → list(values)
    if isinstance(cur, dict):
        return list(cur.values())
    return [cur]


@dataclass
class AgentConfig:
    output_dir: Path
    request_timeout_s: int = 30
    user_agent: str = "ru-fpbench-data-collector/0.1 (+https://example.local)"
    save_per_source: bool = True
    save_merged: bool = True
    max_rows_per_source: Optional[int] = None


class DataCollectionAgent:
    """
    DataCollectionAgent(config='config.yaml').

    You can either pass sources explicitly into run(...), or put `sources:` into config.yaml.
    """

    STANDARD_COLUMNS = [
        "uid",
        "text",
        "audio",
        "image",
        "label",
        "source",
        "collected_at",
        "language",
        "source_type",
        "url",
        "meta",
    ]

    def __init__(self, config: str | dict[str, Any] = "config.yaml") -> None:
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
        else:
            raw_cfg = config

        out_dir = Path(raw_cfg.get("output_dir", "data/raw"))
        out_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = AgentConfig(
            output_dir=out_dir,
            request_timeout_s=int(raw_cfg.get("request_timeout_s", 30)),
            user_agent=str(raw_cfg.get("user_agent", "ru-fpbench-data-collector/0.1 (+https://example.local)")),
            save_per_source=bool(raw_cfg.get("save_per_source", True)),
            save_merged=bool(raw_cfg.get("save_merged", True)),
            max_rows_per_source=raw_cfg.get("max_rows_per_source"),
        )

        self.raw_cfg = raw_cfg

    # -------------------------
    # Skills required by course
    # -------------------------

    def scrape(self, url: str, selector: str, *, source_name: str, label: Any = None,
               language: Optional[str] = None, extra_meta: Optional[dict[str, Any]] = None,
               max_items: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape HTML page and extract text nodes using CSS selector.
        Returns a unified DataFrame.
        """
        headers = {"User-Agent": self.cfg.user_agent}
        r = requests.get(url, headers=headers, timeout=self.cfg.request_timeout_s)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "lxml")
        nodes = soup.select(selector)
        texts: list[str] = []
        for n in nodes:
            t = n.get_text(" ", strip=True)
            if t:
                texts.append(t)

        if max_items is not None:
            texts = texts[:max_items]

        meta = extra_meta or {}
        meta.update({"selector": selector})

        df = pd.DataFrame({
            "text": texts,
            "label": label,
            "source": source_name,
            "collected_at": utc_now_iso(),
            "language": language,
            "source_type": "scrape",
            "url": url,
            "meta": [json.dumps(meta, ensure_ascii=False)] * len(texts),
        })
        return self._finalize(df, seed_role=meta.get("seed_role"))

    def fetch_api(self, endpoint: str, params: dict[str, Any], *,
                  source_name: str,
                  records_path: Optional[list[str]] = None,
                  text_field: str = "text",
                  label: Any = None,
                  language: Optional[str] = None,
                  extra_meta: Optional[dict[str, Any]] = None,
                  max_items: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch JSON from API and extract records into unified DataFrame.

        The API response can be:
        - list[dict]
        - dict with nested list (use records_path)

        text_field can point to an existing field inside each record.
        """
        headers = {"User-Agent": self.cfg.user_agent}
        r = requests.get(endpoint, params=params, headers=headers, timeout=self.cfg.request_timeout_s)
        r.raise_for_status()
        payload = r.json()

        records = extract_records(payload, records_path=records_path)
        if max_items is not None:
            records = records[:max_items]

        texts: list[str] = []
        metas: list[str] = []
        for rec in records:
            if isinstance(rec, dict):
                t = rec.get(text_field)
                # allow fallback
                if t is None and "title" in rec and text_field != "title":
                    t = rec.get("title")
                if t is None:
                    # skip empty records
                    continue
                texts.append(str(t))
                meta = extra_meta.copy() if extra_meta else {}
                meta.update({"endpoint": endpoint, "params": params, "record": {k: rec.get(k) for k in list(rec.keys())[:10]}})
                metas.append(json.dumps(meta, ensure_ascii=False))
            else:
                texts.append(str(rec))
                meta = extra_meta.copy() if extra_meta else {}
                meta.update({"endpoint": endpoint, "params": params})
                metas.append(json.dumps(meta, ensure_ascii=False))

        df = pd.DataFrame({
            "text": texts,
            "label": label,
            "source": source_name,
            "collected_at": utc_now_iso(),
            "language": language,
            "source_type": "api",
            "url": endpoint,
            "meta": metas,
        })
        return self._finalize(df, seed_role=(extra_meta or {}).get("seed_role"))

    def load_dataset(self, name: str, source: Literal["hf", "kaggle"] = "hf", *,
                     source_name: Optional[str] = None,
                     split: str = "train",
                     config_name: Optional[str] = None,
                     text_column: str = "text",
                     label_column: Optional[str] = None,
                     constant_label: Any = None,
                     filters: Optional[list[dict[str, Any]]] = None,
                     language: Optional[str] = None,
                     extra_meta: Optional[dict[str, Any]] = None,
                     max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load open dataset from Hugging Face or Kaggle and map it into unified schema.

        filters supports simple operations:
          - {"column": "subset", "op": "in", "value": ["xstest-should-respond"]}
          - {"column": "label", "op": "==", "value": 1}
        """
        if source_name is None:
            source_name = name if source == "hf" else f"kaggle:{name}"

        if source == "hf":
            df = self._load_hf_dataset(name, split=split, config_name=config_name)
        elif source == "kaggle":
            df = self._load_kaggle_dataset(name)  # downloads into cache and reads as table
        else:
            raise ValueError(f"Unsupported source: {source}")

        if filters:
            df = self._apply_filters(df, filters)

        if max_rows is not None:
            df = df.head(int(max_rows))

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in dataset '{name}'. "
                f"Available columns: {list(df.columns)[:30]}"
            )

        out = pd.DataFrame({
            "text": df[text_column].astype(str),
            "label": (df[label_column] if (label_column and label_column in df.columns) else constant_label),
            "source": source_name,
            "collected_at": utc_now_iso(),
            "language": language,
            "source_type": "hf_dataset" if source == "hf" else "kaggle_dataset",
            "url": None,
            "meta": [json.dumps(extra_meta or {}, ensure_ascii=False)] * len(df),
        })
        return self._finalize(out, seed_role=(extra_meta or {}).get("seed_role"))

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple unified DataFrames, enforce schema, remove obvious bad rows.
        """
        if not sources:
            return ensure_columns(pd.DataFrame(), self.STANDARD_COLUMNS)

        merged = pd.concat(sources, ignore_index=True)

        # basic normalization
        merged["text"] = merged["text"].astype(str).fillna("").str.strip()
        merged.loc[merged["text"] == "", "text"] = None
        merged = merged[merged["text"].notna()]

        # drop duplicates by uid (stable)
        merged = merged.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)

        # enforce fixed schema
        merged = ensure_columns(merged, self.STANDARD_COLUMNS)
        return merged

    # -------------------------
    # High-level run
    # -------------------------

    def run(self, sources: Optional[list[dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Collect from sources and return a unified DataFrame.

        If sources is None, reads config.yaml -> sources.
        """
        sources = sources or self.raw_cfg.get("sources", [])
        if not sources:
            raise ValueError("No sources provided. Pass `sources=[...]` or set `sources:` in config.yaml.")

        dfs: list[pd.DataFrame] = []
        for src in sources:
            stype: SourceType = src["type"]
            name = src.get("name") or src.get("source_name") or stype

            if stype == "hf_dataset":
                df = self.load_dataset(
                    name=src["name"],
                    source="hf",
                    source_name=name,
                    split=src.get("split", "train"),
                    config_name=src.get("config_name"),
                    text_column=src.get("text_column", "text"),
                    label_column=src.get("label_column"),
                    constant_label=src.get("constant_label", src.get("label")),
                    filters=src.get("filters"),
                    language=src.get("language"),
                    extra_meta=src.get("meta"),
                    max_rows=src.get("max_rows", self.cfg.max_rows_per_source),
                )
            elif stype == "kaggle_dataset":
                df = self.load_dataset(
                    name=src["name"],
                    source="kaggle",
                    source_name=name,
                    text_column=src.get("text_column", "text"),
                    label_column=src.get("label_column"),
                    constant_label=src.get("constant_label", src.get("label")),
                    filters=src.get("filters"),
                    language=src.get("language"),
                    extra_meta=src.get("meta"),
                    max_rows=src.get("max_rows", self.cfg.max_rows_per_source),
                )
            elif stype == "scrape":
                df = self.scrape(
                    url=src["url"],
                    selector=src["selector"],
                    source_name=name,
                    label=src.get("label"),
                    language=src.get("language"),
                    extra_meta=src.get("meta"),
                    max_items=src.get("max_items", self.cfg.max_rows_per_source),
                )
            elif stype == "api":
                df = self.fetch_api(
                    endpoint=src["endpoint"],
                    params=src.get("params", {}),
                    source_name=name,
                    records_path=src.get("records_path"),
                    text_field=src.get("text_field", "text"),
                    label=src.get("label"),
                    language=src.get("language"),
                    extra_meta=src.get("meta"),
                    max_items=src.get("max_items", self.cfg.max_rows_per_source),
                )
            else:
                raise ValueError(f"Unknown source type: {stype}")

            if self.cfg.save_per_source:
                self._save_df(df, self.cfg.output_dir / f"{self._safe_filename(name)}.parquet")

            dfs.append(df)

        merged = self.merge(dfs)

        if self.cfg.save_merged:
            self._save_df(merged, self.cfg.output_dir / "merged_raw.parquet")
            self._save_df(merged, self.cfg.output_dir / "merged_raw.csv")

        return merged

    # -------------------------
    # Internal helpers
    # -------------------------

    def _finalize(self, df: pd.DataFrame, seed_role: Optional[str] = None) -> pd.DataFrame:
        """
        Ensure fixed schema + create uid.
        """
        df = df.copy()
        df["audio"] = None
        df["image"] = None
        if "source" not in df.columns:
            df["source"] = "unknown"
        if "collected_at" not in df.columns:
            df["collected_at"] = utc_now_iso()
        if "meta" not in df.columns:
            df["meta"] = None

        # seed_role is stored inside meta, but also influences uid so different transforms won't collide
        # (If you later generate rewrites, set seed_role differently.)
        if seed_role is None:
            # try to extract from meta json (first row)
            try:
                if len(df) > 0 and isinstance(df.iloc[0].get("meta"), str):
                    m = json.loads(df.iloc[0]["meta"])
                    seed_role = m.get("seed_role")
            except Exception:
                seed_role = None

        def _make_uid(row: pd.Series) -> str:
            t = str(row.get("text") or "")
            src = str(row.get("source") or "")
            role = str(seed_role or "")
            return sha1_hex(f"{src}::{role}::{t}")

        df["uid"] = df.apply(_make_uid, axis=1)

        # enforce fixed schema order
        df = ensure_columns(df, self.STANDARD_COLUMNS)

        # final clean
        df["text"] = df["text"].astype(str).fillna("").str.strip()
        df.loc[df["text"] == "", "text"] = None
        df = df[df["text"].notna()].reset_index(drop=True)
        return df

    def _load_hf_dataset(self, name: str, *, split: str, config_name: Optional[str] = None) -> pd.DataFrame:
        if hf_load_dataset is None:
            raise ImportError("datasets is not installed. `pip install datasets`")
        if config_name:
            ds = hf_load_dataset(name, config_name, split=split)
        else:
            ds = hf_load_dataset(name, split=split)
        return ds.to_pandas()

    def _load_kaggle_dataset(self, dataset: str) -> pd.DataFrame:
        """
        Download Kaggle dataset to cache and return as DataFrame.

        Requires:
          - `pip install kaggle`
          - KAGGLE_USERNAME, KAGGLE_KEY or ~/.kaggle/kaggle.json
        """
        cache_dir = self.cfg.output_dir / "_kaggle_cache" / self._safe_filename(dataset)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download+unzip
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", str(cache_dir),
            "--unzip",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError("Kaggle CLI not found. Install with `pip install kaggle`.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Kaggle download failed. Ensure Kaggle credentials are set.\n"
                f"stderr:\n{e.stderr}"
            ) from e

        # Heuristic: read the first CSV/JSONL in the cache dir
        files = list(cache_dir.glob("**/*"))
        table_files = [p for p in files if p.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}]
        if not table_files:
            raise RuntimeError(f"No readable table files found in Kaggle dataset cache: {cache_dir}")

        p = sorted(table_files, key=lambda x: x.stat().st_size, reverse=True)[0]

        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(p, lines=True)
        raise RuntimeError(f"Unsupported Kaggle file format: {p}")

    def _apply_filters(self, df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
        out = df
        for f in filters:
            col = f["column"]
            op = f.get("op", "==")
            val = f["value"]
            if col not in out.columns:
                raise ValueError(f"Filter column '{col}' not in dataset columns.")
            if op == "==":
                out = out[out[col] == val]
            elif op == "!=":
                out = out[out[col] != val]
            elif op == "in":
                out = out[out[col].isin(val)]
            elif op == "not_in":
                out = out[~out[col].isin(val)]
            else:
                raise ValueError(f"Unsupported filter op: {op}")
        return out.reset_index(drop=True)

    def _save_df(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=False)
        elif path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            # default to parquet
            df.to_parquet(path.with_suffix(".parquet"), index=False)

    @staticmethod
    def _safe_filename(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_")
