from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    if p.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported table format: {p.suffix}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_parquet(src: str | Path, dst: str | Path) -> None:
    df = read_table(src)
    write_parquet(df, dst)
