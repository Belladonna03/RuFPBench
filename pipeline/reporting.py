from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_final_report(
    *,
    cfg: dict[str, Any],
    reports_dir: Path,
    artifacts: dict[str, Any],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    proj = cfg.get("project") or {}
    lines = [
        f"# Final report: {proj.get('name', 'RuFPBench')}",
        "",
        "## Artifacts",
    ]
    for k, v in artifacts.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Configuration snapshot")
    lines.append(f"- Task: {proj.get('task', '')}")
    lines.append(f"- Modality: {proj.get('modality', '')}")
    body = "\n".join(lines) + "\n"
    out = reports_dir / "final_report.md"
    out.write_text(body, encoding="utf-8")
    return out


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", df.to_string(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
