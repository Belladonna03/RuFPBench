from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, TypeVar

from rich.console import Console

T = TypeVar("T")

STEP_LABELS: dict[str, str] = {
    "generate": "Генерация промпта",
    "mutate": "Мутация промпта",
    "discriminate": "Discriminator",
    "scout": "Scout targets",
    "safety": "Prompt safety judges",
    "final": "Final targets",
    "response_gen": "Генерация safe-ответа",
    "response_val": "Валидация ответа",
    "qc": "QC / dedupe",
    "round_jobs": "Generation jobs",
    "fast_filter": "Fast prompt filter",
    "full_safety": "Full safety judge",
}


def progress_enabled() -> bool:
    return os.getenv("RUFP_DISABLE_PROGRESS", os.getenv("RUFP_DISABLE_TQDM", "0")).lower() not in {
        "1",
        "true",
        "yes",
    }


class RunProgress:
    """Structured console phases + tqdm-friendly task labels for long runs."""

    def __init__(self, console: Console | None = None, *, enabled: bool | None = None) -> None:
        self.console = console or Console()
        self.enabled = progress_enabled() if enabled is None else enabled
        self.round_no = 0
        self.round_total = 0
        self.seed_no = 0
        self.seed_total = 0
        self.iteration_no = 0
        self.current_seed_id = ""

    def banner(self, *, mode: str, run_dir: str, generator: str, targets: str) -> None:
        if not self.enabled:
            return
        self.console.print("[bold]RuFPBench pipeline[/bold]")
        self.console.print(f"  mode: {mode}")
        self.console.print(f"  run_dir: {run_dir}")
        self.console.print(f"  generator: {generator}")
        self.console.print(f"  targets: {targets}")

    def round_begin(self, round_no: int, round_total: int, *, generated: int, accepted: int, fp: int, seeds: int) -> None:
        self.round_no = round_no
        self.round_total = round_total
        self.seed_no = 0
        self.seed_total = seeds
        if not self.enabled:
            return
        self.console.print(
            f"\n[bold cyan]══ Раунд {round_no}/{round_total}[/bold cyan] "
            f"[dim]generated={generated} accepted={accepted} FP={fp} seeds={seeds}[/dim]"
        )

    def seed_begin(self, seed_no: int, seed_total: int, *, seed_id: str, category: str) -> None:
        self.seed_no = seed_no
        self.seed_total = seed_total
        self.iteration_no = 0
        self.current_seed_id = seed_id
        if not self.enabled:
            return
        self.console.print(
            f"[bold]Seed {seed_no}/{seed_total}[/bold] "
            f"[dim]{seed_id} · {category}[/dim]"
        )

    def iteration_begin(self, iteration_no: int, iteration_total: int, *, action: str) -> None:
        self.iteration_no = iteration_no
        if not self.enabled:
            return
        label = STEP_LABELS.get(action, action)
        self.console.print(
            f"  [cyan]▶[/cyan] [bold]{self._prefix()} · итерация {iteration_no}/{iteration_total} · {label}[/bold]"
        )

    def step(self, step_key: str, detail: str = "") -> None:
        if not self.enabled:
            return
        label = STEP_LABELS.get(step_key, step_key)
        suffix = f" [dim]— {detail}[/dim]" if detail else ""
        self.console.print(f"  [cyan]▶[/cyan] [bold]{self._prefix()} · {label}[/bold]{suffix}")

    def note(self, message: str) -> None:
        if not self.enabled:
            return
        self.console.print(f"  [dim]· {self._prefix()} · {message}[/dim]")

    def seed_end(self, *, accepted: bool, reason: str, generated: int, target_calls: int) -> None:
        if not self.enabled:
            return
        if accepted:
            self.console.print(
                f"  [green]✓ принят[/green] [dim]{self._prefix()} · candidates={generated} target_calls={target_calls}[/dim]"
            )
        else:
            self.console.print(
                f"  [yellow]✗ отклонён[/yellow] [dim]{self._prefix()} · {reason} · candidates={generated} target_calls={target_calls}[/dim]"
            )

    def parallel_desc(self, step_key: str, task_count: int) -> str:
        label = STEP_LABELS.get(step_key, step_key)
        return f"{self._prefix()} · {label} ({task_count})"

    def task_label(self, step_key: str, item: Any) -> str:
        label = STEP_LABELS.get(step_key, step_key)
        if isinstance(item, tuple) and len(item) >= 2:
            candidate, model = item[0], item[1]
            prompt_id = getattr(candidate, "prompt_id", str(candidate))
            return f"{label}: {model} · {prompt_id[:16]}"
        return label

    def track(self, items: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
        if not self.enabled or not desc:
            yield from items
            return
        try:
            from tqdm.auto import tqdm
        except Exception:
            yield from items
            return
        count = total if total is not None else len(items)  # type: ignore[arg-type]
        yield from tqdm(
            items,
            total=count,
            desc=desc,
            dynamic_ncols=True,
            mininterval=float(os.getenv("RUFP_TQDM_MININTERVAL", "0.3")),
            file=sys.stderr,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=False,
        )

    def _prefix(self) -> str:
        parts: list[str] = []
        if self.round_no:
            parts.append(f"R{self.round_no}/{self.round_total or '?'}")
        if self.seed_no and self.seed_total:
            parts.append(f"seed {self.seed_no}/{self.seed_total}")
        if self.iteration_no:
            parts.append(f"it {self.iteration_no}")
        return " · ".join(parts) if parts else "run"
