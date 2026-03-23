"""Centralized console logging for RuFPBench (standard library only)."""

from __future__ import annotations

import logging
import sys
from typing import Any

ROOT_LOGGER = "rufpbench"


class _ComponentFormatter(logging.Formatter):
    """Strip `rufpbench.` prefix so logs read e.g. `[pipeline]` not `[rufpbench.pipeline]`."""

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        prefix = ROOT_LOGGER + "."
        if name.startswith(prefix):
            name = name[len(prefix) :]
        record.component = name  # type: ignore[attr-defined]
        return super().format(record)


def configure_logging(level: str | int = "INFO") -> None:
    """Configure the `rufpbench` logger tree once: human-readable stderr, default INFO."""
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)

    log = logging.getLogger(ROOT_LOGGER)
    log.setLevel(lvl)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(lvl)
    h.setFormatter(
        _ComponentFormatter(
            fmt="%(levelname)-5s [%(component)s] %(message)s",
            datefmt=None,
        )
    )
    log.addHandler(h)
    log.propagate = False


def get_logger(component: str) -> logging.Logger:
    """
    Return a logger under `rufpbench.<component>`.

    Examples: get_logger("pipeline"), get_logger("agents.data_collection").
    """
    c = component.strip()
    if c.startswith(f"{ROOT_LOGGER}."):
        return logging.getLogger(c)
    return logging.getLogger(f"{ROOT_LOGGER}.{c}")


def log_dataframe_summary(logger: logging.Logger, df: Any, *, name: str = "df") -> None:
    """Log row count safely (no DataFrame dump)."""
    try:
        n = len(df)
    except Exception:
        logger.debug("%s: could not determine length", name)
        return
    logger.info("%s rows=%s", name, n)
