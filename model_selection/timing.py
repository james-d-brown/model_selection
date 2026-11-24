"""Wall-clock time of each stage of an analysis, for finding where the time
goes. Timing only: nothing here changes a result.

`timed(label)` records how long its block took and logs it at INFO on the
"model_selection.timing" logger, so that a long run reports each stage as it
finishes. Records accumulate per process from `reset()` until `records()` is
read; the pathways reset at their start, and the selection stage attaches the
records to the selection (SelectionResult.timings).
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("model_selection.timing")
_records: list[dict] = []


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - start
        _records.append({"stage": label, "seconds": round(seconds, 3)})
        logger.info(f"[t] {label}: {seconds:.1f} s")


def elapsed_since(label: str, start: float) -> None:
    """Record a stage that started at time.perf_counter() value `start`,
    for a stage that is not one block."""
    seconds = time.perf_counter() - start
    _records.append({"stage": label, "seconds": round(seconds, 3)})
    logger.info(f"[t] {label}: {seconds:.1f} s")


def reset() -> None:
    _records.clear()


def records() -> list[dict]:
    return list(_records)
