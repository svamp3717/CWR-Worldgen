# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import os
import time
from typing import Iterator


_PROGRESS_RANGE: ContextVar[tuple[float, float]] = ContextVar(
    "cwr_progress_range", default=(0.0, 100.0)
)
_PROGRESS_STARTED_AT: ContextVar[float | None] = ContextVar(
    "cwr_progress_started_at", default=None
)


@contextmanager
def progress_range(start: float, end: float) -> Iterator[None]:
    """Map nested 0..100 progress reports into a slice of the parent range."""

    if end < start:
        raise ValueError("progress range end must not be below start")
    parent_start, parent_end = _PROGRESS_RANGE.get()
    span = parent_end - parent_start
    mapped_start = parent_start + span * max(0.0, min(100.0, float(start))) / 100.0
    mapped_end = parent_start + span * max(0.0, min(100.0, float(end))) / 100.0
    token = _PROGRESS_RANGE.set((mapped_start, mapped_end))
    try:
        yield
    finally:
        _PROGRESS_RANGE.reset(token)


def start_progress_session() -> None:
    """Start or restart the elapsed-time clock used by progress reports."""

    _PROGRESS_STARTED_AT.set(time.perf_counter())


def reset_progress_session() -> None:
    """Clear the elapsed-time clock, primarily for tests and embedded callers."""

    _PROGRESS_STARTED_AT.set(None)


def progress_elapsed_seconds() -> float:
    """Return elapsed seconds since the active progress session began."""

    started_at = _PROGRESS_STARTED_AT.get()
    if started_at is None:
        start_progress_session()
        started_at = _PROGRESS_STARTED_AT.get()
    assert started_at is not None
    return max(0.0, time.perf_counter() - started_at)


def format_duration(seconds: float, *, milliseconds: bool = True) -> str:
    """Format an elapsed duration as HH:MM:SS, optionally with milliseconds."""

    if not milliseconds:
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, whole_seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"

    total_milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _progress_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def report_progress(percent: int, stage: str) -> None:
    """Emit timestamped build progress for terminal and GUI front-ends."""

    if os.environ.get("CWR_PROGRESS", "") not in {"1", "true", "yes", "machine"}:
        return
    raw = max(0, min(100, int(percent)))
    start, end = _PROGRESS_RANGE.get()
    value = max(0, min(100, int(round(start + (end - start) * raw / 100.0))))
    timestamp = _progress_timestamp()
    elapsed = format_duration(progress_elapsed_seconds())
    mode = os.environ.get("CWR_PROGRESS_FORMAT", "human").casefold()
    if mode == "machine":
        # Keep percent and stage in their historic first three fields. Newer
        # consumers can read the appended timestamp and elapsed duration while
        # older parsers continue to receive usable progress events.
        print(f"CWR_PROGRESS\t{value}\t{stage}\t{timestamp}\t{elapsed}", flush=True)
        return
    width = 28
    filled = round(width * value / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{timestamp}] [+{elapsed}] [{bar}] {value:3d}%  {stage}", flush=True)
