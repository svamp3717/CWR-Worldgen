# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic retries for temporary Overpass API outages."""
from __future__ import annotations

import re
import time
from typing import Any

_TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_TEXT = (
    "timed out",
    "timeout",
    "temporary failure",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
    "network is unreachable",
    "remote end closed",
    "empty response",
)


def _is_transient_overpass_failure(exc: BaseException) -> bool:
    text = str(exc)
    if not text.startswith("all Overpass endpoints failed:"):
        return False
    http_codes = [int(value) for value in re.findall(r"\bHTTP\s+(\d{3})\b", text, flags=re.IGNORECASE)]
    if any(code in _TRANSIENT_HTTP_CODES for code in http_codes):
        return True
    lowered = text.casefold()
    if any(marker in lowered for marker in _TRANSIENT_TEXT):
        return True
    if http_codes:
        return False
    # Network-library exceptions often vary by platform and locale. If every
    # endpoint failed without an HTTP status, one bounded retry cycle is safer
    # than immediately killing a long-running world build.
    return True


def _format_delay(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    return f"{minutes}:{remainder:02d}"


def install_overpass_retries() -> None:
    """Wrap the shared Overpass fetcher with bounded automatic retries."""
    from . import location_example as location

    original = location._fetch_overpass
    if bool(getattr(original, "_cwr_overpass_retries", False)):
        return

    def fetch_overpass_with_retries(
        *args: Any,
        retry_rounds: int = 3,
        retry_delay_seconds: float = 120.0,
        retry_countdown_seconds: float = 30.0,
        **kwargs: Any,
    ) -> str:
        if retry_rounds < 1:
            raise ValueError("Overpass retry rounds must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("Overpass retry delay cannot be negative")
        if retry_countdown_seconds <= 0:
            raise ValueError("Overpass retry countdown interval must be positive")

        progress_callback = kwargs.get("progress_callback")
        last_error: RuntimeError | None = None
        for round_index in range(1, retry_rounds + 1):
            round_start = round((round_index - 1) * 95 / retry_rounds)
            round_end = round(round_index * 95 / retry_rounds)

            def round_progress(percent: int, stage: str) -> None:
                if progress_callback is None:
                    return
                bounded = max(0, min(100, int(percent)))
                mapped = round_start + round((round_end - round_start) * bounded / 100)
                progress_callback(mapped, stage)

            call_kwargs = dict(kwargs)
            call_kwargs["progress_callback"] = round_progress if progress_callback is not None else None
            try:
                selected = original(*args, **call_kwargs)
            except RuntimeError as exc:
                last_error = exc
                if round_index >= retry_rounds or not _is_transient_overpass_failure(exc):
                    break
            else:
                if progress_callback is not None:
                    progress_callback(100, "Overpass source ready")
                return selected

            next_round = round_index + 1
            remaining = float(retry_delay_seconds)
            if progress_callback is not None:
                progress_callback(
                    round_end,
                    "All Overpass endpoints are temporarily unavailable "
                    f"(round {round_index}/{retry_rounds}). Automatic retry "
                    f"{next_round}/{retry_rounds} in {_format_delay(remaining)}.",
                )
            while remaining > 0:
                sleep_for = min(float(retry_countdown_seconds), remaining)
                time.sleep(sleep_for)
                remaining = max(0.0, remaining - sleep_for)
                if progress_callback is not None and remaining > 0:
                    progress_callback(
                        round_end,
                        "Overpass servers are still unavailable; automatic retry "
                        f"{next_round}/{retry_rounds} in {_format_delay(remaining)}.",
                    )

        assert last_error is not None
        if not _is_transient_overpass_failure(last_error):
            raise last_error
        details = str(last_error)
        prefix = "all Overpass endpoints failed:\n"
        if details.startswith(prefix):
            details = details[len(prefix) :]
        raise RuntimeError(
            f"Overpass is still unavailable after {retry_rounds} automatic attempt round(s). "
            "This is usually temporary server overload or a network problem, not a problem "
            "with your selected terrain. Existing source files are safe; try again later.\n"
            "Last endpoint errors:\n"
            + details
        ) from last_error

    fetch_overpass_with_retries._cwr_overpass_retries = True  # type: ignore[attr-defined]
    location._fetch_overpass = fetch_overpass_with_retries
