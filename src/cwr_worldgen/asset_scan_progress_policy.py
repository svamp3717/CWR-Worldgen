# SPDX-License-Identifier: GPL-3.0-or-later
"""Expose targeted CWA scan cache behavior and exhaustive fallbacks in progress."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence


_INSTALLED = False


def install_asset_scan_progress_policy() -> None:
    """Wrap generator asset validation with useful timing/cache diagnostics."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import fast_asset_scan_policy as fast
    from . import generator
    from .progress import report_progress

    original_scan = generator.scan_assets

    def scan_assets_with_progress(
        roots: Sequence[Path],
        selected_models: Iterable[str],
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        refresh: bool = False,
    ):
        selected = tuple(str(value) for value in selected_models)
        if not roots:
            return original_scan(
                roots,
                selected,
                cache_dir=cache_dir,
                use_cache=use_cache,
                refresh=refresh,
            )

        report_progress(
            82,
            f"Validating {len(selected)} selected CWA assets with targeted package lookup",
        )
        started = perf_counter()
        result, stats = fast._targeted_scan(
            roots,
            selected,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh=refresh,
        )
        targeted_seconds = perf_counter() - started

        if not result.missing_models and not result.missing_dependencies:
            report_progress(
                82,
                "Targeted CWA asset scan complete "
                f"({len(result.records)} records; PBO indexes "
                f"{stats.index_hits} hits/{stats.index_misses} misses; "
                f"{targeted_seconds:.2f}s)",
            )
            return result

        missing_count = len(result.missing_models) + len(result.missing_dependencies)
        report_progress(
            82,
            "Targeted CWA lookup unresolved "
            f"{missing_count} asset/dependency item(s) after {targeted_seconds:.2f}s; "
            "falling back to exhaustive asset scan",
        )
        fallback_started = perf_counter()
        full = fast._FULL_SCAN(
            roots,
            selected,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh=refresh,
        )
        report_progress(
            82,
            f"Exhaustive CWA asset scan complete ({perf_counter() - fallback_started:.2f}s)",
        )
        return full

    generator.scan_assets = scan_assets_with_progress
    _INSTALLED = True
