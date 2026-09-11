# SPDX-License-Identifier: GPL-3.0-or-later
"""Expose fast CWA scan behavior without recursively crawling the whole install."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence


_INSTALLED = False


def _sample_unresolved(result, *, limit: int = 8) -> str:
    values = [*result.missing_models, *result.missing_dependencies]
    if not values:
        return ""
    sample = ", ".join(values[:limit])
    if len(values) > limit:
        sample += f", +{len(values) - limit} more"
    return sample


def install_asset_scan_progress_policy() -> None:
    """Wrap generator asset validation with timing/cache diagnostics.

    The targeted scanner is now authoritative for normal GUI builds. If an asset
    is not present in a standard package/loose-file layout, return that unresolved
    result immediately instead of falling back to the historical recursive full
    catalogue scan. Strict-assets mode can then fail explicitly; normal builds can
    continue without paying minutes to search the entire game directory.
    """

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
        detail = _sample_unresolved(result)
        report_progress(
            82,
            "Targeted CWA lookup unresolved "
            f"{missing_count} item(s) after {targeted_seconds:.2f}s; "
            "skipping exhaustive full-install scan"
            + (f" [{detail}]" if detail else ""),
        )
        return result

    generator.scan_assets = scan_assets_with_progress
    _INSTALLED = True

    # Keep game-install validation intentionally narrow: one stock runway
    # background texture per preset plus stock models in standard package paths.
    from .single_runway_background_policy import install_single_runway_background_policy

    install_single_runway_background_policy()

    # Reusable runway cell PAAs belong beside the other cross-world caches, not
    # in a separate top-level dot-directory left over from the earlier cache.
    from .shared_runway_cache_policy import install_shared_runway_cache_policy

    install_shared_runway_cache_policy()

    # Dense maps can contain hundreds of thousands of SingleObject4 records.
    # Install the vectorized serializer beneath the runway pre-write wrapper so
    # texture preparation still runs before the final RVW4 bytes are emitted.
    from .fast_wrp_write_policy import install_fast_wrp_write_policy

    install_fast_wrp_write_policy()

    # Exact runway backgrounds only need one PAA from a stock PBO. Read that
    # entry by header offset instead of reading the whole package, and expose
    # runway cache/render timing separately from RVW4 serialization.
    from .runway_performance_policy import install_runway_performance_policy

    install_runway_performance_policy()

    # Sports pitches use the same exact-background terrain-cell approach as
    # runways: keep the preset's ordinary grass artwork and paint only the field
    # markings in world coordinates, with persistent per-cell cache reuse.
    from .sports_pitch_surface_policy import install_sports_pitch_surface_policy

    install_sports_pitch_surface_policy()
