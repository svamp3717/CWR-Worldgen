# SPDX-License-Identifier: GPL-3.0-or-later
"""Make runway preparation explicit in progress and avoid whole-PBO reads."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json


_INSTALLED = False


def _extract_pbo_asset_streaming(exact, path: Path, target: str) -> bytes | None:
    """Read only the selected PBO entry instead of loading the whole package."""
    target = exact._canonical(target)
    with Path(path).open("rb") as stream:
        entries: list[tuple[str, int, int, int]] = []
        properties: dict[str, str] = {}
        while True:
            name = exact._read_cstring(stream)
            fields = stream.read(exact._PBO_FIELDS.size)
            if len(fields) != exact._PBO_FIELDS.size:
                raise ValueError("truncated PBO header")
            packing, original_size, _reserved, _timestamp, data_size = (
                exact._PBO_FIELDS.unpack(fields)
            )
            if not name:
                if packing == exact._PBO_PROPERTIES:
                    while True:
                        key = exact._read_cstring(stream)
                        if not key:
                            break
                        properties[key.casefold()] = exact._read_cstring(stream)
                    continue
                break
            entries.append((name, packing, original_size, data_size))

        prefix = properties.get("prefix", "").replace("/", "\\").strip("\\")
        if not prefix:
            prefix = Path(path).stem
        data_cursor = stream.tell()

        for name, packing, original_size, data_size in entries:
            combined = name.replace("/", "\\").lstrip("\\")
            if prefix and not exact._canonical(combined).startswith(
                exact._canonical(prefix) + "\\"
            ):
                combined = prefix + "\\" + combined

            if exact._canonical(combined) != target:
                data_cursor += data_size
                continue

            stream.seek(data_cursor)
            stored = stream.read(data_size)
            if len(stored) != data_size:
                raise ValueError(f"truncated PBO entry {name}")
            if packing == 0:
                return stored
            if packing == exact._PBO_COMPRESSED and original_size > 0:
                return exact._decompress_pbo_payload(stored, original_size)
            return None

    return None


def _runway_report_detail(source_dir: Path) -> str:
    path = Path(source_dir) / "runway-textures.json"
    if not path.is_file():
        return ""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""

    cells = int(report.get("runway_cells", 0) or 0)
    hits = int(report.get("cache_hits", 0) or 0)
    misses = int(report.get("cache_misses", 0) or 0)
    generated = int(report.get("generated_runway_textures", 0) or 0)
    return (
        f"{cells} cells; cache {hits} hits/{misses} rendered; "
        f"{generated} unique PAAs"
    )


def install_runway_performance_policy() -> None:
    """Keep runway preparation fast and visible before RVW4 serialization."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import runway_exact_background_policy as exact
    from . import runway_surface_policy as runway
    from .progress import report_progress

    # Exact background loading previously called Path.read_bytes() on an entire
    # O.pbo/Data.pbo just to extract one terrain PAA. Parse the small header and
    # seek directly to the selected payload instead.
    exact._extract_pbo_asset = lambda path, target: _extract_pbo_asset_streaming(
        exact, path, target
    )
    try:
        exact._read_external_asset_cached.cache_clear()
    except AttributeError:
        pass

    original_apply = runway.apply_generated_runway_texture_table

    def apply_runway_textures_with_progress(
        source_dir, dataset, projection, spec, texture_indices, texture_paths
    ):
        touched = runway.runway_texture_cell_indices(dataset, projection, spec)
        if not touched:
            return original_apply(
                source_dir, dataset, projection, spec, texture_indices, texture_paths
            )

        report_progress(
            86,
            f"Preparing runway terrain textures ({len(touched):,} touched cells)",
        )
        started = perf_counter()
        result = original_apply(
            source_dir, dataset, projection, spec, texture_indices, texture_paths
        )
        elapsed = perf_counter() - started
        detail = _runway_report_detail(Path(source_dir))
        report_progress(
            86,
            "Runway terrain textures ready"
            + (f" ({detail}; {elapsed:.2f}s)" if detail else f" ({elapsed:.2f}s)"),
        )
        return result

    runway.apply_generated_runway_texture_table = apply_runway_textures_with_progress
    _INSTALLED = True
