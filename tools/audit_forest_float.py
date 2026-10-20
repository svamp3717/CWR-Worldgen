#!/usr/bin/env python3
"""Report trees/forest objects whose *serialized origins* disagree with terrain.

Both 4WVR and OPRW are supported. The diagnostic compares game-scaled
terrain to the WRP object origin, which can expose misplaced roots but is not
a renderer emulator. CWR-CE may recenter or terrain-deform tree model geometry
after loading, and stock forest blocks have their own model-specific pivots.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import re
import struct
import tempfile

from cwr_worldgen.osm import (
    _oriented_footprint_elevation_samples,
    _square_elevation_samples,
    _triangle_elevation_bounds,
)
from cwr_worldgen.pbo import is_pbo_path, read_pbo
from cwr_worldgen.model import CWA_RVW4_HEIGHT_SCALE
from cwr_island_studio.oprw import open_island_document

_RVW4_HEADER = struct.Struct("<4sii")
_RVW4_OBJECT = struct.Struct("<12fi76s")
_HEIGHT_SCALE_METRES = CWA_RVW4_HEIGHT_SCALE


def _classify_tree_model(model: str) -> str | None:
    folded = model.replace("/", "\\").casefold()
    if not folded.endswith(".p3d"):
        return None
    if "les" in folded and ("ctver" in folded or "les_nw_" in folded):
        return "forest_square"
    if "les" in folded and "trojuhelnik" in folded:
        return "forest_triangle"
    if folded.startswith("o\\tree\\"):
        excluded = ("dd_bush", "dd_rakosi", "les_", "ker", "krovi")
        return None if any(token in folded for token in excluded) else "tree"
    if folded.startswith(r"data3d\str"):
        return "tree"
    return None


def _read_source(path: Path, cell_size_override: float | None) -> tuple[bytes, float]:
    if not is_pbo_path(path):
        return path.read_bytes(), float(cell_size_override or 25.0)

    entries = {entry.name: entry.data for entry in read_pbo(path)}
    wrp_names = [name for name in entries if name.casefold().endswith(".wrp")]
    if len(wrp_names) != 1:
        raise ValueError(f"expected exactly one WRP in {path}, found {len(wrp_names)}")
    cell_size = cell_size_override
    config = entries.get("config.cpp")
    if cell_size is None and config is not None:
        match = re.search(rb"\blandGrid\s*=\s*([0-9.]+)", config)
        if match:
            cell_size = float(match.group(1))
    return entries[wrp_names[0]], float(cell_size or 25.0)


def _parse_wrp(data: bytes) -> tuple[int, tuple[float, ...], list[tuple[int, str, float, float, float, float]]]:
    if data[:4] == b"OPRW":
        # Island Studio's OPRW reader already decodes v2/v3 compression and
        # float height samples in *metres*. No height multiplier applies.
        with tempfile.TemporaryDirectory(prefix="cwr-forest-audit-") as folder:
            path = Path(folder) / "world.wrp"
            path.write_bytes(data)
            with open_island_document(path, cwa_runtime_scale=True) as document:
                if document.width != document.height:
                    raise ValueError("expected a square OPRW world")
                heights = tuple(
                    document.terrain_at(x, z)
                    for z in range(document.height) for x in range(document.width)
                )
                placements = []
                for i in range(document.object_count):
                    obj = document.object_at(i)
                    placements.append(
                        (obj.object_id, obj.model_path, obj.x, obj.y, obj.z, obj.heading)
                    )
                return document.width, heights, placements
    stream = io.BytesIO(data)
    raw_header = stream.read(_RVW4_HEADER.size)
    if len(raw_header) != _RVW4_HEADER.size:
        raise ValueError("truncated RVW4 header")
    magic, width, height = _RVW4_HEADER.unpack(raw_header)
    if magic != b"4WVR" or width != height:
        raise ValueError("expected a square 4WVR world")
    cells = width * height
    raw_heights = stream.read(cells * 2)
    if len(raw_heights) != cells * 2:
        raise ValueError("truncated RVW4 height grid")
    elevations = tuple(
        value * _HEIGHT_SCALE_METRES
        for value in struct.unpack(f"<{cells}h", raw_heights)
    )
    stream.seek(cells * 2 + 512 * 32, io.SEEK_CUR)
    objects: list[tuple[int, str, float, float, float, float]] = []
    while stream.tell() + _RVW4_OBJECT.size <= len(data):
        values = _RVW4_OBJECT.unpack(stream.read(_RVW4_OBJECT.size))
        raw_model = values[13].split(b"\0", 1)[0]
        if not raw_model:
            break
        heading = math.degrees(math.atan2(-values[2], values[0])) % 360.0
        objects.append(
            (
                int(values[12]),
                raw_model.decode("ascii"),
                float(values[9]),
                float(values[10]),
                float(values[11]),
                heading,
            )
        )
    return width, elevations, objects


def audit(
    source: Path,
    *,
    cell_size: float | None = None,
    square_footprint: float = 50.0,
    triangle_footprint: float = 35.0,
) -> list[dict[str, object]]:
    data, resolved_cell_size = _read_source(source, cell_size)
    cells, elevations, objects = _parse_wrp(data)
    rows: list[dict[str, object]] = []
    for object_id, model, x, y, z, heading in objects:
        category = _classify_tree_model(model)
        if category is None:
            continue
        if category == "forest_square":
            supports = _square_elevation_samples(
                elevations, cells, resolved_cell_size, x, z, square_footprint
            )
            terrain_low, terrain_high = min(supports), max(supports)
        elif category == "forest_triangle":
            supports = _oriented_footprint_elevation_samples(
                elevations,
                cells,
                resolved_cell_size,
                x,
                z,
                triangle_footprint * 0.58,
                triangle_footprint,
                heading,
            )
            terrain_low, terrain_high = min(supports), max(supports)
        else:
            terrain_low, terrain_high = _triangle_elevation_bounds(
                elevations, cells, resolved_cell_size, x, z
            )
        rows.append(
            {
                "object_id": object_id,
                "category": category,
                "model": model,
                "x": x,
                "z": z,
                "object_y": y,
                "terrain_low": terrain_low,
                "terrain_high": terrain_high,
                "estimated_float": max(0.0, y - terrain_low),
                "estimated_burial": max(0.0, terrain_high - y),
                "heading": heading,
            }
        )
    rows.sort(key=lambda row: (-float(row["estimated_burial"]), int(row["object_id"])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="generated world PBO or loose WRP")
    parser.add_argument("--csv", type=Path, help="output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="vertical origin gap to flag in either direction (default: 0.5 m)")
    parser.add_argument("--cell-size", type=float, help="terrain cell size for loose WRP or override")
    parser.add_argument("--square-footprint", type=float, default=50.0)
    parser.add_argument("--triangle-footprint", type=float, default=35.0)
    args = parser.parse_args()

    rows = audit(
        args.source,
        cell_size=args.cell_size,
        square_footprint=args.square_footprint,
        triangle_footprint=args.triangle_footprint,
    )
    buried = [
        row for row in rows
        if float(row["estimated_burial"]) > args.threshold + 1.0e-4
    ]
    floating = [
        row for row in rows
        if float(row["estimated_float"]) > args.threshold + 1.0e-4
    ]
    output = args.csv or args.source.with_name(args.source.stem + "_forest_float_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "object_id", "category", "model", "x", "z", "object_y",
        "terrain_low", "terrain_high", "estimated_float", "estimated_burial", "heading",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    maximum_burial = max((float(row["estimated_burial"]) for row in rows), default=0.0)
    maximum_float = max((float(row["estimated_float"]) for row in rows), default=0.0)
    print(f"tree/forest objects audited: {len(rows)}")
    print(f"maximum estimated origin burial: {maximum_burial:.3f} m")
    print(f"maximum estimated origin float: {maximum_float:.3f} m")
    print(f"origins buried > {args.threshold:.3f} m: {len(buried)}")
    print(f"origins floating > {args.threshold:.3f} m: {len(floating)}")
    print(f"CSV: {output}")
    if rows:
        print("most buried origins (not actual model root depth):")
        for row in rows[:10]:
            print(
                f"  id={row['object_id']} origin-burial={float(row['estimated_burial']):.3f}m "
                f"x={float(row['x']):.1f} z={float(row['z']):.1f} {row['model']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
