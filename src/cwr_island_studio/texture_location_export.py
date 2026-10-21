"""Lossless, north-up texture-location atlas for 4WVR and OPRW Island Studio maps.

Exports exact texture-layer cells, including pending 4WVR paint edits. This
does not access or require ground texture artwork, PBOs, or editable worlds.
"""
from __future__ import annotations

from collections import Counter
import base64
import colorsys
import csv
from html import escape
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .core import IslandDocument


def _colour(slot: int) -> tuple[int, int, int]:
    # Each ID maps to a stable, visually distinct colour. This is a categorical
    # diagnostic map, not a rendering of the actual PAC/PAA texture artwork.
    hue = (slot * 0.6180339887498949) % 1.0
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, 0.72, 0.90))


def _path_for(paths: tuple[str, ...], slot: int) -> str:
    return paths[slot] if 0 <= slot < len(paths) else ""


def _csv_member(archive: zipfile.ZipFile, name: str, fields: tuple[str, ...]):
    """Yield a csv.writer with a compressed output stream (no giant CSV buffer)."""
    from contextlib import contextmanager

    @contextmanager
    def writer():
        with archive.open(name, "w", force_zip64=True) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                output = csv.writer(text)
                output.writerow(fields)
                yield output

    return writer()


def _encoded_png(pixels: np.ndarray) -> bytes:
    result = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(result, format="PNG", optimize=True)
    return result.getvalue()


def _html_viewer(
    title: str, preview_png: bytes, id_png: bytes, texture_paths: dict[int, str],
    width: int, height: int, step_x: float, step_z: float,
) -> str:
    preview = base64.b64encode(preview_png).decode("ascii")
    ids = base64.b64encode(id_png).decode("ascii")
    # WRP texture strings are untrusted. Escape characters that might break
    # out of the inline JSON script element.
    paths_json = json.dumps(
        {str(k): v for k, v in texture_paths.items()}, ensure_ascii=True,
    ).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Island Studio texture atlas: {escape(title)}</title>
<style>
body {{font:15px system-ui,sans-serif;background:#1b2224;color:#f3f3f3;margin:16px}}
#map {{display:block;max-width:100%;image-rendering:pixelated;cursor:crosshair;border:1px solid #65736a}}
#info {{margin:12px 0;padding:8px;background:#303b3d;white-space:pre-wrap}}
small {{color:#b8c7c4}}
</style></head><body>
<h2>Texture locations: {escape(title)}</h2>
<p><small>North is up; east is right. Each coloured pixel is exactly one texture-layer cell.
Hover or click for the in-game world position and original WRP texture path.
No internet connection required.</small></p>
<div id="info" role="status">Move the pointer over the map.</div>
<img id="map" width="{width}" height="{height}" alt="Colour-coded map of WRP ground texture slots"
src="data:image/png;base64,{preview}">
<script>
"use strict";
const paths = {paths_json};
const width = {width}, height = {height};
const stepX = {step_x!r}, stepZ = {step_z!r};
const ids = new Image();
ids.src = "data:image/png;base64,{ids}";
const canvas = document.createElement("canvas");
canvas.width = width; canvas.height = height;
const ctx = canvas.getContext("2d", {{willReadFrequently:true}});
ids.onload = () => ctx.drawImage(ids,0,0);
const map = document.getElementById("map");
const info = document.getElementById("info");
function inspect(event) {{
  if (!ids.complete || !ids.naturalWidth) return;
  const rect = map.getBoundingClientRect();
  const x = Math.min(width-1, Math.max(0, Math.floor((event.clientX-rect.left)*width/rect.width)));
  const northRow = Math.min(height-1, Math.max(0, Math.floor((event.clientY-rect.top)*height/rect.height)));
  const rgba = ctx.getImageData(x,northRow,1,1).data;
  let slot = (rgba[0]<<16)|(rgba[1]<<8)|rgba[2];
  if (slot >= 8388608) slot -= 16777216;
  const z = height-1-northRow;
  const path = paths[String(slot)] || "(blank, reserved or invalid texture slot)";
  info.textContent = "Texture slot: " + slot + "\\nPath: " + path
    + "\\nTexture cell (X, Z): " + x + ", " + z
    + "\\nWorld centre (X, Z metres): "
    + ((x+0.5)*stepX).toFixed(2) + ", " + ((z+0.5)*stepZ).toFixed(2);
}}
map.addEventListener("mousemove", inspect);
map.addEventListener("click", inspect);
</script></body></html>
"""


def export_texture_location_atlas(
    doc: IslandDocument,
    destination: Path | str,
    *,
    progress=None,
) -> Path:
    """Write one portable .zip atlas with every texture-layer cell accounted for.

    The north-up RGB texture-ids.png encodes signed 24-bit slot IDs exactly:
    (red << 16) + (green << 8) + blue, subtracting 2**24 for values >= 2**23.
    Texture-runs.csv covers every cell with inclusive X ranges, avoiding a
    potentially multi-gigabyte individual-cell CSV for very large maps.
    """
    destination = Path(destination)
    if destination.suffix.casefold() != ".zip":
        raise ValueError("Texture atlas destination must be a .zip file")
    width = int(doc.textures.shape[1])
    height = int(doc.textures.shape[0])
    if width <= 0 or height <= 0 or width * height > 20_000_000:
        raise ValueError("Texture atlas is limited to 20 million texture-layer cells")
    if (doc.textures.shape != (height, width)
            or not np.issubdtype(doc.textures.dtype, np.integer)):
        raise ValueError("Invalid texture-layer array")
    step_x = float(doc.world_width) / width
    step_z = float(doc.world_height) / height
    paths = tuple(doc.texture_paths)
    overlays: dict[int, dict[int, int]] = {}
    for position, slot in getattr(doc, "texture_edits", {}).items():
        row, col = divmod(position, width)
        if 0 <= row < height and 0 <= col < width:
            overlays.setdefault(row, {})[col] = int(slot)

    colours = np.empty((height, width, 3), dtype=np.uint8)
    encoded = np.empty_like(colours)
    counts: Counter[int] = Counter()
    neighbours: Counter[tuple[int, int]] = Counter()
    extents: dict[int, list[int]] = {}
    preceding: np.ndarray | None = None

    def update_pairs(a: np.ndarray, b: np.ndarray) -> None:
        different = a != b
        if not np.any(different):
            return
        pair = np.stack((np.minimum(a[different], b[different]),
                         np.maximum(a[different], b[different])), axis=1)
        keys, amounts = np.unique(pair, axis=0, return_counts=True)
        neighbours.update({
            (int(k[0]), int(k[1])): int(n)
            for k, n in zip(keys, amounts)
        })

    if progress is not None:
        progress("Exporting exact north-up terrain texture locations…")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: failed exports do not leave a partial user-visible ZIP.
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            with _csv_member(archive, "texture-runs.csv", (
                "north_row", "game_z_row", "x_first_cell", "x_last_cell_inclusive",
                "x_min_metres", "x_max_metres", "z_min_metres", "z_max_metres",
                "texture_slot", "texture_path",
            )) as runs:
                for north_row in range(height):
                    game_z_row = height - 1 - north_row
                    row = np.asarray(doc.textures[game_z_row], dtype=np.int32).copy()
                    for col, slot in overlays.get(game_z_row, {}).items():
                        row[col] = slot
                    unique, number = np.unique(row, return_counts=True)
                    counts.update({int(k): int(n) for k, n in zip(unique, number)})
                    raw = row.astype(np.int64) & 0xFFFFFF
                    encoded[north_row, :, 0] = (raw >> 16).astype(np.uint8)
                    encoded[north_row, :, 1] = (raw >> 8).astype(np.uint8)
                    encoded[north_row, :, 2] = raw.astype(np.uint8)
                    for slot in unique:
                        colours[north_row, row == slot] = _colour(int(slot))
                    update_pairs(row[:-1], row[1:])
                    if preceding is not None:
                        update_pairs(row, preceding)
                    preceding = row
                    run_starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
                    run_ends = np.r_[run_starts[1:], width]
                    for start, end in zip(run_starts, run_ends):
                        start, end = int(start), int(end)
                        slot = int(row[start])
                        runs.writerow((
                            north_row, game_z_row, start, end - 1,
                            round(start * step_x, 6), round(end * step_x, 6),
                            round(game_z_row * step_z, 6),
                            round((game_z_row + 1) * step_z, 6),
                            slot, _path_for(paths, slot),
                        ))
                        bounds = extents.setdefault(
                            slot, [start, end - 1, game_z_row, game_z_row],
                        )
                        bounds[0] = min(bounds[0], start)
                        bounds[1] = max(bounds[1], end - 1)
                        bounds[2] = min(bounds[2], game_z_row)
                        bounds[3] = max(bounds[3], game_z_row)
                    if progress is not None and north_row and north_row % 256 == 0:
                        progress(f"Scanned {north_row:,} of {height:,} texture rows…")

            preview_png = _encoded_png(colours)
            id_png = _encoded_png(encoded)
            archive.writestr("texture-map.png", preview_png)
            archive.writestr("texture-ids.png", id_png)
            all_slots = sorted(set(range(len(paths))) | set(counts))
            with _csv_member(archive, "texture-paths.csv", (
                "texture_slot", "texture_path", "cell_count",
                "x_first_cell", "x_last_cell", "z_south_cell", "z_north_cell",
            )) as legend:
                for slot in all_slots:
                    bounds = extents.get(slot)
                    legend.writerow((
                        slot, _path_for(paths, slot), counts[slot],
                        *(bounds if bounds is not None else ("", "", "", "")),
                    ))
            with _csv_member(archive, "texture-adjacencies.csv", (
                "texture_slot_a", "texture_path_a", "texture_slot_b",
                "texture_path_b", "shared_cell_edges",
            )) as edges:
                for (a, b), number in sorted(
                    neighbours.items(), key=lambda pair: (-pair[1], pair[0]),
                ):
                    edges.writerow((a, _path_for(paths, a), b, _path_for(paths, b), number))
            meta = {
                "schema": "island-studio-texture-atlas-v1",
                "source": doc.source_label,
                "texture_grid_width": width,
                "texture_grid_height": height,
                "world_width_metres": float(doc.world_width),
                "world_height_metres": float(doc.world_height),
                "texture_cell_width_metres": step_x,
                "texture_cell_height_metres": step_z,
                "orientation": "north-up: pixel (0,0) is NW, game X east, game Z north",
                "texture_paths": {str(slot): path for slot, path in enumerate(paths)},
                "used_texture_slots": len(counts),
                "covered_cells": sum(counts.values()),
                "sparse_pending_texture_edits_included": bool(overlays),
            }
            archive.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))
            archive.writestr(
                "texture-map.html",
                _html_viewer(doc.source_label, preview_png, id_png,
                             {slot: _path_for(paths, slot) for slot in all_slots},
                             width, height, step_x, step_z),
            )
            archive.writestr("README.txt", (
                "Island Studio ground texture atlas, north up; east right.\n"
                "Open texture-map.html in a browser and hover/click cells to read original paths and game X/Z.\n"
                "texture-map.png: colour-coded image with one pixel per texture-layer cell.\n"
                "texture-ids.png: EXACT texture slot ID at every cell (signed 24-bit R/G/B).\n"
                "texture-paths.csv: complete WRP texture table and counts; zero means unused.\n"
                "texture-runs.csv: exact locations; adjacent same-slot X cells are grouped into inclusive runs.\n"
                "texture-adjacencies.csv: number of shared cell edges between each distinct pair of textures.\n"
                "metadata.json: grid dimensions, map scale, orientation, paths, and export schema.\n"
                "Game coordinates: X=(x_cell+0.5)*texture_cell_width_metres,\n"
                " Z=(game_z_row+0.5)*texture_cell_height_metres.\n"
                "Pixel Y=0 is the northernmost game-Z row; game Z increases toward north.\n"
                "If a 4WVR world has unsaved paint edits, this atlas includes them without modifying its source.\n"
            ))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if progress is not None:
        progress(f"Mapped {width:,} × {height:,} terrain texture cells.")
    return destination
