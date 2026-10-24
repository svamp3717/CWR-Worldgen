# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare real stock and modded paved P3D donors without altering road builds.

SFP paved models have been observed to cover mixed gravel junctions correctly
where original CWA paved models do not. Geometry, render flags and donor
metadata must be measured before prescribing a height or overlap workaround.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Sequence

from .assets import canonical_asset_path, read_asset_record_bytes, scan_assets
from .legacy_proxy_models import (
    ProxyCloneError,
    _read_mlod_visual,
    _read_odol_visual,
    inspect_roadway_surface_style,
    inspect_visual_model_dimensions,
    inspect_visual_surface_style,
)

STOCK_PAVED = r"o\road\sil25.p3d"
STOCK_CURVE = r"o\road\sil10 25.p3d"
SFP_STRAIGHT = r"sfp_objects\roads\sil25.p3d"
SFP_CURVE = r"sfp_objects\roads\sil10 25.p3d"
DEFAULT_ROADS = (STOCK_PAVED, SFP_STRAIGHT, STOCK_CURVE, SFP_CURVE)
_LAND_FLAG_MASK = 0x0900


def _hex_counts(values: Sequence[int]) -> dict[str, int]:
    return {
        f"0x{int(flag) & 0xffffffff:08x}": count
        for flag, count in sorted(Counter(values).items())
    }


def summarize_road_model(data: bytes) -> dict[str, object]:
    """Summarize actual drawable model faces, not merely the model filename."""
    if data.startswith(b"ODOL"):
        lod, land, textures = _read_odol_visual(
            data, strip_land_deform=False,
        )
        source_format = "ODOL"
    elif data.startswith(b"MLOD"):
        lod, land, textures = _read_mlod_visual(
            data, strip_land_deform=False,
        )
        source_format = "MLOD"
    else:
        raise ProxyCloneError("road model is neither ODOL nor MLOD")

    if not lod.points or not lod.faces:
        raise ProxyCloneError("road model has no drawable visual points/faces")

    visual_y = [float(lod.points[int(vertex[0])][1])
                for face in lod.faces for vertex in face.vertices
                if 0 <= int(vertex[0]) < len(lod.points)]
    windings: list[int] = []
    for face in lod.faces:
        if len(face.vertices) < 3:
            continue
        indices = [int(v[0]) for v in face.vertices[:3]]
        if not all(0 <= i < len(lod.points) for i in indices):
            continue
        a, b, c = (lod.points[i] for i in indices)
        # Project each face into the road's X/Z plane. Opposite signs are
        # different face windings, which can affect native CWA road rendering.
        signed = ((b[2] - a[2]) * (c[0] - a[0])
                  - (b[0] - a[0]) * (c[2] - a[2]))
        if abs(signed) > 1e-9:
            windings.append(1 if signed > 0 else -1)

    point_flags = tuple(int(flag) & 0xffffffff for flag in lod.point_flags)
    face_flags = tuple(int(face.flags) & 0xffffffff for face in lod.faces)
    summary: dict[str, object] = {
        "format": source_format,
        "visual_points": len(lod.points),
        "visual_faces": len(lod.faces),
        "visual_textures": sorted(textures),
        "visual_point_flags": _hex_counts(point_flags),
        "visual_face_flags": _hex_counts(face_flags),
        "visual_projected_winding": {
            "positive": sum(v > 0 for v in windings),
            "negative": sum(v < 0 for v in windings),
        },
        "visual_land_flagged_points": land,
        "visual_height_min_metres": round(min(visual_y), 5),
        "visual_height_max_metres": round(max(visual_y), 5),
        "visual_normal_y": {
            "positive": sum(float(n[1]) > 0.01 for n in lod.normals),
            "negative": sum(float(n[1]) < -0.01 for n in lod.normals),
        },
        "visual_texture_face_counts": dict(sorted(Counter(
            face.texture.replace("/", "\\").casefold()
            for face in lod.faces
        ).items())),
    }
    try:
        dims = inspect_visual_model_dimensions(data)
        summary["bounds"] = {
            "width_metres": round(dims.width_metres, 5),
            "length_metres": round(dims.length_metres, 5),
            "height_metres": round(dims.height_metres, 5),
            "connector_width_metres": (
                None if dims.connector_width_metres is None
                else round(dims.connector_width_metres, 5)
            ),
        }
    except ProxyCloneError as exc:
        summary["bounds_error"] = str(exc)
    try:
        style = inspect_visual_surface_style(data)
        summary["dominant_visual_style"] = {
            "texture": style.texture_path,
            "point_flag": f"0x{style.point_flag & 0xffffffff:08x}",
            "face_flag": f"0x{style.face_flag & 0xffffffff:08x}",
        }
    except ProxyCloneError as exc:
        summary["dominant_visual_error"] = str(exc)
    try:
        roadway = inspect_roadway_surface_style(data)
        summary["roadway"] = {
            "texture": roadway.texture_path,
            "point_flag": f"0x{roadway.point_flag & 0xffffffff:08x}",
            "face_flag": f"0x{roadway.face_flag & 0xffffffff:08x}",
        }
    except ProxyCloneError as exc:
        summary["roadway_error"] = str(exc)
    return summary


def compare_paved_models(
    roots: Sequence[Path],
    models: Sequence[str] = DEFAULT_ROADS,
) -> dict[str, object]:
    """Resolve stock/SFP P3Ds in loose folders or PBOs and report differences.

    No bundled proprietary mod assets are required. Missing donors are reported
    explicitly; callers must not infer that two models are geometrically equal.
    """
    normalized = tuple(canonical_asset_path(value) for value in models)
    if len(normalized) < 2 or len(normalized) % 2:
        raise ValueError("models must be (baseline, reference) pairs")
    if len(set(normalized)) != len(normalized):
        raise ValueError("models must resolve to distinct paths")
    scan = scan_assets(
        tuple(Path(value) for value in roots), normalized, use_cache=False,
    )
    # Downloaded PBOs are commonly named "SFP_objects(2).pbo". The archive
    # does not declare a prefix, so the PBO reader uses its *filename* as the
    # namespace. Permit that unambiguous duplicate-download suffix without
    # silently treating a different mod as the trusted SFP reference.
    selected = {}
    aliases = {}
    for record in scan.records:
        if record.path in normalized:
            selected.setdefault(record.path, record)
            continue
        prefix, separator, rest = record.path.partition("\\")
        match = re.fullmatch(r"(.+?)\\s*\\(\\d+\\)", prefix)
        if not separator or match is None:
            continue
        canonical = match.group(1) + separator + rest
        if canonical in normalized:
            aliases.setdefault(canonical, []).append(record)

    summaries = {}
    for model in normalized:
        record = selected.get(model)
        if record is None:
            matches = aliases.get(model, ())
            if len(matches) == 1:
                record = matches[0]
            elif len(matches) > 1:
                summaries[model] = {
                    "status": "ambiguous",
                    "sources": [value.source for value in matches],
                }
                continue
        if record is None:
            summaries[model] = {"status": "missing"}
            continue
        try:
            data = read_asset_record_bytes(record)
            summary = summarize_road_model(data)
            summaries[model] = {
                "status": "ok",
                "source": record.source,
                "sha256": record.sha256,
                **summary,
            }
        except (OSError, ValueError, ProxyCloneError) as exc:
            summaries[model] = {
                "status": "unreadable", "source": record.source,
                "error": str(exc),
            }

    available = [path for path in normalized
                 if summaries[path]["status"] == "ok"]
    comparisons: dict[str, dict[str, object]] = {}
    for index in range(0, len(normalized), 2):
        baseline, model = normalized[index:index + 2]
        if baseline not in available or model not in available:
            continue
        candidate = summaries[model]
        reference = summaries[baseline]
        differences = {}
        fields = (
            "format", "visual_points", "visual_faces",
            "visual_point_flags", "visual_face_flags",
            "visual_projected_winding", "visual_land_flagged_points",
            "visual_height_min_metres", "visual_height_max_metres",
            "visual_normal_y", "bounds", "roadway",
        )
        for field in fields:
            if candidate.get(field) != reference.get(field):
                differences[field] = {
                    "baseline": reference.get(field),
                    "candidate": candidate.get(field),
                }
        comparisons[model] = {
            "compared_to": baseline,
            "differences": differences,
        }
    return {
        "schema": 1,
        "models": summaries,
        "comparisons": comparisons,
        "all_models_readable": len(available) == len(normalized),
        "unreadable_pbos": list(scan.unreadable_pbos),
    }
