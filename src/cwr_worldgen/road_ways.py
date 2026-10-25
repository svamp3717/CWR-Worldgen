# SPDX-License-Identifier: GPL-3.0-or-later
"""CWR-CE native, model-free OSM-style road way export.

The engine's feature/osm-style-road-ways reader expects local X/Z coordinates,
shared integer node IDs, and a surface *texture*, not a P3D model name.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from .osm import (
    BboxProjection,
    OsmDataset,
    PEDESTRIAN_ONLY_HIGHWAYS,
    road_is_dirt,
    road_is_gravel,
    road_is_supported,
    road_width_metres,
)

# Hard limits in CWR-CE RoadNet::ReadWayDefinitions / LoadWays.
_MAX_NODES = 100_000
_MAX_WAYS = 10_000
_MAX_BYTES = 8 * 1024 * 1024
_MAX_LINE = 8192
_MAX_NODES_PER_WAY = 512  # comfortably below the engine's 4096-node limit
_COORD_DECIMALS = 4


def is_osm_ways_profile(profile: str) -> bool:
    return profile == "cwr-ce-ways"


def _surface(tags: Mapping[str, str]) -> str:
    if tags.get("highway", "").casefold() in PEDESTRIAN_ONLY_HIGHWAYS:
        return "dirt"
    if road_is_gravel(tags):
        return "gravel"
    return "dirt" if road_is_dirt(tags) else "paved"


def _width(tags: Mapping[str, str]) -> float:
    # The normalized layer exposes a width in metres as "width". Original OSM
    # input may provide width=5.5, but free-form strings are not reliably units.
    for key in ("width", "width_m"):
        try:
            value = float(tags.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0.5 <= value <= 50.0:
            return value
    return min(50.0, max(0.5, road_width_metres(tags)))


def _clip(
    a: tuple[float, float], b: tuple[float, float], limit: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clipping, keeping coordinates strictly inside the WRP."""
    x0, z0 = a
    dx, dz = b[0] - x0, b[1] - z0
    if not all(map(math.isfinite, (x0, z0, dx, dz))):
        return None
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, limit - x0),
                 (-dz, z0), (dz, limit - z0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                lower = max(lower, t)
            else:
                upper = min(upper, t)
            if lower > upper:
                return None
    start = (round(max(0.0, min(limit, x0 + lower * dx)), _COORD_DECIMALS),
             round(max(0.0, min(limit, z0 + lower * dz)), _COORD_DECIMALS))
    end = (round(max(0.0, min(limit, x0 + upper * dx)), _COORD_DECIMALS),
           round(max(0.0, min(limit, z0 + upper * dz)), _COORD_DECIMALS))
    if math.dist(start, end) < 0.001:
        return None
    return start, end


def _runs(
    points: Sequence[tuple[float, float]], limit: float,
) -> list[list[tuple[float, float]]]:
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for a, b in zip(points, points[1:]):
        clipped = _clip(a, b, limit)
        if clipped is None:
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        start, end = clipped
        if current and current[-1] != start:
            if len(current) >= 2:
                runs.append(current)
            current = []
        if not current:
            current.append(start)
        if current[-1] != end:
            current.append(end)
    if len(current) >= 2:
        runs.append(current)
    return runs


def write_osm_road_ways(
    path: Path,
    dataset: OsmDataset,
    projection: BboxProjection,
    *,
    include_minor_roads: bool,
    surface_textures: Mapping[str, str],
) -> dict[str, int | str]:
    """Export terrain-following non-bridge roads as a deterministic sidecar.

    Point identity comes from the normalized projected coordinates, so roads
    snapped to the same at-grade junction reuse a single node ID. Distinct
    coordinates (including visually crossing lines) stay disconnected.
    """
    textures: dict[str, str] = {}
    for surface in ("paved", "dirt", "gravel"):
        texture = str(surface_textures.get(surface) or "").replace("/", "\\")
        if (not texture or any(char.isspace() for char in texture)
                or texture.startswith("\\") or texture.endswith("\\")
                or not texture.casefold().endswith((".paa", ".pac"))):
            raise ValueError(
                f"OSM ways require a valid single-token Roadway texture for {surface}: {texture!r}"
            )
        textures[surface] = texture

    limit = projection.world_size - 0.01
    if limit <= 0:
        raise ValueError("OSM ways require positive terrain dimensions")
    nodes: dict[tuple[float, float], int] = {}
    lines: list[str] = []
    skipped_elevated = 0
    for road in sorted(dataset.roads, key=lambda item: item.osm_key):
        tags = road.tags
        if not road_is_supported(tags, include_minor=include_minor_roads):
            continue
        # Engine ribbons follow the ground. Preserve separately modelled
        # bridge decks and never paint ground-level roads through them.
        if tags.get("bridge", "").strip().casefold() not in {"", "no", "false", "0"}:
            skipped_elevated += 1
            continue
        points = tuple(projection.to_world(point) for point in road.points)
        for run in _runs(points, limit):
            for index in range(0, len(run) - 1, _MAX_NODES_PER_WAY - 1):
                chunk = run[index:index + _MAX_NODES_PER_WAY]
                if len(chunk) < 2:
                    continue
                ids: list[int] = []
                for point in chunk:
                    if point not in nodes:
                        nodes[point] = len(nodes) + 1
                        if len(nodes) > _MAX_NODES:
                            raise ValueError(
                                "CWR-CE .ways supports at most 100,000 nodes; "
                                "reduce the generated area or road detail"
                            )
                    ids.append(nodes[point])
                if len(lines) >= _MAX_WAYS:
                    raise ValueError(
                        "CWR-CE .ways supports at most 10,000 ways; "
                        "reduce the generated area or road detail"
                    )
                lines.append(
                    f"way {len(lines) + 1} {_width(tags):.3f} {textures[_surface(tags)]} "
                    + " ".join(map(str, ids))
                )
    # Emit nodes in ID order. Both record types may occur in either order.
    node_lines = [
        f"node {node_id} {x:.4f} {z:.4f}"
        for (x, z), node_id in nodes.items()
    ]
    content = (
        "# CWR-CE native OSM-style roads; local terrain X/Z in metres\n"
        + "\n".join(node_lines + lines) + "\n"
    ).encode("utf-8")
    if len(content) > _MAX_BYTES or any(
        len(line.encode("utf-8")) > _MAX_LINE for line in node_lines + lines
    ):
        raise ValueError(
            "CWR-CE .ways size/line limit exceeded; reduce road detail or map area"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.name,
        "nodes": len(nodes),
        "ways": len(lines),
        "excluded_bridge_features": skipped_elevated,
        "format": "cwr-ce-osm-ways-v1",
    }
