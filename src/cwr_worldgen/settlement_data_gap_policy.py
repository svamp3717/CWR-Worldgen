# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Callable, Iterable

from . import osm as _osm

# These deliberately mirror the existing settlement-detail radii. A place marker
# with no building centre inside this fairly generous area is a strong signal that
# the source data is missing most or all of the settlement rather than merely one
# outlying house.
SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES: dict[str, float] = {
    "town": 800.0,
    "village": 420.0,
    "hamlet": 240.0,
}
MAX_SETTLEMENT_GAP_WARNINGS = 8

_OVERTURE_AVAILABLE: ContextVar[bool] = ContextVar(
    "cwr_settlement_gap_overture_available", default=False
)
_WARNED_DATASETS: ContextVar[frozenset[str]] = ContextVar(
    "cwr_settlement_gap_warned_datasets", default=frozenset()
)
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class SettlementDataGap:
    place_type: str
    name: str
    radius_metres: float
    x: float
    z: float


def _clean_place_name(value: object, place_type: str) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(character for character in text if character.isprintable())
    text = text.replace('"', "'")
    return text or f"unnamed {place_type}"


def _building_anchor_positions(dataset, projection) -> Iterable[tuple[float, float]]:
    """Yield one cheap representative point for every source-backed building.

    At this stage the dataset contains normalized OSM buildings plus accepted
    Overture enrichment, but no later synthetic settlement buildings. Polygon
    vertex means are sufficient here because the search radii are hundreds of
    metres and avoid pulling Shapely into this lightweight warning pass.
    """

    for feature in getattr(dataset, "building_points", ()) or ():
        yield projection.to_world(feature.point)

    for feature in getattr(dataset, "building_polygons", ()) or ():
        for polygon in getattr(feature, "polygons", ()) or ():
            ring = tuple(getattr(polygon, "outer", ()) or ())
            if len(ring) < 3:
                continue
            points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            if not points:
                continue
            latitude = sum(float(point[0]) for point in points) / len(points)
            longitude = sum(float(point[1]) for point in points) / len(points)
            yield projection.to_world((latitude, longitude))


def find_settlement_building_gaps(dataset, projection) -> tuple[SettlementDataGap, ...]:
    """Return named towns/villages with no OSM or enriched building nearby."""

    bucket_size = min(SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES.values())
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for x, z in _building_anchor_positions(dataset, projection):
        if not (math.isfinite(x) and math.isfinite(z)):
            continue
        buckets[(math.floor(x / bucket_size), math.floor(z / bucket_size))].append((x, z))

    gaps: list[SettlementDataGap] = []
    world_size = float(getattr(projection, "world_size", 0.0))
    for place in sorted(getattr(dataset, "places", ()) or (), key=lambda item: item.osm_key):
        place_type = str(place.tags.get("place", "")).strip().casefold()
        radius = SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES.get(place_type)
        if radius is None:
            continue
        x, z = projection.to_world(place.point)
        if not (math.isfinite(x) and math.isfinite(z)):
            continue
        if world_size > 0.0 and not (0.0 <= x <= world_size and 0.0 <= z <= world_size):
            continue

        radius_squared = radius * radius
        bx0 = math.floor((x - radius) / bucket_size)
        bz0 = math.floor((z - radius) / bucket_size)
        bx1 = math.floor((x + radius) / bucket_size)
        bz1 = math.floor((z + radius) / bucket_size)
        found = False
        for bz in range(bz0, bz1 + 1):
            for bx in range(bx0, bx1 + 1):
                if any(
                    (x - building_x) ** 2 + (z - building_z) ** 2 <= radius_squared
                    for building_x, building_z in buckets.get((bx, bz), ())
                ):
                    found = True
                    break
            if found:
                break
        if found:
            continue

        gaps.append(SettlementDataGap(
            place_type=place_type,
            name=_clean_place_name(place.tags.get("name"), place_type),
            radius_metres=radius,
            x=x,
            z=z,
        ))
    return tuple(gaps)


def notify_missing_settlement_buildings(
    dataset,
    projection,
    *,
    overture_available: bool,
    progress_callback: Callable[[int, str], None] | None,
) -> tuple[SettlementDataGap, ...]:
    """Emit persistent progress-log warnings when both building sources are empty."""

    if not overture_available:
        return ()
    gaps = find_settlement_building_gaps(dataset, projection)
    if not gaps or progress_callback is None:
        return gaps

    progress_callback(
        0,
        f"WARNING: settlement source gap detected: {len(gaps):,} town/village/hamlet place marker(s) have no OSM or Overture buildings nearby.",
    )
    for gap in gaps[:MAX_SETTLEMENT_GAP_WARNINGS]:
        progress_callback(
            0,
            f'WARNING: {gap.place_type} "{gap.name}" has no OSM or Overture buildings within {gap.radius_metres:g} m of its place marker; the generated settlement may be missing.',
        )
    remaining = len(gaps) - MAX_SETTLEMENT_GAP_WARNINGS
    if remaining > 0:
        progress_callback(
            0,
            f"WARNING: {remaining:,} additional settlement source gap(s) omitted from the progress log.",
        )
    return gaps


def _wrap_overture_resolver(original):
    def wrapped(*args, **kwargs):
        path = original(*args, **kwargs)
        _OVERTURE_AVAILABLE.set(path is not None)
        _WARNED_DATASETS.set(frozenset())
        return path

    wrapped._cwr_settlement_data_gap_policy = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_spatial_index_prepare(original):
    def wrapped(dataset, projection, *args, **kwargs):
        fingerprint = str(getattr(dataset, "normalized_fingerprint", "")).strip()
        identity = fingerprint or f"object:{id(dataset)}"
        warned = _WARNED_DATASETS.get()
        if identity not in warned:
            notify_missing_settlement_buildings(
                dataset,
                projection,
                overture_available=_OVERTURE_AVAILABLE.get(),
                progress_callback=kwargs.get("progress_callback"),
            )
            _WARNED_DATASETS.set(warned | {identity})
        return original(dataset, projection, *args, **kwargs)

    wrapped._cwr_settlement_data_gap_policy = True  # type: ignore[attr-defined]
    return wrapped


def install_settlement_data_gap_policy() -> None:
    """Warn Milestone 9 builds about settlements absent from both building sources."""

    global _INSTALLED
    if _INSTALLED:
        return

    # Milestone 9 binds these helpers directly from their source modules, so the
    # policy is installed after milestone9 itself has loaded and wraps that module's
    # live globals. This also means cached Overture enrichment is covered: the
    # warning runs immediately before the spatial index regardless of cache hits.
    from . import milestone9 as _milestone9

    resolver = _milestone9._resolve_overture_buildings_geojson
    if not getattr(resolver, "_cwr_settlement_data_gap_policy", False):
        _milestone9._resolve_overture_buildings_geojson = _wrap_overture_resolver(resolver)

    prepare = _milestone9.prepare_spatial_index
    if not getattr(prepare, "_cwr_settlement_data_gap_policy", False):
        _milestone9.prepare_spatial_index = _wrap_spatial_index_prepare(prepare)

    _INSTALLED = True
