# SPDX-License-Identifier: GPL-3.0-or-later
"""Public Overpass endpoint selection and regional mirror routing.

The default list follows the public instances documented by the OpenStreetMap
wiki. Regional instances are only selected when the complete requested bbox is
inside a conservative coverage envelope; custom endpoint lists remain exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


GLOBAL_OVERPASS_URLS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


@dataclass(frozen=True, slots=True)
class RegionalOverpassMirror:
    name: str
    url: str
    coverage_bbox: tuple[float, float, float, float]


# Conservative envelopes avoid sending an out-of-coverage sliver to a regional
# service. Coordinates are south, west, north, east.
REGIONAL_OVERPASS_MIRRORS: tuple[RegionalOverpassMirror, ...] = (
    RegionalOverpassMirror(
        "Switzerland",
        "https://overpass.osm.ch/api/interpreter",
        (45.80, 5.90, 47.85, 10.55),
    ),
    RegionalOverpassMirror(
        "Britain and Ireland",
        "https://overpass.atownsend.org.uk/api/",
        (49.80, -10.80, 60.90, 1.80),
    ),
    RegionalOverpassMirror(
        "Virginia",
        "https://overpass.maprva.org/api/interpreter",
        (36.55, -83.70, 39.50, -75.15),
    ),
    RegionalOverpassMirror(
        "Ethiopia",
        "https://ethiopia.overpass.openplaceguide.org/api/interpreter",
        (3.35, 32.95, 14.90, 47.95),
    ),
)


def _normalise_url(url: str) -> str:
    return url.strip()


def _bbox_within(
    bbox: tuple[float, float, float, float],
    coverage: tuple[float, float, float, float],
) -> bool:
    south, west, north, east = bbox
    cover_south, cover_west, cover_north, cover_east = coverage
    return (
        south >= cover_south
        and west >= cover_west
        and north <= cover_north
        and east <= cover_east
    )


def overpass_urls_for_bbox(
    configured_urls: Sequence[str],
    bbox: tuple[float, float, float, float],
) -> tuple[str, ...]:
    """Return deduplicated endpoints appropriate for *bbox*.

    The built-in defaults gain applicable regional mirrors before the global
    fallback list. An explicitly supplied list is preserved exactly, apart from
    trimming whitespace and removing duplicates.
    """

    configured = tuple(_normalise_url(url) for url in configured_urls if url.strip())
    use_public_routing = configured == GLOBAL_OVERPASS_URLS
    candidates: list[str] = []
    if use_public_routing:
        candidates.extend(
            mirror.url
            for mirror in REGIONAL_OVERPASS_MIRRORS
            if _bbox_within(bbox, mirror.coverage_bbox)
        )
    candidates.extend(configured)

    result: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        key = url.rstrip("/").casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(url)
    return tuple(result)
