# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Mapping, Sequence

from .house_style_catalogue import REGION_PROFILES, RegionProfile, point_in_profile


_CHRISTIAN_BUILDING_VALUES = frozenset({"church", "chapel", "cathedral"})
_MOSQUE_BUILDING_VALUES = frozenset({"mosque"})
_SYNAGOGUE_BUILDING_VALUES = frozenset({"synagogue"})
_TEMPLE_BUILDING_VALUES = frozenset({"temple"})
_SHRINE_BUILDING_VALUES = frozenset({"shrine"})
_NON_CHURCH_WORSHIP_BUILDINGS = frozenset(
    _MOSQUE_BUILDING_VALUES
    | _SYNAGOGUE_BUILDING_VALUES
    | _TEMPLE_BUILDING_VALUES
    | _SHRINE_BUILDING_VALUES
)
_TEMPLE_RELIGIONS = frozenset({
    "buddhist",
    "hindu",
    "jain",
    "shinto",
    "sikh",
    "taoist",
})
_ORTHODOX_DENOMINATION_MARKERS = frozenset({
    "coptic",
    "armenian_apostolic",
    "armenian apostolic",
    "ethiopian_orthodox",
    "eritrean_orthodox",
    "syriac_orthodox",
})


def _fold_tag(tags: Mapping[str, str], name: str) -> str:
    return str(tags.get(name, "") or "").casefold().strip()


def _is_orthodox_christian(tags: Mapping[str, str]) -> bool:
    denomination = _fold_tag(tags, "denomination").replace("-", "_")
    religion = _fold_tag(tags, "religion").replace("-", "_")
    if "orthodox" in denomination or "orthodox" in religion:
        return True
    return denomination in _ORTHODOX_DENOMINATION_MARKERS


def worship_building_class(tags: Mapping[str, str]) -> str:
    """Return a global worship-building semantic class or an empty string.

    OSM uses ``amenity=place_of_worship`` across religions, while ``building=*``
    may be either specific (``mosque``, ``synagogue``, ``church``) or generic.
    Classify the religious building first so country/region house palettes cannot
    accidentally style it as an ordinary dwelling. The result is architectural,
    not a statement about the occupants or the surrounding population.
    """

    building = _fold_tag(tags, "building")
    amenity = _fold_tag(tags, "amenity")
    religion = _fold_tag(tags, "religion")

    if building in _MOSQUE_BUILDING_VALUES:
        return "mosque"
    if building in _SYNAGOGUE_BUILDING_VALUES:
        return "synagogue"
    if building in _TEMPLE_BUILDING_VALUES:
        return "temple"
    if building in _SHRINE_BUILDING_VALUES:
        return "shrine"

    christian_building = building in _CHRISTIAN_BUILDING_VALUES
    place_of_worship = amenity == "place_of_worship"
    if christian_building or (place_of_worship and religion == "christian"):
        return "orthodox_church" if _is_orthodox_christian(tags) else "church"

    if not place_of_worship:
        return ""
    if religion in {"muslim", "islam"}:
        return "mosque"
    if religion in {"jewish", "judaism"}:
        return "synagogue"
    if religion in _TEMPLE_RELIGIONS:
        return "temple"
    if religion == "christian":
        return "orthodox_church" if _is_orthodox_christian(tags) else "church"
    return "place_of_worship"


def is_worship_building(tags: Mapping[str, str]) -> bool:
    return bool(worship_building_class(tags))


def is_actual_church(tags: Mapping[str, str]) -> bool:
    """Return true only for Christian church-family buildings.

    Orthodox churches remain in CWR's dedicated church geometry family. Mosques,
    synagogues, temples and generic worship halls deliberately do not receive the
    Christian tower-and-spire model.
    """

    return worship_building_class(tags) in {"church", "orthodox_church"}


def _best_profile(
    candidates: Sequence[RegionProfile], longitude: float, latitude: float
) -> RegionProfile | None:
    if not candidates:
        return None
    geographically_matching = [
        profile for profile in candidates if point_in_profile(longitude, latitude, profile)
    ]
    pool = geographically_matching or list(candidates)
    return max(pool, key=lambda profile: (profile.priority, -profile.map_region_number))


def detect_region(
    bbox: tuple[float, float, float, float],
    tag_sources: Sequence[Mapping[str, str]] = (),
) -> RegionProfile | None:
    """Detect the current architectural house-style region.

    The 24-region catalogue is loaded from ``house_styles/*.json``. Explicit
    country tags remain authoritative; when a country can reasonably span more
    than one style region, the selected bbox centre breaks the tie. Without
    country metadata, the bbox centre is matched against the JSON envelopes.

    ``RegionProfile.identifier`` retains the older broad identifiers for API
    compatibility. ``house_style_identifier`` exposes the precise 24-region
    identifier used by the procedural building selector.
    """

    south, west, north, east = bbox
    latitude = (south + north) * 0.5
    longitude = (west + east) * 0.5

    explicit_country_seen = False
    for tags in tag_sources:
        country = str(
            tags.get("addr:country") or tags.get("country_code") or ""
        ).casefold().strip()
        if not country:
            continue
        explicit_country_seen = True
        candidates = [
            profile for profile in REGION_PROFILES if country in profile.country_aliases
        ]
        selected = _best_profile(candidates, longitude, latitude)
        if selected is not None:
            return selected
    if explicit_country_seen:
        return None

    candidates = [
        profile for profile in REGION_PROFILES
        if point_in_profile(longitude, latitude, profile)
    ]
    return _best_profile(candidates, longitude, latitude)
