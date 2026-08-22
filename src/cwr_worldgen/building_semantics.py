# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Mapping, Sequence

from .house_style_catalogue import REGION_PROFILES, RegionProfile, point_in_profile


_CHRISTIAN_BUILDING_VALUES = frozenset({"church", "chapel", "cathedral"})
_NON_CHURCH_WORSHIP_BUILDINGS = frozenset({"mosque", "synagogue", "temple", "shrine"})


def is_actual_church(tags: Mapping[str, str]) -> bool:
    """Return true only for explicitly Christian church buildings.

    A generic ``amenity=place_of_worship`` is not enough. OSM uses that tag for
    mosques, synagogues, temples, meeting halls, and many other buildings that
    should not receive a Christian tower-and-spire model.
    """

    building = str(tags.get("building", "")).casefold()
    if building in _CHRISTIAN_BUILDING_VALUES:
        return True
    if building in _NON_CHURCH_WORSHIP_BUILDINGS:
        return False
    amenity = str(tags.get("amenity", "")).casefold()
    religion = str(tags.get("religion", "")).casefold()
    return amenity == "place_of_worship" and religion == "christian"


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
