# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep generic buildings inside mapped school campuses out of the barn fallback.

OSM commonly maps the school grounds as ``amenity=school, building=no`` and the
individual classroom blocks as plain ``building=yes``.  Normalization
intentionally does not replace those physical building tags with ``building=school``
because a campus may also contain explicitly typed garages, sheds, warehouses,
etc.  The unfortunate side effect is that a large generic rural classroom block
then satisfies the procedural modeler's oversized-building heuristic and becomes
a barn.

This policy preserves that distinction.  Generic physical footprints whose
representative point lies inside a polygonal school campus receive the neutral
``building:use=school`` semantic hint.  Their original ``building=yes`` and lack
of a direct ``amenity=school`` tag remain intact.  Both CWR's base family chooser
and the house-modeler classifier understand the hint before size-based rural
fallbacks run.  Explicit auxiliary building types remain authoritative.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from shapely.strtree import STRtree

_NORMALIZED_SCHEMA_VERSION = 21
_GENERIC_BUILDING_KINDS = frozenset({"", "yes", "building"})
_SCHOOL_USE_VALUES = frozenset({"school", "education", "educational"})
_INSTALLED = False
_ORIGINAL_NORMALIZE_BUILDINGS = None
_ORIGINAL_STYLE_CLASSIFIER = None
_ORIGINAL_CWR_FAMILY = None


def _school_use_applies(tags: Mapping[str, str]) -> bool:
    """Return True when a non-explicit building carries the school-use hint."""

    use = str(tags.get("building:use", "") or "").strip().casefold()
    if use not in _SCHOOL_USE_VALUES:
        return False
    building = str(tags.get("building", "") or "").strip().casefold()
    return building in _GENERIC_BUILDING_KINDS | {"school", "public", "civic"}


def _candidate_accepts_school_hint(properties: Mapping[str, Any], normalization) -> bool:
    """Do not erase explicit auxiliary or conflicting civic semantics."""

    building_kind = str(properties.get("building_kind", "yes") or "yes").strip().casefold()
    if building_kind not in _GENERIC_BUILDING_KINDS:
        return False

    existing_use = str(properties.get("building:use", "") or "").strip().casefold()
    if existing_use and existing_use not in _SCHOOL_USE_VALUES | {"yes", "building"}:
        return False

    # Reconstruct enough OSM-style tags to detect explicit shop, worship or
    # social-facility meaning already attached to an otherwise generic shell.
    semantic_tags = {
        str(key): str(value)
        for key, value in properties.items()
        if value not in {None, ""}
    }
    semantic_tags["building"] = building_kind or "yes"
    existing_semantic = normalization._semantic_building_kind(semantic_tags)
    return existing_semantic in {"", "school"}


def _school_campuses(elements, projection, boundary, normalization):
    """Yield polygonal school sites that are not themselves physical buildings."""

    for element in elements:
        raw_tags = element.get("tags")
        if not isinstance(raw_tags, Mapping):
            continue
        tags = {str(key): str(value) for key, value in raw_tags.items()}
        if normalization._semantic_building_kind(tags) != "school":
            continue
        building_value = str(tags.get("building", "") or "").strip().casefold()
        has_building = bool(building_value) and building_value not in {
            "no", "false", "0", "none",
        }
        if has_building or bool(tags.get("man_made")):
            continue
        source_id = normalization._osm_id(element)
        for polygon in normalization._element_polygons(element, projection, boundary):
            if not polygon.is_empty:
                yield polygon, source_id


def _install_normalization_hint() -> None:
    global _ORIGINAL_NORMALIZE_BUILDINGS
    from . import normalization

    _ORIGINAL_NORMALIZE_BUILDINGS = normalization._normalize_buildings
    # Existing schema-20 bundles were normalized before this semantic hint
    # existed.  Force one rebuild so a fixed executable cannot silently reuse
    # the old building metadata forever.
    normalization.NORMALIZED_SCHEMA_VERSION = max(
        int(normalization.NORMALIZED_SCHEMA_VERSION),
        _NORMALIZED_SCHEMA_VERSION,
    )

    def normalize_buildings(
        elements: Sequence[Mapping[str, Any]],
        projection,
        boundary,
        roads,
        spec,
        *,
        road_corridor=None,
        progress_callback=None,
    ):
        buildings, statistics = _ORIGINAL_NORMALIZE_BUILDINGS(
            elements,
            projection,
            boundary,
            roads,
            spec,
            road_corridor=road_corridor,
            progress_callback=progress_callback,
        )
        statistics.setdefault("school_campus_hints", 0)
        if not buildings:
            return buildings, statistics

        campuses = tuple(_school_campuses(elements, projection, boundary, normalization))
        if not campuses:
            return buildings, statistics

        tree = STRtree([candidate.geometry for candidate in buildings])
        hinted: set[int] = set()
        for campus, source_id in campuses:
            for raw_index in tree.query(campus, predicate="intersects"):
                index = int(raw_index)
                candidate = buildings[index]
                if not campus.covers(candidate.geometry.representative_point()):
                    continue
                if not _candidate_accepts_school_hint(candidate.properties, normalization):
                    continue
                candidate.properties["building:use"] = "school"
                candidate.source_ids.add(source_id)
                # The original normalizer has already materialized provenance
                # into properties by this point, so keep that public list in sync.
                candidate.properties["source_ids"] = sorted(candidate.source_ids)
                hinted.add(index)

        statistics["school_campus_hints"] = len(hinted)
        return buildings, statistics

    normalization._normalize_buildings = normalize_buildings


def _install_style_classifier() -> None:
    global _ORIGINAL_STYLE_CLASSIFIER
    from . import osm_house_modeler_styles as styles

    _ORIGINAL_STYLE_CLASSIFIER = styles.classify_building

    def classify_building(
        tags: Mapping[str, str],
        width_m: float | None = None,
        length_m: float | None = None,
        *,
        settlement: str = "rural",
    ):
        if _school_use_applies(tags):
            return styles.BuildingClassification("school", "school")
        return _ORIGINAL_STYLE_CLASSIFIER(
            tags,
            width_m,
            length_m,
            settlement=settlement,
        )

    styles.classify_building = classify_building


def _install_cwr_family_classifier() -> None:
    global _ORIGINAL_CWR_FAMILY
    from . import procedural_buildings as buildings

    _ORIGINAL_CWR_FAMILY = buildings._family

    def family(
        tags: Mapping[str, str],
        width_m: float | None = None,
        length_m: float | None = None,
        *,
        settlement_context: str = "rural",
    ) -> str:
        if _school_use_applies(tags):
            return "school"
        return _ORIGINAL_CWR_FAMILY(
            tags,
            width_m,
            length_m,
            settlement_context=settlement_context,
        )

    buildings._family = family


def install_school_campus_policy() -> None:
    """Install school-campus normalization and classification as one policy."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_normalization_hint()
    _install_style_classifier()
    _install_cwr_family_classifier()
    _INSTALLED = True
