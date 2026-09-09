# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep mapped school campuses semantically and visually school-like.

OSM commonly maps the school grounds as ``amenity=school, building=no`` and the
individual classroom blocks as plain ``building=yes``. Normalization
intentionally does not replace those physical building tags with ``building=school``
because a campus may also contain explicitly typed garages, sheds, warehouses,
etc. The unfortunate side effect is that a large generic rural classroom block
then satisfies the procedural modeler's oversized-building heuristic and becomes
a barn.

This policy preserves that distinction. Generic physical footprints whose
representative point lies inside a polygonal school campus receive the neutral
``building:use=school`` semantic hint. Their original ``building=yes`` and lack
of a direct ``amenity=school`` tag remain intact. Both CWR's base family chooser
and the house-modeler classifier understand the hint before size-based rural
fallbacks run. Explicit auxiliary building types remain authoritative.

Short non-enterable gabled schools need one extra visual exception. A 3 m school
with a roughly 1 m roof rise has less than the generic 2.55 m facade-band safety
threshold, so the closed facade renderer used to replace every painted-window
wall with its plain material. The resulting correctly-classified school looked
like a barn. School closed facades may use one window band down to 1.80 m, and
the entrance atlas is guaranteed enough bays to keep a window on each side of a
central door. Enterable schools keep the normal real-opening path unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from shapely.strtree import STRtree

_NORMALIZED_SCHEMA_VERSION = 21
_GENERIC_BUILDING_KINDS = frozenset({"", "yes", "building"})
_SCHOOL_USE_VALUES = frozenset({"school", "education", "educational"})
_SCHOOL_CLOSED_WINDOW_MIN_BAND_HEIGHT_M = 1.80
_SCHOOL_FRONT_MAX_BAY_SPACING_M = 1.50
_SCHOOL_VISUAL_CACHE_REVISION = "school-closed-windows-v1"
_INSTALLED = False
_ORIGINAL_NORMALIZE_BUILDINGS = None
_ORIGINAL_STYLE_CLASSIFIER = None
_ORIGINAL_CWR_FAMILY = None
_ORIGINAL_CLOSED_FACADE_BANDS = None
_ORIGINAL_BUILDING_CACHE_KEY = None
_ORIGINAL_FIDELITY_RENDER = None
_ORIGINAL_BRIDGE_FRONT_TEXTURE = None
_ORIGINAL_TEXTURE_CACHE_IDENTITY = None


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
    # Candidate properties also contain lists such as source_ids, so avoid set
    # membership tests on arbitrary values here.
    semantic_tags = {
        str(key): str(value)
        for key, value in properties.items()
        if value is not None and value != ""
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


def _school_front_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return facade metadata that leaves windows beside a school entrance."""

    result = dict(metadata)
    raw_window = metadata.get("window") or {}
    if not isinstance(raw_window, Mapping):
        return result
    window = dict(raw_window)
    try:
        width = float(window.get("width_m", 0.0) or 0.0)
        height = float(window.get("height_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return result
    if width <= 0.0 or height <= 0.0:
        return result

    try:
        spacing = float(window.get("target_bay_spacing_m", 4.0) or 4.0)
    except (TypeError, ValueError):
        spacing = 4.0
    try:
        density = float(window.get("density_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        density = 1.0

    # Both active closed-facade renderers calculate the number of candidate
    # windows from (4 m / bay spacing) * density. Three candidates are enough for
    # a central entrance to consume the middle bay while preserving both sides.
    window["target_bay_spacing_m"] = min(
        max(0.25, spacing), _SCHOOL_FRONT_MAX_BAY_SPACING_M
    )
    window["density_multiplier"] = max(1.0, density)
    result["window"] = window
    return result


def _install_normalization_hint() -> None:
    global _ORIGINAL_NORMALIZE_BUILDINGS
    from . import normalization

    _ORIGINAL_NORMALIZE_BUILDINGS = normalization._normalize_buildings
    # Existing schema-20 bundles were normalized before this semantic hint
    # existed. Force one rebuild so a fixed executable cannot silently reuse
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


def _install_closed_school_facades() -> None:
    """Keep one painted window row on short non-enterable school walls."""

    global _ORIGINAL_CLOSED_FACADE_BANDS, _ORIGINAL_BUILDING_CACHE_KEY
    from . import procedural_buildings as buildings

    _ORIGINAL_CLOSED_FACADE_BANDS = buildings._closed_facade_bands
    _ORIGINAL_BUILDING_CACHE_KEY = buildings.cache_key

    def closed_facade_bands(
        key,
        wall_height: float,
        *,
        span_m: float,
        ground_texture: str,
        upper_texture: str,
        plain_texture: str,
        preserve_ground_texture: bool = False,
    ):
        if key.family != "school" or key.interiors:
            return _ORIGINAL_CLOSED_FACADE_BANDS(
                key,
                wall_height,
                span_m=span_m,
                ground_texture=ground_texture,
                upper_texture=upper_texture,
                plain_texture=plain_texture,
                preserve_ground_texture=preserve_ground_texture,
            )

        height = max(0.0, float(wall_height))
        if height <= 1.0e-6:
            return ()
        storeys = buildings._facade_storey_count(key, height)
        if storeys <= 0:
            return ((0.0, height, plain_texture, False),)
        band_height = min(
            buildings.VISIBLE_FACADE_STOREY_HEIGHT_M,
            height / max(1, storeys),
        )
        if band_height < _SCHOOL_CLOSED_WINDOW_MIN_BAND_HEIGHT_M - 1.0e-6:
            return ((0.0, height, plain_texture, False),)

        bands: list[tuple[float, float, str, bool]] = []
        for storey in range(storeys):
            y0 = storey * band_height
            y1 = min(height, (storey + 1) * band_height)
            requested_texture = ground_texture if storey == 0 else upper_texture
            texture = (
                requested_texture
                if storey == 0 and preserve_ground_texture
                else buildings._closed_facade_texture(
                    key,
                    requested_texture,
                    plain_texture,
                    span_m=span_m,
                    height_m=y1 - y0,
                    upper_band=storey > 0,
                )
            )
            bands.append((y0, y1, texture, texture != plain_texture))

        used_top = min(height, storeys * band_height)
        if height - used_top > 1.0e-5:
            bands.append((used_top, height, plain_texture, False))
        return tuple(bands)

    def cache_key(namespace: str, payload: Any) -> str:
        effective_namespace = str(namespace)
        if effective_namespace.startswith("procedural-building-model-") and isinstance(payload, Mapping):
            variant = payload.get("variant")
            if isinstance(variant, Mapping) and str(variant.get("family", "")).casefold() == "school":
                effective_namespace = (
                    f"{effective_namespace}-{_SCHOOL_VISUAL_CACHE_REVISION}"
                )
        return _ORIGINAL_BUILDING_CACHE_KEY(effective_namespace, payload)

    buildings._closed_facade_bands = closed_facade_bands
    buildings.cache_key = cache_key


def _install_school_front_windows() -> None:
    """Keep side windows when the closed school entrance occupies the centre bay."""

    global _ORIGINAL_FIDELITY_RENDER, _ORIGINAL_BRIDGE_FRONT_TEXTURE
    global _ORIGINAL_TEXTURE_CACHE_IDENTITY
    from . import osm_house_modeler_fidelity as fidelity
    from . import osm_house_modeler_texture_bridge as bridge

    _ORIGINAL_FIDELITY_RENDER = fidelity.render_modeler_facade_texture
    _ORIGINAL_BRIDGE_FRONT_TEXTURE = bridge.modeler_front_texture_image
    _ORIGINAL_TEXTURE_CACHE_IDENTITY = bridge.modeler_texture_cache_identity

    def render_modeler_facade_texture(
        base_image,
        metadata: Mapping[str, Any],
        *,
        family: str,
        front: bool,
    ):
        effective_metadata = (
            _school_front_metadata(metadata)
            if front and str(family).casefold() == "school"
            else metadata
        )
        return _ORIGINAL_FIDELITY_RENDER(
            base_image,
            effective_metadata,
            family=family,
            front=front,
        )

    def modeler_front_texture_image(
        family: str,
        size: int = 128,
        regional_style: str = "default",
        texture_variant: int = 0,
        outbuilding_kind: str = "",
    ):
        if str(family).casefold() != "school":
            return _ORIGINAL_BRIDGE_FRONT_TEXTURE(
                family,
                size=size,
                regional_style=regional_style,
                texture_variant=texture_variant,
                outbuilding_kind=outbuilding_kind,
            )

        # The direct asset-cache worker calls this bridge function rather than
        # the runtime texture wrapper, so render the same corrected facade here.
        base = bridge._wall_material_image(
            regional_style, int(texture_variant), int(size)
        )
        metadata = _school_front_metadata(
            bridge.texture_metadata_from_token(regional_style)
        )
        composed = render_modeler_facade_texture(
            base,
            metadata,
            family=family,
            front=True,
        )
        return bridge.cwa_exposure_compensate(composed)

    def modeler_texture_cache_identity(
        kind: str,
        *,
        style_token: str = "default",
        texture_variant: int = 0,
        family: str = "",
        outbuilding_kind: str = "",
        roof_token: str = "",
        size: int = 128,
    ) -> str:
        identity = _ORIGINAL_TEXTURE_CACHE_IDENTITY(
            kind,
            style_token=style_token,
            texture_variant=texture_variant,
            family=family,
            outbuilding_kind=outbuilding_kind,
            roof_token=roof_token,
            size=size,
        )
        if str(kind).casefold() == "front" and str(family).casefold() == "school":
            return f"{identity}|{_SCHOOL_VISUAL_CACHE_REVISION}"
        return identity

    fidelity.render_modeler_facade_texture = render_modeler_facade_texture
    bridge.modeler_front_texture_image = modeler_front_texture_image
    bridge.modeler_texture_cache_identity = modeler_texture_cache_identity


def install_school_campus_policy() -> None:
    """Install school-campus normalization, classification, and facade fixes."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_normalization_hint()
    _install_style_classifier()
    _install_cwr_family_classifier()
    _install_closed_school_facades()
    _install_school_front_windows()
    _INSTALLED = True
