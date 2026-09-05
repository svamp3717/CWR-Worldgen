from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from functools import lru_cache
import json
from typing import Any, Mapping, Sequence


@dataclass(slots=True, frozen=True)
class RegionProfile:
    identifier: str
    display_name: str
    map_region_number: int
    priority: int
    country_aliases: frozenset[str]
    polygon: tuple[tuple[float, float], ...]
    envelopes: tuple[tuple[float, float, float, float], ...]
    contexts: Mapping[str, Any]
    detail_revision: str = ""
    detail_level: str = ""
    regional_overview: str = ""


@dataclass(slots=True, frozen=True)
class CountryProfile:
    identifier: str
    display_name: str
    iso_alpha2: str
    iso_alpha3: str
    parent_region_identifier: str
    aliases: frozenset[str]
    envelopes: tuple[tuple[float, float, float, float], ...]
    geometry: Mapping[str, Any] | None
    contexts: Mapping[str, Any]
    detail_revision: str = ""
    detail_level: str = ""
    country_overview: str = ""
    data_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BuildingClassification:
    family: str
    building_class: str
    outbuilding_kind: str = ""


@dataclass(slots=True, frozen=True)
class StyleChoice:
    region_identifier: str
    region_name: str
    facade_style: str
    roof_style: str
    context: str
    family: str
    building_class: str = "residential"
    outbuilding_kind: str = ""
    country_code: str = ""
    country_name: str = ""
    country_profile_identifier: str = ""
    country_detail_level: str = ""
    country_overview: str = ""
    wall_material: str = ""
    roof_material: str = ""
    foundation_type: str = ""
    storey_height_m: float = 3.0
    wall_thickness_m: float = 0.22
    default_levels: int = 0
    automatic_max_levels: int = 0
    foundation_depth_m: float = 1.0
    visible_plinth_m: float = 0.0
    roof_pitch_degrees: float = 35.0
    eave_overhang_m: float = 0.35
    colour_palette: tuple[str, ...] = ()
    window_spec: Mapping[str, Any] = field(default_factory=dict)
    door_spec: Mapping[str, Any] = field(default_factory=dict)
    family_profile: Mapping[str, Any] = field(default_factory=dict)
    building_class_profile: Mapping[str, Any] = field(default_factory=dict)
    roof_storey: bool = False
    roof_storey_probability: float = 0.0
    roof_storey_spec: Mapping[str, Any] = field(default_factory=dict)
    exterior_detail_spec: Mapping[str, Any] = field(default_factory=dict)
    detail_revision: str = ""
    detail_level: str = ""
    regional_overview: str = ""


_DENSITY_SCALE = {
    "none": 0.0,
    "very-low": 0.22,
    "low": 0.45,
    "low-medium": 0.68,
    "medium": 1.0,
    "high": 1.30,
    "high_on_front": 1.05,
}


def project_style_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    root_catalogue = project_root / "house_styles"
    if root_catalogue.is_dir():
        return root_catalogue
    return Path(__file__).resolve().parent / "house_styles"


def discover_style_dir(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        return explicit if explicit.is_dir() else None
    cwd_catalogue = Path.cwd() / "house_styles"
    if cwd_catalogue.is_dir() and any(cwd_catalogue.glob("*.json")):
        return cwd_catalogue.resolve()
    catalogue = project_style_dir()
    if catalogue.is_dir() and any(catalogue.glob("*.json")):
        return catalogue.resolve()
    return None


def load_profiles(style_dir: Path | None = None) -> tuple[RegionProfile, ...]:
    directory = discover_style_dir(style_dir)
    if directory is None:
        return ()
    profiles: list[RegionProfile] = []
    for path in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or int(doc.get("schema_version", 0)) not in {1, 2}:
            continue
        match = doc.get("match") or {}
        if not isinstance(match, dict):
            continue
        contexts = doc.get("contexts")
        if not isinstance(contexts, dict):
            old = {
                "description": doc.get("description", doc.get("display_name", "")),
                "selection": doc.get("selection", {}),
                "roof_defaults": doc.get("roof_defaults", {}),
            }
            contexts = {"rural": old, "town_city": old}
        try:
            profiles.append(RegionProfile(
                identifier=str(doc.get("identifier", path.stem)).casefold(),
                display_name=str(doc.get("display_name", path.stem)),
                map_region_number=int(doc.get("map_region_number", 0)),
                priority=int(doc.get("priority", 0)),
                country_aliases=frozenset(str(v).casefold() for v in match.get("country_aliases", [])),
                polygon=tuple((float(x), float(y)) for x, y in match.get("polygon_lon_lat", [])),
                envelopes=tuple(tuple(float(v) for v in box) for box in match.get("envelopes_lon_lat", [])),
                contexts=contexts,
                detail_revision=str(doc.get("detail_revision", "")),
                detail_level=str(doc.get("detail_level", "")),
                regional_overview=str(doc.get("regional_overview", "")),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(profiles, key=lambda p: p.map_region_number))


def project_country_style_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    root_catalogue = project_root / "country_styles"
    if root_catalogue.is_dir():
        return root_catalogue
    return Path(__file__).resolve().parent / "country_styles"


def discover_country_style_dir(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        return explicit if explicit.is_dir() else None
    cwd_catalogue = Path.cwd() / "country_styles"
    if cwd_catalogue.is_dir() and any(cwd_catalogue.glob("*.json")):
        return cwd_catalogue.resolve()
    catalogue = project_country_style_dir()
    if catalogue.is_dir() and any(catalogue.glob("*.json")):
        return catalogue.resolve()
    return None


@lru_cache(maxsize=4)
def _load_country_profiles_cached(directory_text: str) -> tuple[CountryProfile, ...]:
    directory = Path(directory_text)
    profiles: list[CountryProfile] = []
    for path in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, Mapping) or not doc.get("iso_alpha2"):
            continue
        match = doc.get("match") or {}
        contexts = doc.get("contexts") or {}
        if not isinstance(match, Mapping) or not isinstance(contexts, Mapping):
            continue
        geometry = match.get("geometry")
        if geometry is not None and not isinstance(geometry, Mapping):
            geometry = None
        try:
            profiles.append(CountryProfile(
                identifier=str(doc.get("identifier", path.stem)).casefold(),
                display_name=str(doc.get("display_name", path.stem)),
                iso_alpha2=str(doc.get("iso_alpha2", "")).upper(),
                iso_alpha3=str(doc.get("iso_alpha3", "")).upper(),
                parent_region_identifier=str(doc.get("parent_region_identifier", "")).casefold(),
                aliases=frozenset(str(v).casefold() for v in match.get("country_aliases", [])),
                envelopes=tuple(tuple(float(v) for v in box) for box in match.get("envelopes_lon_lat", [])),
                geometry=geometry,
                contexts=contexts,
                detail_revision=str(doc.get("detail_revision", "")),
                detail_level=str(doc.get("detail_level", "")),
                country_overview=str(doc.get("country_overview", "")),
                data_provenance=doc.get("data_provenance") or {},
            ))
        except (TypeError, ValueError):
            continue
    return tuple(profiles)


def load_country_profiles(country_style_dir: Path | None = None) -> tuple[CountryProfile, ...]:
    directory = discover_country_style_dir(country_style_dir)
    if directory is None:
        return ()
    return _load_country_profiles_cached(str(directory.resolve()))


def clear_country_profile_cache() -> None:
    """Force the next country catalogue load to re-read local JSON files."""
    _load_country_profiles_cached.cache_clear()


def find_country_profile(
    profiles: Sequence[CountryProfile], value: str | None,
) -> CountryProfile | None:
    """Resolve a user-facing country selector value to a country profile.

    Accept ISO alpha-2/alpha-3 codes, profile identifiers, display names, aliases,
    and GUI labels such as ``SE — Sweden``. ``auto``/empty means no forced country.
    """
    raw = str(value or "").strip()
    if not raw or raw.casefold() == "auto":
        return None
    # GUI labels deliberately begin with ISO2 so the stable part survives a
    # translated/edited display name.
    if "—" in raw:
        raw = raw.split("—", 1)[0].strip()
    elif " - " in raw and len(raw.split(" - ", 1)[0].strip()) in {2, 3}:
        raw = raw.split(" - ", 1)[0].strip()
    key = raw.casefold()
    for profile in profiles:
        if key in {
            profile.identifier.casefold(),
            profile.display_name.casefold(),
            profile.iso_alpha2.casefold(),
            profile.iso_alpha3.casefold(),
        } or key in profile.aliases:
            return profile
    raise ValueError(f"Unknown country preset {value!r}")


def country_selector_label(profile: CountryProfile) -> str:
    return f"{profile.iso_alpha2} — {profile.display_name}"


def _point_in_polygon(lon: float, lat: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > lat) != (yj > lat)) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _contains(profile: RegionProfile, lon: float, lat: float) -> bool:
    if profile.polygon and _point_in_polygon(lon, lat, profile.polygon):
        return True
    return any(w <= lon <= e and s <= lat <= n for w, s, e, n in profile.envelopes)


def choose_region(profiles: Sequence[RegionProfile], lon: float, lat: float, tags: Mapping[str, str]) -> RegionProfile | None:
    country = (tags.get("addr:country") or tags.get("country") or "").casefold().strip()
    candidates = []
    for profile in profiles:
        country_match = bool(country and country in profile.country_aliases)
        geo_match = _contains(profile, lon, lat)
        if country_match or geo_match:
            candidates.append((1 if country_match else 0, profile.priority, profile))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2].map_region_number))[2]


def _point_in_geo_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
    points: list[tuple[float, float]] = []
    for point in ring:
        if isinstance(point, Sequence) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return len(points) >= 3 and _point_in_polygon(lon, lat, points)


def _point_in_country_geometry(lon: float, lat: float, geometry: Mapping[str, Any] | None) -> bool:
    if not geometry:
        return False
    kind = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates") or []
    polygons = coordinates if kind == "MultiPolygon" else [coordinates] if kind == "Polygon" else []
    for polygon in polygons:
        if not isinstance(polygon, Sequence) or not polygon:
            continue
        outer = polygon[0]
        if not isinstance(outer, Sequence) or not _point_in_geo_ring(lon, lat, outer):
            continue
        holes = polygon[1:]
        if any(isinstance(hole, Sequence) and _point_in_geo_ring(lon, lat, hole) for hole in holes):
            continue
        return True
    return False


def _country_envelope_area(profile: CountryProfile) -> float:
    areas = [max(0.0, e - w) * max(0.0, n - s) for w, s, e, n in profile.envelopes]
    return min(areas) if areas else float("inf")


def choose_country(
    profiles: Sequence[CountryProfile], lon: float, lat: float, tags: Mapping[str, str],
) -> CountryProfile | None:
    explicit_values = [
        tags.get("addr:country"), tags.get("country"), tags.get("country_code"),
        tags.get("is_in:country_code"), tags.get("ISO3166-1:alpha2"), tags.get("ISO3166-1:alpha3"),
    ]
    explicit = {str(value).casefold().strip() for value in explicit_values if value}
    if explicit:
        for profile in profiles:
            if explicit & profile.aliases or profile.iso_alpha2.casefold() in explicit or profile.iso_alpha3.casefold() in explicit:
                return profile

    geometry_matches = [profile for profile in profiles if _point_in_country_geometry(lon, lat, profile.geometry)]
    if geometry_matches:
        return min(geometry_matches, key=_country_envelope_area)

    # Only use rectangular envelopes when no polygon geometry exists for that
    # country. This is mostly for tiny territories absent from the bundled
    # low-resolution boundary source, and avoids bbox overlap overriding a real border.
    envelope_matches = [
        profile for profile in profiles
        if profile.geometry is None and any(w <= lon <= e and s <= lat <= n for w, s, e, n in profile.envelopes)
    ]
    if envelope_matches:
        return min(envelope_matches, key=_country_envelope_area)
    return None


def _outbuilding_kind_from_dimensions(width_m: float | None, length_m: float | None) -> str:
    if width_m is None or length_m is None:
        return "shed"
    minor, major = sorted((max(0.1, float(width_m)), max(0.1, float(length_m))))
    return "garage" if minor >= 2.4 and major >= 4.8 else "shed"


def classify_building(
    tags: Mapping[str, str],
    width_m: float | None = None,
    length_m: float | None = None,
    *,
    settlement: str = "rural",
) -> BuildingClassification:
    value = (tags.get("building") or "yes").casefold().strip()
    amenity = (tags.get("amenity") or "").casefold().strip()
    man_made = (tags.get("man_made") or "").casefold().strip()

    if value == "cabin":
        return BuildingClassification("residential", "cabin")
    if value == "cottage":
        return BuildingClassification("residential", "cottage")
    if value in {"garage", "garages", "carport"}:
        return BuildingClassification("outbuilding", "garage", "garage")
    if value in {"shed", "hut"}:
        return BuildingClassification("outbuilding", "shed", "shed")
    if value in {"barn", "farm", "farm_auxiliary", "stable", "cowshed", "greenhouse"}:
        return BuildingClassification("agricultural", "barn")
    if value in {"industrial", "warehouse", "hangar", "factory", "manufacture"} or man_made in {"works", "storage_tank", "silo"}:
        specific = "warehouse" if value == "warehouse" else "industrial"
        return BuildingClassification("industrial", specific)
    if value in {"terrace", "row_house", "townhouse"}:
        return BuildingClassification("townhouse", "townhouse")
    if value in {"apartments", "commercial", "office", "retail", "hotel", "civic", "public"}:
        return BuildingClassification("urban", value)
    if amenity == "school" or value in {"school", "university", "college", "kindergarten"}:
        return BuildingClassification("school", "school")
    if tags.get("shop") or value in {"shop", "supermarket", "kiosk"}:
        return BuildingClassification("shop", "shop")

    if width_m is not None and length_m is not None:
        minor, major = sorted((max(0.1, float(width_m)), max(0.1, float(length_m))))
        area = minor * major
        aspect = major / minor
        if value in {"", "yes"} and area <= 72.0 and minor <= 8.5 and major <= 12.0:
            kind = _outbuilding_kind_from_dimensions(minor, major)
            return BuildingClassification("outbuilding", kind, kind)
        oversized_rural = (
            (major >= 32.0 and minor >= 8.0)
            or minor >= 20.0
            or area >= 600.0
            or (aspect >= 3.0 and major >= 24.0)
        )
        if value in {"", "yes"} and settlement != "town_city" and oversized_rural:
            return BuildingClassification("agricultural", "barn")
    return BuildingClassification("residential", "residential")


def building_family(tags: Mapping[str, str], width_m: float | None = None, length_m: float | None = None) -> str:
    return classify_building(tags, width_m, length_m).family


def settlement_context(
    tags: Mapping[str, str], requested: str = "auto", *,
    width_m: float | None = None, length_m: float | None = None,
) -> str:
    if requested in {"rural", "town_city"}:
        return requested
    family = classify_building(tags, width_m, length_m).family
    return "town_city" if family in {"urban", "townhouse", "shop", "school", "industrial"} else "rural"


def _normalise_roof(value: str | None) -> str | None:
    if not value:
        return None
    key = value.casefold().replace("-", "_")
    aliases = {
        "gable": "gabled", "gabled": "gabled", "saltbox": "gabled", "half_hipped": "hipped",
        "hip": "hipped", "hipped": "hipped", "pyramid": "pyramidal", "pyramidal": "pyramidal",
        "flat": "flat", "dome": "dome", "onion": "onion", "skillion": "gabled", "shed": "gabled",
        "mansard": "hipped",
    }
    return aliases.get(key)


def _digest(seed: str) -> bytes:
    return sha256(seed.encode("utf-8")).digest()


def _deterministic_percent(seed: str) -> int:
    return int.from_bytes(_digest(seed)[:4], "big") % 100


def _pick_index(seed: str, size: int) -> int:
    return 0 if size <= 0 else int.from_bytes(_digest(seed)[:4], "big") % size


def _pick_from_list(values: Sequence[Any], seed: str, default: Any = "") -> Any:
    clean = [value for value in values if value not in (None, "")]
    if not clean:
        return default
    return clean[_pick_index(seed, len(clean))]


def _pick_weighted(
    entries: Sequence[Mapping[str, Any]], seed: str, *, value_key: str,
    weight_key: str = "weight", default: Any = "",
) -> Any:
    total = 0.0
    normalised: list[tuple[float, Any]] = []
    for entry in entries:
        try:
            weight = float(entry.get(weight_key, 0))
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0.0:
            continue
        total += weight
        normalised.append((total, entry.get(value_key)))
    if not normalised:
        return default
    roll = (int.from_bytes(_digest(seed)[:8], "big") / 2**64) * total
    for upper, value in normalised:
        if roll <= upper:
            return value
    return normalised[-1][1]


def _pick_cumulative(entries: Sequence[Mapping[str, Any]], seed: str, default: str) -> str:
    roll = _deterministic_percent(seed)
    for entry in entries:
        try:
            upper = int(entry.get("lt", 100))
        except (TypeError, ValueError):
            upper = 100
        if roll < upper:
            return str(entry.get("style", default))
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default



def _wall_thickness_m(geometry: Mapping[str, Any], wall_material: str) -> float:
    values = geometry.get("wall_thickness_m") or {}
    if not isinstance(values, Mapping):
        return max(0.10, min(0.55, _as_float(values, 0.22)))
    material = str(wall_material or "").casefold()
    if any(token in material for token in ("timber", "wood", "panel", "siding", "metal", "sheet")):
        key = "lightweight"
    elif any(token in material for token in ("insulat", "heavy", "rammed earth")):
        key = "heavy_or_insulated"
    else:
        key = "masonry"
    return max(0.10, min(0.55, _as_float(values.get(key), _as_float(values.get("masonry"), 0.22))))

def _range_mid(value: Any, default: float) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        return (_as_float(value[0], default) + _as_float(value[1], default)) * 0.5
    return _as_float(value, default)


def _window_spec(
    details: Mapping[str, Any], family_profile: Mapping[str, Any], family: str,
    building_class: str, seed: str,
) -> dict[str, Any]:
    windows = details.get("windows") or {}
    dims = windows.get("default_dimensions_m") or {}
    special_sets = windows.get("special_sets") or {}
    preferred = str(family_profile.get("preferred_window_set") or "context_default")
    density_name = str(family_profile.get("window_density") or "medium")
    density_multiplier = _DENSITY_SCALE.get(density_name, 1.0)

    width = _as_float(dims.get("width"), 1.15)
    height = _as_float(dims.get("height"), 1.25)
    sill = _as_float(dims.get("sill_height"), 0.9)
    edge_margin = _as_float(dims.get("minimum_edge_margin"), 0.55)
    bay_spacing = _as_float(dims.get("target_bay_spacing"), 3.0)
    type_name = str(_pick_weighted(windows.get("type_distribution") or [], seed + ":type", value_key="type", default="casement"))

    set_name = preferred
    if preferred not in special_sets:
        set_name = {
            "urban": "large_regular",
            "shop": "storefront",
            "school": "large_regular",
            "industrial": "industrial",
            "agricultural": "utility_small",
            "outbuilding": "utility_small",
        }.get(family, preferred)
    spec = special_sets.get(set_name) or {}
    if spec:
        width = _range_mid(spec.get("width_m"), width)
        height = _range_mid(spec.get("height_m"), height)
        type_name = str(_pick_from_list(spec.get("types") or [], seed + ":set", type_name))

    procedural = windows.get("procedural_placement") or {}
    placement_values: dict[str, Any] = {
        key: value for key, value in procedural.items()
        if key not in {"family_overrides", "building_class_overrides"}
    }
    family_overrides = procedural.get("family_overrides") or {}
    if isinstance(family_overrides, Mapping):
        override = family_overrides.get(family) or {}
        if isinstance(override, Mapping):
            placement_values.update(override)
    class_overrides = procedural.get("building_class_overrides") or {}
    if isinstance(class_overrides, Mapping):
        override = class_overrides.get(building_class) or {}
        if isinstance(override, Mapping):
            placement_values.update(override)

    placement_style = str(_pick_weighted(
        placement_values.get("style_distribution") or [],
        seed + ":placement-style", value_key="style", default="regular_aligned",
    ))

    # Building-class constraints are intentionally stricter than broad regional
    # family profiles. A barn/garage/shed is not a tiny house merely because a
    # regional outbuilding profile mentions utility windows. Warehouses may have
    # glazing, but it should be the exception rather than a domestic window grid.
    if "density_multiplier" in placement_values:
        density_multiplier = max(0.0, _as_float(placement_values.get("density_multiplier"), density_multiplier))
    window_probability = max(0.0, min(1.0, _as_float(placement_values.get("window_probability"), 1.0)))
    if building_class in {"shed", "garage", "barn"}:
        density_multiplier = 0.0
        window_probability = 0.0
    elif building_class == "warehouse":
        window_probability = max(0.0, min(1.0, _as_float(placement_values.get("window_probability"), 0.20)))
        density_multiplier = min(density_multiplier, max(0.0, _as_float(placement_values.get("density_multiplier"), 0.30)))
    if window_probability < 1.0:
        unit = int.from_bytes(_digest(seed + ":window-presence")[:8], "big") / 2**64
        if unit >= window_probability:
            density_multiplier = 0.0

    return {
        "type": type_name,
        "set_name": set_name,
        "density": density_name,
        "density_multiplier": density_multiplier,
        "window_probability": window_probability,
        "width_m": width,
        "height_m": height,
        "sill_height_m": sill,
        "edge_margin_m": edge_margin,
        "target_bay_spacing_m": bay_spacing,
        "frame_material": str(_pick_from_list(windows.get("frame_materials") or [], seed + ":frame", "painted timber")),
        "glazing": str(windows.get("glazing", "")),
        "shutters_and_screens": tuple(str(v) for v in (windows.get("shutters_and_screens") or [])),
        "trim": str(windows.get("trim", "")),
        "placement": dict(windows.get("placement") or {}),
        "rule": str(family_profile.get("window_rule", "")),
        "placement_style": placement_style,
        "horizontal_jitter_fraction": _as_float(placement_values.get("horizontal_jitter_fraction"), 0.0),
        "vertical_jitter_m": _as_float(placement_values.get("vertical_jitter_m"), 0.0),
        "omit_bay_probability": _as_float(placement_values.get("omit_bay_probability"), 0.0),
        "floor_phase_shift_fraction": _as_float(placement_values.get("floor_phase_shift_fraction"), 0.0),
        "front_density_multiplier": _as_float(placement_values.get("front_density_multiplier"), 1.0),
        "side_density_multiplier": _as_float(placement_values.get("side_density_multiplier"), 1.0),
        "rear_density_multiplier": _as_float(placement_values.get("rear_density_multiplier"), 1.0),
        "minimum_windows_per_primary_facade": max(0, int(_as_float(placement_values.get("minimum_windows_per_primary_facade"), 1))),
        "maximum_windows_per_wall": max(0, int(_as_float(placement_values.get("maximum_windows_per_wall"), 12))),
        "paired_group_gap_fraction": _as_float(placement_values.get("paired_group_gap_fraction"), 0.34),
    }


def _door_spec(
    details: Mapping[str, Any], family_profile: Mapping[str, Any], family: str,
    building_class: str, outbuilding_kind: str, seed: str,
) -> dict[str, Any]:
    doors = details.get("doors") or {}
    placement = doors.get("placement") or {}
    primary = doors.get("primary_entry_dimensions_m") or {}
    service = doors.get("service_entry_dimensions_m") or {}
    dtype = str(_pick_weighted(doors.get("type_distribution") or [], seed + ":type", value_key="type", default="panel"))
    utility_role = ""
    block: Mapping[str, Any] = {}
    if family == "industrial":
        utility_role, block = "warehouse", doors.get("warehouse") or {}
    elif family == "agricultural":
        utility_role, block = "barn", doors.get("barn") or {}
    elif family == "outbuilding" and outbuilding_kind == "garage":
        utility_role, block = "garage", doors.get("garage") or {}
    elif family == "outbuilding":
        utility_role = "shed"
    elif family == "shop":
        utility_role, block = "shop", doors.get("shop") or {}
    elif family == "school":
        utility_role, block = "school", doors.get("school") or {}
    if block:
        dtype = str(block.get("type") or dtype)

    return {
        "type": dtype,
        "set_name": str(family_profile.get("preferred_door_set") or "context_default"),
        "materials": tuple(str(v) for v in (doors.get("materials") or [])),
        "primary_width_m": _as_float(primary.get("width"), 0.95),
        "primary_height_m": _as_float(primary.get("height"), 2.1),
        "service_width_m": _as_float(service.get("width"), 0.9),
        "service_height_m": _as_float(service.get("height"), 2.05),
        "utility_role": utility_role,
        "utility_width_m": _range_mid(block.get("width_m"), _as_float(primary.get("width"), 0.95)),
        "utility_height_m": _range_mid(block.get("height_m"), _as_float(primary.get("height"), 2.1)),
        "corner_clearance_m": _as_float(placement.get("corner_clearance_m"), 0.7),
        "keep_clear_of_windows_m": _as_float(placement.get("keep_clear_of_windows_m"), 0.35),
        "placement": dict(placement),
        "rule": str(family_profile.get("door_rule", "")),
        "building_class": building_class,
    }




def _seeded_range(value: Any, seed: str, default: float) -> float:
    """Pick a deterministic value from a scalar or [min,max] range."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        low = _as_float(value[0], default)
        high = _as_float(value[1], default)
        if high < low:
            low, high = high, low
        unit = int.from_bytes(_digest(seed)[:8], "big") / 2**64
        return low + (high - low) * unit
    return _as_float(value, default)


def _feature_probability(block: Mapping[str, Any], family: str, building_class: str) -> float:
    by_class = block.get("probability_by_building_class") or {}
    by_family = block.get("probability_by_family") or {}
    value = None
    if isinstance(by_class, Mapping):
        value = by_class.get(building_class)
    if value is None and isinstance(by_family, Mapping):
        value = by_family.get(family)
    if value is None:
        value = block.get("probability", 0.0)
    return max(0.0, min(1.0, _as_float(value, 0.0)))


def _exterior_detail_spec(
    details: Mapping[str, Any], family: str, building_class: str,
    tags: Mapping[str, str], seed: str,
) -> dict[str, Any]:
    """Resolve region/country exterior-detail rules into deterministic choices.

    The JSON keeps probabilities, material/style distributions and dimensional
    ranges.  This function performs the seeded selection once so the geometry
    layer can remain pleasantly ignorant of national architectural policy.
    """
    raw = details.get("exterior_details") or {}
    if not isinstance(raw, Mapping):
        return {}

    result: dict[str, Any] = {}
    explicit_yes = {"yes", "true", "1", "present"}

    for feature in ("stairs", "porches", "chimneys", "balconies"):
        block = raw.get(feature) or {}
        if not isinstance(block, Mapping):
            continue
        probability = _feature_probability(block, family, building_class)
        tag_name = {
            "stairs": "entrance:steps",
            "porches": "porch",
            "chimneys": "chimney",
            "balconies": "balcony",
        }[feature]
        tagged = str(tags.get(tag_name, "")).casefold().strip()
        forced = tagged in explicit_yes
        unit = int.from_bytes(_digest(seed + ":" + feature)[:8], "big") / 2**64
        enabled = forced or (bool(block.get("enabled", True)) and unit < probability)
        if feature == "chimneys" and building_class in {"shed", "garage", "apartments", "apartment"}:
            probability = 0.0
            enabled = False
        style = str(_pick_weighted(
            block.get("styles") or [], seed + f":{feature}:style",
            value_key="type", default=str(block.get("default_style", feature.rstrip("s"))),
        ))
        material = str(_pick_weighted(
            block.get("materials") or [], seed + f":{feature}:material",
            value_key="material", default=str(block.get("material", "")),
        ))
        result[feature] = {
            "enabled": enabled,
            "probability": probability,
            "type": style,
            "material": material,
            "width_m": _seeded_range(block.get("width_m"), seed + f":{feature}:width", 1.8),
            "depth_m": _seeded_range(block.get("depth_m"), seed + f":{feature}:depth", 1.0),
            "height_m": _seeded_range(block.get("height_m"), seed + f":{feature}:height", 1.0),
            "step_rise_m": _seeded_range(block.get("step_rise_m"), seed + f":{feature}:rise", 0.16),
            "step_depth_m": _seeded_range(block.get("step_depth_m"), seed + f":{feature}:step-depth", 0.30),
            "max_steps": max(1, int(_as_float(block.get("max_steps"), 4))),
            "railing_height_m": _seeded_range(block.get("railing_height_m"), seed + f":{feature}:rail", 0.95),
            "post_spacing_m": _seeded_range(block.get("post_spacing_m"), seed + f":{feature}:post-spacing", 1.4),
            "count": max(1, int(_as_float(_pick_weighted(
                block.get("count_distribution") or [], seed + f":{feature}:count",
                value_key="count", default=1,
            ), 1))),
            "rules": dict(block.get("rules") or {}),
        }

    rain = raw.get("rainwater") or {}
    if isinstance(rain, Mapping):
        gutter_probability = _feature_probability(rain, family, building_class)
        unit = int.from_bytes(_digest(seed + ":rainwater")[:8], "big") / 2**64
        result["rainwater"] = {
            "enabled": bool(rain.get("enabled", True)) and unit < gutter_probability,
            "probability": gutter_probability,
            "material": str(rain.get("material", "painted metal")),
            "gutter_width_m": _seeded_range(rain.get("gutter_width_m"), seed + ":gutter-width", 0.10),
            "downspout_width_m": _seeded_range(rain.get("downspout_width_m"), seed + ":downspout-width", 0.08),
            "downspouts": max(0, int(_as_float(rain.get("downspouts"), 2))),
            "rules": dict(rain.get("rules") or {}),
        }

    budget = raw.get("feature_budget") or {}
    if isinstance(budget, Mapping):
        result["feature_budget"] = {
            "minimum": max(0, int(_as_float(budget.get("minimum"), 0))),
            "maximum": max(0, int(_as_float(budget.get("maximum"), 5))),
        }
    return result

def _roof_storey_choice(
    details: Mapping[str, Any], family: str, building_class: str, roof_style: str,
    tags: Mapping[str, str], seed: str, *, explicit_roof: bool,
) -> tuple[bool, float, dict[str, Any], str]:
    """Choose whether the uppermost level should live inside the roof volume.

    ``roof:levels`` is authoritative when present. Otherwise regional JSON
    probabilities are sampled deterministically by seed/building class. The
    current geometry supports gable-end windows only, so a procedurally selected
    roof storey can switch an untagged incompatible roof to gabled when the
    region explicitly allows that compatibility fallback.
    """
    raw = details.get("roof_storeys") or {}
    if not isinstance(raw, Mapping):
        return False, 0.0, {}, roof_style
    eligible = {str(v).casefold() for v in (raw.get("eligible_roof_shapes") or ["gabled"])}
    by_class = raw.get("probability_by_building_class") or {}
    by_family = raw.get("probability_by_family") or {}
    probability = _as_float(
        (by_class.get(building_class) if isinstance(by_class, Mapping) else None),
        _as_float((by_family.get(family) if isinstance(by_family, Mapping) else None), 0.0),
    )
    probability = max(0.0, min(1.0, probability))

    roof_levels_text = str(tags.get("roof:levels", "")).strip()
    explicit_roof_levels = roof_levels_text != ""
    roof_levels = _as_float(roof_levels_text, 0.0) if explicit_roof_levels else 0.0
    if explicit_roof_levels:
        selected = roof_levels > 0.0
    else:
        selected = (int.from_bytes(_digest(seed + ":roof-storey")[:8], "big") / 2**64) < probability

    selected_roof = roof_style
    if selected and selected_roof not in eligible:
        if not explicit_roof and bool(raw.get("force_compatible_roof_when_selected", False)) and "gabled" in eligible:
            selected_roof = "gabled"
        else:
            selected = False

    policy = raw.get("window_policy") or {}
    spec = dict(policy) if isinstance(policy, Mapping) else {}
    spec["minimum_total_levels"] = max(2, int(_as_float(raw.get("minimum_total_levels"), 2)))
    spec["eligible_roof_shapes"] = tuple(sorted(eligible))
    distribution = spec.get("windows_per_gable_distribution") or []
    spec["windows_per_gable"] = max(1, int(_as_float(_pick_weighted(
        distribution if isinstance(distribution, Sequence) else [],
        seed + ":attic-window-count", value_key="count", default=1,
    ), 1)))
    return selected, probability, spec, selected_roof


def choose_style(
    profiles: Sequence[RegionProfile], lon: float, lat: float, tags: Mapping[str, str], way_id: int,
    requested_context: str = "auto", preset: str = "auto", *,
    country_preset: str = "auto",
    width_m: float | None = None, length_m: float | None = None, seed: int | str = 0,
) -> StyleChoice:
    context_key = settlement_context(tags, requested_context, width_m=width_m, length_m=length_m)
    classification = classify_building(tags, width_m, length_m, settlement=context_key)
    family = classification.family
    country_profiles = load_country_profiles()
    if str(country_preset or "auto").strip().casefold() != "auto" and not country_profiles:
        raise ValueError("A country preset was requested, but no local country_styles catalogue is available")
    forced_country = find_country_profile(country_profiles, country_preset) if country_profiles else None
    country_profile = forced_country or (choose_country(country_profiles, lon, lat, tags) if country_profiles else None)
    profile = None

    if forced_country is not None:
        # A selected country is authoritative and always brings its correct parent
        # region with it. This keeps country overrides layered on the architecture
        # baseline they were authored for, regardless of coordinates or region UI.
        profile = next((p for p in profiles if p.identifier == forced_country.parent_region_identifier), None)
        if profile is None:
            raise ValueError(
                f"Country preset {forced_country.display_name!r} requires missing region "
                f"{forced_country.parent_region_identifier!r}"
            )
    elif preset != "auto":
        profile = next((p for p in profiles if p.identifier == preset.casefold()), None)
        if profile is None:
            raise ValueError(f"Unknown style preset {preset!r}")
        # A manually forced regional preset is authoritative only when the country
        # itself is still automatic. Auto-detected country detail is applied only
        # if it belongs to that region.
        if country_profile is not None and country_profile.parent_region_identifier != profile.identifier:
            country_profile = None
    else:
        profile = choose_region(profiles, lon, lat, tags)
        if country_profile is not None:
            parent = next((p for p in profiles if p.identifier == country_profile.parent_region_identifier), None)
            if parent is not None:
                profile = parent

    explicit_roof = _normalise_roof(tags.get("roof:shape"))
    seed_text = f"{way_id}:{seed}:{family}:{classification.building_class}"

    if profile is None:
        region_id, region_name = _fallback_region(lon, lat)
        if explicit_roof:
            roof = explicit_roof
        elif family in {"agricultural", "outbuilding"}:
            roof = "gabled"
        elif family == "industrial":
            roof = "flat"
        else:
            roof = _fallback_roof(region_id)
        return StyleChoice(
            region_id, region_name, _fallback_facade(region_id, tags), roof, context_key,
            family, classification.building_class, classification.outbuilding_kind,
            wall_material=str(tags.get("building:material", "")),
            roof_material=str(tags.get("roof:material", "")),
            wall_thickness_m=0.22,
            default_levels=1 if classification.building_class == "cabin" else (2 if classification.building_class == "cottage" else 0),
            automatic_max_levels=1 if classification.building_class == "cabin" else (2 if classification.building_class == "cottage" else 0),
        )

    context_source = country_profile.contexts if country_profile is not None else profile.contexts
    context = context_source.get(context_key) or context_source.get("rural") or {}
    selection = context.get("selection") or {}
    details = context.get("architectural_details") or {}
    source_identifier = country_profile.identifier if country_profile is not None else profile.identifier
    materials = details.get("materials") or {}
    geometry = details.get("geometry_defaults") or {}
    roof_detail = geometry.get("roof") or {}
    family_profile = (details.get("building_family_profiles") or {}).get(family) or {}
    building_class_profile = (details.get("building_class_profiles") or {}).get(classification.building_class) or {}

    facade = str(selection.get("default_style", "default"))
    for rule in selection.get("tag_rules", []):
        field = str(rule.get("field", ""))
        values = {str(v).casefold() for v in rule.get("values", [])}
        families = set(rule.get("families") or [])
        if field and tags.get(field, "").casefold() in values and (not families or family in families):
            facade = str(rule.get("style", facade))
            break
    else:
        distributions = selection.get("family_distributions") or {}
        facade = _pick_cumulative(
            distributions.get(family) or distributions.get("*") or [],
            seed_text + f":{source_identifier}:facade",
            facade,
        )

    roof = explicit_roof
    if roof is None:
        family_default = _normalise_roof((context.get("roof_defaults") or {}).get(family))
        weighted = _normalise_roof(str(_pick_weighted(
            roof_detail.get("shape_distribution") or [], seed_text + f":{source_identifier}:roof",
            value_key="shape", default="",
        )))
        roof = family_default or weighted
        if roof is None and family == "industrial":
            roof = "flat"
        if roof is None:
            roof = _normalise_roof((context.get("roof_defaults") or {}).get("*")) or "gabled"

    wall_materials = materials.get("common_wall_materials") or []
    roof_materials = roof_detail.get("materials") or materials.get("common_roof_materials") or []
    foundation = geometry.get("foundation") or {}
    pitch = roof_detail.get("pitch_degrees") or {}
    eaves = roof_detail.get("eave_overhang_m") or {}

    wall_material = str(tags.get("building:material") or _pick_from_list(wall_materials, seed_text + ":wall-material", facade))
    roof_material = str(tags.get("roof:material") or _pick_from_list(roof_materials, seed_text + ":roof-material", ""))
    foundation_type = str(_pick_from_list(foundation.get("types") or [], seed_text + ":foundation", "concrete foundation"))
    window_spec = _window_spec(
        details, family_profile, family, classification.building_class,
        seed_text + f":{source_identifier}:window",
    )
    door_spec = _door_spec(
        details, family_profile, family, classification.building_class,
        classification.outbuilding_kind, seed_text + f":{source_identifier}:door",
    )
    roof_storey, roof_storey_probability, roof_storey_spec, roof = _roof_storey_choice(
        details, family, classification.building_class, roof, tags,
        seed_text + f":{source_identifier}", explicit_roof=explicit_roof is not None,
    )
    exterior_detail_spec = _exterior_detail_spec(
        details, family, classification.building_class, tags,
        seed_text + f":{source_identifier}:exterior-details",
    )

    storey_heights = geometry.get("storey_height_m") or {}
    return StyleChoice(
        profile.identifier,
        profile.display_name,
        facade,
        roof,
        context_key,
        family,
        classification.building_class,
        classification.outbuilding_kind,
        country_code=country_profile.iso_alpha2 if country_profile is not None else "",
        country_name=country_profile.display_name if country_profile is not None else "",
        country_profile_identifier=country_profile.identifier if country_profile is not None else "",
        country_detail_level=country_profile.detail_level if country_profile is not None else "",
        country_overview=country_profile.country_overview if country_profile is not None else "",
        wall_material=wall_material,
        roof_material=roof_material,
        foundation_type=foundation_type,
        storey_height_m=_as_float(building_class_profile.get("storey_height_m"), _as_float(storey_heights.get(family), 3.0)),
        wall_thickness_m=_wall_thickness_m(geometry, wall_material),
        default_levels=max(0, int(_as_float(building_class_profile.get("default_levels"), 0))),
        automatic_max_levels=max(0, int(_as_float(building_class_profile.get("automatic_max_levels"), 0))),
        foundation_depth_m=_as_float(foundation.get("default_below_grade_depth_m"), 1.0),
        visible_plinth_m=_as_float(foundation.get("default_visible_plinth_m"), 0.0),
        roof_pitch_degrees=_as_float(pitch.get("preferred"), 35.0),
        eave_overhang_m=_range_mid(eaves, 0.35),
        colour_palette=tuple(str(v) for v in (materials.get("typical_colour_palette") or [])),
        window_spec=window_spec,
        door_spec=door_spec,
        family_profile=family_profile,
        building_class_profile=building_class_profile,
        roof_storey=roof_storey,
        roof_storey_probability=roof_storey_probability,
        roof_storey_spec=roof_storey_spec,
        exterior_detail_spec=exterior_detail_spec,
        detail_revision=profile.detail_revision,
        detail_level=profile.detail_level,
        regional_overview=profile.regional_overview,
    )


def _fallback_region(lon: float, lat: float) -> tuple[str, str]:
    if lat >= 66:
        return "arctic_polar", "Arctic / Polar"
    if -25 <= lon <= 45 and 35 <= lat <= 72:
        return "europe", "Europe"
    if -170 <= lon <= -50 and lat >= 15:
        return "north_america", "North America"
    if -90 <= lon <= -30 and -60 <= lat <= 15:
        return "south_america", "South America"
    if -20 <= lon <= 55 and -36 <= lat <= 35:
        return "africa", "Africa"
    if 25 <= lon <= 180 and -10 <= lat <= 80:
        return "asia", "Asia"
    if 110 <= lon <= 180 and -50 <= lat <= -10:
        return "oceania", "Oceania"
    return "global", "Global"


def _fallback_roof(region: str) -> str:
    return "flat" if region == "africa" else "gabled"


def _fallback_facade(region: str, tags: Mapping[str, str]) -> str:
    material = (tags.get("building:material") or "").casefold()
    if material:
        return material
    return {"north_america": "wood", "europe": "stucco", "africa": "stucco", "asia": "concrete"}.get(region, "default")
