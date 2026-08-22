# SPDX-License-Identifier: GPL-3.0-or-later
"""Data-driven regional house-style catalogue.

Regional detection, settlement-specific rural/town-city descriptions,
material/colour overrides, weighted façade selection, and roof defaults live in
``house_styles/*.json``. Keeping this policy in data makes it possible to tune
architecture without editing the procedural building renderer itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

_HOUSE_STYLES_DIR = Path(__file__).resolve().parent / "house_styles"
_TOWN_CITY_CONTEXTS = frozenset({"urban", "town", "city"})
_ALLOWED_ROOF_STYLES = frozenset({"flat", "gabled", "hipped", "pyramidal", "dome", "onion"})


@dataclass(frozen=True, slots=True)
class HouseStyleContext:
    """Settlement-specific architectural policy for one geographic region."""

    description: str
    selection: Mapping[str, Any]
    roof_defaults: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RegionProfile:
    """One geographic house-style profile loaded from JSON.

    ``identifier`` intentionally preserves the older broad identifiers where a
    profile declares ``legacy_identifier``. New code should use
    ``house_style_identifier`` when it needs the precise 24-region catalogue.

    ``description``, ``selection`` and ``roof_defaults`` remain aliases for the
    rural context so older callers keep working while new code can select the
    ``town_city`` context explicitly.
    """

    identifier: str
    house_style_identifier: str
    display_name: str
    description: str
    map_region_number: int
    priority: int = 0
    polygon_lon_lat: tuple[tuple[float, float], ...] = ()
    envelopes_lon_lat: tuple[tuple[float, float, float, float], ...] = ()
    country_aliases: frozenset[str] = frozenset()
    selection: Mapping[str, Any] | None = None
    roof_defaults: Mapping[str, str] | None = None
    contexts: Mapping[str, HouseStyleContext] | None = None
    legacy_default: bool = False


def _tuple_pairs(value: Any, *, filename: str) -> tuple[tuple[float, float], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{filename}: polygon_lon_lat must be a list")
    result: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{filename}: polygon coordinate must contain lon, lat")
        result.append((float(item[0]), float(item[1])))
    return tuple(result)


def _tuple_envelopes(value: Any, *, filename: str) -> tuple[tuple[float, float, float, float], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{filename}: envelopes_lon_lat must be a list")
    result: list[tuple[float, float, float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError(f"{filename}: envelope must contain west, south, east, north")
        west, south, east, north = (float(part) for part in item)
        if west > east or south > north:
            raise ValueError(f"{filename}: invalid geographic envelope {item!r}")
        result.append((west, south, east, north))
    return tuple(result)


def _normalise_selection(value: Any, *, filename: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: selection must be an object")
    supported = value.get("supported_families", [])
    tag_rules = value.get("tag_rules", [])
    distributions = value.get("family_distributions", {})
    if not isinstance(supported, list) or not all(isinstance(item, str) for item in supported):
        raise ValueError(f"{filename}: supported_families must be a string list")
    if not isinstance(tag_rules, list):
        raise ValueError(f"{filename}: tag_rules must be a list")
    if not isinstance(distributions, dict):
        raise ValueError(f"{filename}: family_distributions must be an object")

    clean_rules: list[dict[str, Any]] = []
    for rule in tag_rules:
        if not isinstance(rule, dict):
            raise ValueError(f"{filename}: every tag rule must be an object")
        field = str(rule.get("field", "")).strip()
        style = str(rule.get("style", "")).strip()
        values = rule.get("values", [])
        families = rule.get("families")
        if not field or not style or not isinstance(values, list):
            raise ValueError(f"{filename}: malformed tag rule {rule!r}")
        if families is not None and (
            not isinstance(families, list) or not all(isinstance(item, str) for item in families)
        ):
            raise ValueError(f"{filename}: tag-rule families must be a string list")
        clean_rules.append({
            "field": field,
            "style": style,
            "values": frozenset(str(item).casefold() for item in values),
            "families": frozenset(families) if families is not None else None,
        })

    clean_distributions: dict[str, tuple[tuple[int, str], ...]] = {}
    for family, entries in distributions.items():
        if not isinstance(entries, list):
            raise ValueError(f"{filename}: distribution for {family!r} must be a list")
        clean_entries: list[tuple[int, str]] = []
        previous = -1
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{filename}: distribution entries must be objects")
            threshold = int(entry.get("lt", -1))
            style = str(entry.get("style", "")).strip()
            if threshold <= previous or not 0 <= threshold <= 100 or not style:
                raise ValueError(f"{filename}: malformed distribution entry {entry!r}")
            previous = threshold
            clean_entries.append((threshold, style))
        clean_distributions[str(family)] = tuple(clean_entries)

    return {
        "supported_families": frozenset(supported),
        "tag_rules": tuple(clean_rules),
        "family_distributions": clean_distributions,
        "default_style": str(value.get("default_style", "default")) or "default",
    }


def _normalise_roof_defaults(value: Any, *, filename: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: roof_defaults must be an object")
    clean_roofs = {str(key): str(item) for key, item in value.items()}
    unsupported = sorted(set(clean_roofs.values()) - _ALLOWED_ROOF_STYLES)
    if unsupported:
        raise ValueError(
            f"{filename}: roof_defaults contains unsupported roof style(s): {', '.join(unsupported)}"
        )
    return clean_roofs


def _normalise_context(value: Any, *, filename: str, context_name: str) -> HouseStyleContext:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: contexts.{context_name} must be an object")
    description = str(value.get("description", "")).strip()
    if not description:
        raise ValueError(f"{filename}: contexts.{context_name}.description is required")
    return HouseStyleContext(
        description=description,
        selection=_normalise_selection(
            value.get("selection", {}), filename=f"{filename}: contexts.{context_name}"
        ),
        roof_defaults=_normalise_roof_defaults(
            value.get("roof_defaults", {}), filename=f"{filename}: contexts.{context_name}"
        ),
    )


def _load_profiles() -> tuple[RegionProfile, ...]:
    if not _HOUSE_STYLES_DIR.is_dir():
        raise RuntimeError(f"house-style data directory is missing: {_HOUSE_STYLES_DIR}")

    profiles: list[RegionProfile] = []
    identifiers: set[str] = set()
    numbers: set[int] = set()
    for path in sorted(_HOUSE_STYLES_DIR.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load house-style file {path.name}: {exc}") from exc
        schema_version = int(document.get("schema_version", 0)) if isinstance(document, dict) else 0
        if not isinstance(document, dict) or schema_version not in {1, 2}:
            raise RuntimeError(f"{path.name}: unsupported or missing schema_version")

        style_identifier = str(document.get("identifier", "")).strip()
        display_name = str(document.get("display_name", "")).strip()
        map_number = int(document.get("map_region_number", 0))
        if not style_identifier or not display_name or not 1 <= map_number <= 24:
            raise RuntimeError(f"{path.name}: missing identifier/display_name/map_region_number")
        if style_identifier in identifiers or map_number in numbers:
            raise RuntimeError(f"{path.name}: duplicate house-style identifier or map number")
        identifiers.add(style_identifier)
        numbers.add(map_number)

        match = document.get("match", {})
        if not isinstance(match, dict):
            raise RuntimeError(f"{path.name}: match must be an object")
        aliases = match.get("country_aliases", [])
        if not isinstance(aliases, list):
            raise RuntimeError(f"{path.name}: country_aliases must be a list")
        legacy_identifier = str(document.get("legacy_identifier", "")).strip()
        public_identifier = legacy_identifier or style_identifier

        try:
            polygon = _tuple_pairs(match.get("polygon_lon_lat", []), filename=path.name)
            envelopes = _tuple_envelopes(match.get("envelopes_lon_lat", []), filename=path.name)
            if schema_version == 1:
                # Backward compatibility for external/custom catalogues written
                # against the first JSON schema. Treat their single policy as
                # both rural and town/city rather than failing at import time.
                rural = HouseStyleContext(
                    description=str(document.get("description", "")).strip() or display_name,
                    selection=_normalise_selection(document.get("selection", {}), filename=path.name),
                    roof_defaults=_normalise_roof_defaults(
                        document.get("roof_defaults", {}), filename=path.name
                    ),
                )
                contexts = {"rural": rural, "town_city": rural}
            else:
                raw_contexts = document.get("contexts", {})
                if not isinstance(raw_contexts, dict):
                    raise ValueError(f"{path.name}: contexts must be an object")
                missing = {"rural", "town_city"} - set(raw_contexts)
                if missing:
                    raise ValueError(
                        f"{path.name}: contexts is missing {', '.join(sorted(missing))}"
                    )
                contexts = {
                    name: _normalise_context(raw_contexts[name], filename=path.name, context_name=name)
                    for name in ("rural", "town_city")
                }
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        rural = contexts["rural"]
        profiles.append(RegionProfile(
            identifier=public_identifier,
            house_style_identifier=style_identifier,
            display_name=display_name,
            description=rural.description,
            map_region_number=map_number,
            priority=int(document.get("priority", 0)),
            polygon_lon_lat=polygon,
            envelopes_lon_lat=envelopes,
            country_aliases=frozenset(str(alias).casefold().strip() for alias in aliases if str(alias).strip()),
            selection=rural.selection,
            roof_defaults=rural.roof_defaults,
            contexts=contexts,
            legacy_default=bool(document.get("legacy_default", False)),
        ))

    if len(profiles) != 24 or numbers != set(range(1, 25)):
        raise RuntimeError("house-style catalogue must contain exactly map regions 1 through 24")
    return tuple(sorted(profiles, key=lambda item: item.map_region_number))


REGION_PROFILES: tuple[RegionProfile, ...] = _load_profiles()
HOUSE_STYLE_PRESET_AUTO = "auto"
HOUSE_STYLE_PRESET_IDENTIFIERS: tuple[str, ...] = tuple(
    profile.house_style_identifier for profile in REGION_PROFILES
)
HOUSE_STYLE_PRESET_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (profile.house_style_identifier, profile.display_name) for profile in REGION_PROFILES
)
_PROFILE_BY_STYLE_IDENTIFIER = {profile.house_style_identifier: profile for profile in REGION_PROFILES}
_PROFILE_BY_IDENTIFIER: dict[str, RegionProfile] = dict(_PROFILE_BY_STYLE_IDENTIFIER)
for profile in REGION_PROFILES:
    existing = _PROFILE_BY_IDENTIFIER.get(profile.identifier)
    if existing is None or profile.legacy_default:
        _PROFILE_BY_IDENTIFIER[profile.identifier] = profile


def get_region_profile(identifier: str | None) -> RegionProfile | None:
    if not identifier:
        return None
    return _PROFILE_BY_IDENTIFIER.get(str(identifier).casefold())


def normalise_house_style_preset(value: str | None) -> str:
    """Return a canonical 24-region building preset identifier or ``auto``.

    ``auto`` keeps the geographic area/country detection path. Explicit presets
    intentionally accept only the precise JSON catalogue identifiers rather than
    the broader legacy compatibility aliases.
    """

    text = str(value or HOUSE_STYLE_PRESET_AUTO).strip().casefold()
    if not text or text == HOUSE_STYLE_PRESET_AUTO:
        return HOUSE_STYLE_PRESET_AUTO
    if text not in _PROFILE_BY_STYLE_IDENTIFIER:
        choices = ", ".join(HOUSE_STYLE_PRESET_IDENTIFIERS)
        raise ValueError(f"unknown house-style preset {value!r}; expected auto or one of: {choices}")
    return text


def house_style_preset_profile(value: str | None) -> RegionProfile | None:
    """Resolve an explicit building preset; return ``None`` for automatic mode."""

    identifier = normalise_house_style_preset(value)
    if identifier == HOUSE_STYLE_PRESET_AUTO:
        return None
    return _PROFILE_BY_STYLE_IDENTIFIER[identifier]


def point_in_profile(longitude: float, latitude: float, profile: RegionProfile) -> bool:
    if profile.polygon_lon_lat and _point_in_polygon(longitude, latitude, profile.polygon_lon_lat):
        return True
    return any(
        west <= longitude <= east and south <= latitude <= north
        for west, south, east, north in profile.envelopes_lon_lat
    )


def _point_in_polygon(lon: float, lat: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _context_key(settlement_context: str | None) -> str:
    return "town_city" if str(settlement_context or "").casefold() in _TOWN_CITY_CONTEXTS else "rural"


def get_house_style_context(
    region_identifier: str | None, settlement_context: str = "rural"
) -> HouseStyleContext | None:
    profile = get_region_profile(region_identifier)
    if profile is None:
        return None
    contexts = profile.contexts or {}
    return contexts.get(_context_key(settlement_context)) or contexts.get("rural")


def settlement_style_description(
    region_identifier: str | None, settlement_context: str = "rural"
) -> str | None:
    context = get_house_style_context(region_identifier, settlement_context)
    return context.description if context is not None else None


def select_regional_style(
    region_identifier: str | None,
    family: str,
    tags: Mapping[str, str],
    width_m: float,
    length_m: float,
    *,
    settlement_context: str = "rural",
) -> str:
    """Select a procedural façade style from the JSON catalogue."""

    context = get_house_style_context(region_identifier, settlement_context)
    if context is None:
        return "default"
    selection = context.selection
    supported = selection["supported_families"]
    if family not in supported:
        return str(selection["default_style"])

    for rule in selection["tag_rules"]:
        families = rule["families"]
        if families is not None and family not in families:
            continue
        value = str(tags.get(rule["field"], "")).casefold()
        if value in rule["values"]:
            return str(rule["style"])

    identity = json.dumps(
        {
            "building": tags.get("building", ""),
            "name": tags.get("name", ""),
            "levels": tags.get("building:levels", ""),
            "width": round(width_m, 2),
            "length": round(length_m, 2),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    value = int.from_bytes(sha256(identity.encode("utf-8")).digest()[:2], "little") % 100
    distributions = selection["family_distributions"]
    entries = distributions.get(family) or distributions.get("*") or ()
    for threshold, style in entries:
        if value < threshold:
            return style
    return str(selection["default_style"])


def default_roof_style(
    region_identifier: str | None,
    family: str,
    *,
    settlement_context: str = "rural",
) -> str | None:
    context = get_house_style_context(region_identifier, settlement_context)
    if context is None:
        return None
    return context.roof_defaults.get(family) or context.roof_defaults.get("*")
