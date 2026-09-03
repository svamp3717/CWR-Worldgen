from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_runtime.py"
text = path.read_text(encoding="utf-8")

old = '''def _prepare_geographic_context(self, dataset, projection):
    self._modeler_projection = projection
    return _ORIGINAL_PREPARE_GEO(self, dataset, projection)
'''
new = '''def _prepare_geographic_context(self, dataset, projection):
    self._modeler_projection = projection
    result = _ORIGINAL_PREPARE_GEO(self, dataset, projection)
    # Keep CWR's long-standing public region identifiers stable. The exact modeler
    # country/region ids remain available separately and still drive StyleChoice.
    legacy = {
        "mediterranean_europe": "western_europe",
        "eastern_europe_balkans": "eastern_europe",
        "north_africa": "africa",
        "west_africa": "africa",
        "east_africa": "africa",
        "central_southern_africa": "africa",
    }
    precise_region = str(self.region_identifier or "")
    self.region_identifier = legacy.get(precise_region, precise_region) or None
    self.detected_house_style_identifier = self.region_identifier
    override_profile = house_style_preset_profile(self.house_style_preset)
    if override_profile is not None:
        self.house_style_identifier = override_profile.house_style_identifier
    elif self.country_style_identifier:
        self.house_style_identifier = self.country_style_identifier
    elif precise_region:
        self.house_style_identifier = precise_region
    return result
'''
if old not in text and new not in text:
    raise RuntimeError("prepare geographic context wrapper anchor not found")
text = text.replace(old, new, 1)

old = '''    try:
        choice = resolve_style(
            tags=tags,
            latitude=latitude,
            longitude=longitude,
            width_m=width_m,
            length_m=length_m,
            settlement_context=settlement_context,
            regional_preset=_regional_preset(self),
            seed=str(getattr(self, "world_name", "cwr-worldgen")),
        )
'''
new = '''    # CWR has several OFP-specific semantic rules that are more expressive than
    # the standalone modeler's dimension fallback (isolated dwellings, town context,
    # social facilities, shops). Feed that resolved semantic intent to the modeler
    # while preserving the original tags for explicit material/roof/opening rules.
    style_tags = dict(tags)
    building_value = str(style_tags.get("building", "") or "").casefold()
    if base.family == "shop":
        style_tags["building"] = "shop"
    elif base.family == "townhouse" and building_value in {"", "yes", "building"}:
        style_tags["building"] = "townhouse"
    elif base.family == "urban" and building_value in {"", "yes", "building", "warehouse"}:
        style_tags["building"] = "apartments"
    elif base.family == "residential" and building_value in {"", "yes", "building"}:
        style_tags["building"] = "cabin" if base.isolated_dwelling else "house"
    elif base.family == "agricultural" and building_value in {"", "yes", "building"}:
        style_tags["building"] = "barn"
    elif base.family == "outbuilding" and building_value in {"", "yes", "building"}:
        style_tags["building"] = base.outbuilding_kind or "shed"
    try:
        choice = resolve_style(
            tags=style_tags,
            latitude=latitude,
            longitude=longitude,
            width_m=width_m,
            length_m=length_m,
            settlement_context=settlement_context,
            regional_preset=_regional_preset(self),
            seed=str(getattr(self, "world_name", "cwr-worldgen")),
        )
'''
if old not in text and new not in text:
    raise RuntimeError("style resolution anchor not found")
text = text.replace(old, new, 1)

text = text.replace(
    '    family = base.family if base.family == "church" else str(choice.family or base.family)\n',
    '    # CWR family is authoritative for engine semantics and variant reservation.\n    family = base.family\n',
    1,
)
text = text.replace(
    '    levels = base.facade_storeys if family == "church" else requested_levels(tags, choice)\n',
    '    levels = base.facade_storeys if family == "church" else (1 if base.isolated_dwelling else requested_levels(style_tags, choice))\n',
    1,
)
text = text.replace(
    '        outbuilding_kind=(str(choice.outbuilding_kind or base.outbuilding_kind) if family == "outbuilding" else ""),\n',
    '        outbuilding_kind=(str(base.outbuilding_kind or choice.outbuilding_kind) if family == "outbuilding" else ""),\n',
    1,
)
old_wall = '''def _interior_wall_thickness(key):
    styled = float(getattr(key, "wall_thickness_m", 0.0) or 0.0)
    if styled > 0.0:
        return max(0.10, min(0.60, styled))
    return _ORIGINAL_INTERIOR_WALL_THICKNESS(key)
'''
new_wall = '''def _interior_wall_thickness(key):
    # Never make CWR's collision-safe shell thinner than its proven clearance.
    # Country styles may request a thicker wall and that is still honoured.
    baseline = float(_ORIGINAL_INTERIOR_WALL_THICKNESS(key))
    styled = float(getattr(key, "wall_thickness_m", 0.0) or 0.0)
    return max(baseline, max(0.10, min(0.60, styled))) if styled > 0.0 else baseline
'''
if old_wall not in text and new_wall not in text:
    raise RuntimeError("interior wall thickness anchor not found")
text = text.replace(old_wall, new_wall, 1)

path.write_text(text, encoding="utf-8", newline="\n")
