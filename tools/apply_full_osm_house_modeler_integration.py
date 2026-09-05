from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"{label} anchor not found")
    return text.replace(old, new, 1)


def patch_procedural_buildings() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "procedural_buildings.py"
    text = path.read_text(encoding="utf-8")

    anchor = "    isolated_dwelling: bool = False\n"
    fields = anchor + """    # Detailed OSM House Modeler style signature. These values are deliberately
    # immutable/hashable because they affect geometry, textures and P3D cache identity.
    building_class: str = ""
    country_style_identifier: str = ""
    wall_material: str = ""
    roof_material: str = ""
    foundation_type: str = ""
    storey_height_m: float = 3.0
    wall_thickness_m: float = 0.22
    style_foundation_depth_m: float = 0.5
    visible_plinth_m: float = 0.0
    roof_pitch_degrees: float = 0.0
    eave_overhang_m: float = 0.0
    colour_palette: tuple[str, ...] = ()
    window_width_m: float = 0.0
    window_height_m: float = 0.0
    window_sill_height_m: float = 0.0
    window_edge_margin_m: float = 0.0
    window_bay_spacing_m: float = 0.0
    window_density_multiplier: float = 1.0
    window_type: str = ""
    window_placement_style: str = ""
    window_frame_material: str = ""
    door_width_m: float = 0.0
    door_height_m: float = 0.0
    door_corner_clearance_m: float = 0.0
    door_window_clearance_m: float = 0.0
    door_type: str = ""
    door_material: str = ""
    roof_storey: bool = False
    roof_storey_probability: float = 0.0
    roof_storey_windows_per_gable: int = 0
    roof_storey_spec_json: str = ""
    exterior_detail_spec_json: str = ""
    texture_style_token: str = ""
"""
    text = replace_once(text, anchor, fields, "BuildingVariantKey style fields")

    regional_pattern = re.compile(
        r"    @staticmethod\n    def _regional_style_index\(regional_style: str\) -> int:\n.*?(?=\n    @staticmethod\n    def _base36_code)",
        re.S,
    )
    if "unknown modeler styles receive deterministic slots" not in text:
        match = regional_pattern.search(text)
        if not match:
            raise RuntimeError("regional style index method not found")
        replacement = '''    @staticmethod
    def _regional_style_index(regional_style: str) -> int:
        # Keep the historical slots stable, while detailed country/material tokens and
        # new modeler facade ids receive deterministic slots in a larger 64-style bank.
        # The three-character base36 filename budget comfortably covers this expansion.
        known = {
            "default": 0, "sweden_red": 1, "sweden_yellow": 2,
            "eastern_plaster": 3, "eastern_brick": 4,
            "eastern_whitewash": 5, "eastern_panel": 6,
            "western_stucco": 3, "western_brick": 4,
            "western_stone": 5, "western_half_timber": 6,
            "africa_earth": 7, "africa_whitewash": 8,
            "africa_block": 9, "africa_colour": 10,
            "middle_east_sandstone": 11, "middle_east_whitewash": 12,
            "middle_east_adobe": 13, "middle_east_concrete": 14,
        }
        base_style = str(regional_style or "default").split("|", 1)[0]
        if regional_style in known:
            return known[regional_style]
        if base_style in known and "|" not in str(regional_style):
            return known[base_style]
        # unknown modeler styles receive deterministic slots 16..63.
        digest = sha256(str(regional_style).encode("utf-8")).digest()
        return 16 + int.from_bytes(digest[:2], "big") % 48
'''
        text = text[: match.start()] + replacement + text[match.end() :]

    text = text.replace(
        "value = (family_index * 16 + style_index) * self.texture_variants",
        "value = (family_index * 64 + style_index) * self.texture_variants",
    )

    roof_pattern = re.compile(
        r"    def _roof_texture\(self, roof_style: str, texture_variant: int = 0\) -> str:\n.*?(?=\n    def _foundation_texture)",
        re.S,
    )
    if "material_slot" not in text[text.find("def _roof_texture"):text.find("def _foundation_texture")]:
        match = roof_pattern.search(text)
        if not match:
            raise RuntimeError("roof texture method not found")
        replacement = '''    def _roof_texture(self, roof_style: str, texture_variant: int = 0) -> str:
        token = str(roof_style or "gabled")
        base_roof = token.split("|", 1)[0]
        roof_index = {
            "flat": 0, "gabled": 1, "hipped": 2,
            "pyramidal": 3, "dome": 4, "onion": 5,
        }[base_roof]
        material_slot = (
            int.from_bytes(sha256(token.encode("utf-8")).digest()[:2], "big") % 16
            if "|" in token else 0
        )
        value = (roof_index * 16 + material_slot) * self.texture_variants
        value += _normalise_texture_variant(texture_variant, self.texture_variants)
        return rf"{self.world_name}\\d\\r{self._base36_code(value)}.paa"
'''
        text = text[: match.start()] + replacement + text[match.end() :]

    start = text.index("    def write_assets(")
    write_assets = text[start:]
    write_assets = write_assets.replace(
        "(key.family, key.regional_style, key.texture_variant, key.outbuilding_kind)",
        "(key.family, (key.texture_style_token or key.regional_style), key.texture_variant, key.outbuilding_kind)",
    )
    write_assets = write_assets.replace(
        "(key.family, key.regional_style, key.texture_variant)",
        "(key.family, (key.texture_style_token or key.regional_style), key.texture_variant)",
    )
    write_assets = write_assets.replace(
        "key.family, key.regional_style, key.texture_variant, key.outbuilding_kind",
        "key.family, (key.texture_style_token or key.regional_style), key.texture_variant, key.outbuilding_kind",
    )
    write_assets = write_assets.replace(
        "key.family, key.regional_style, key.texture_variant",
        "key.family, (key.texture_style_token or key.regional_style), key.texture_variant",
    )
    write_assets = write_assets.replace(
        "(key.roof_style, key.texture_variant) for key in selected",
        "(f\"{key.roof_style}|{key.roof_material}|{','.join(key.colour_palette[:4])}\", key.texture_variant) for key in selected",
    )
    write_assets = write_assets.replace(
        "self._roof_texture(key.roof_style, key.texture_variant)",
        "self._roof_texture(f\"{key.roof_style}|{key.roof_material}|{','.join(key.colour_palette[:4])}\", key.texture_variant)",
    )
    write_assets = write_assets.replace(
        '"roof_pitch_degrees": self.roof_pitch_degrees,',
        '"roof_pitch_degrees": (key.roof_pitch_degrees or self.roof_pitch_degrees),',
    )
    write_assets = write_assets.replace(
        "roof_pitch_degrees=self.roof_pitch_degrees,",
        "roof_pitch_degrees=(key.roof_pitch_degrees or self.roof_pitch_degrees),",
    )
    text = text[:start] + write_assets

    path.write_text(text, encoding="utf-8", newline="\n")


def patch_upgrade_details() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"
    text = path.read_text(encoding="utf-8")
    import_anchor = "from . import procedural_buildings as _pb\n"
    text = replace_once(
        text,
        import_anchor,
        import_anchor + "from .osm_house_modeler_full_style import detail_spec_from_key\n",
        "detail spec import",
    )
    text = replace_once(
        text,
        "    plan = detail_plan_for_key(key, foundation_depth=foundation_depth)\n",
        "    plan = detail_plan_for_key(key, foundation_depth=foundation_depth)\n    detail_spec = detail_spec_from_key(key)\n",
        "detail spec resolution",
    )
    text = replace_once(
        text,
        "    if plan.stairs:\n        door_half, _door_height, _pivot = _pb._door_dimensions(key)\n        rise = 0.16\n",
        "    if plan.stairs:\n        stair_spec = detail_spec.get(\"stairs\") or {}\n        door_half, _door_height, _pivot = _pb._door_dimensions(key)\n        rise = max(0.08, float(stair_spec.get(\"step_rise_m\", 0.16) or 0.16))\n",
        "styled stairs",
    )
    text = text.replace(
        "min(5, int(math.ceil(max(0.18, foundation_depth) / rise)))",
        "min(max(1, int(stair_spec.get(\"max_steps\", 5) or 5)), int(math.ceil(max(0.18, foundation_depth) / rise)))",
        1,
    )
    text = text.replace("        tread = 0.30\n", "        tread = max(0.16, float(stair_spec.get(\"step_depth_m\", 0.30) or 0.30))\n", 1)
    text = text.replace(
        "            max(door_half * 2.0 + 0.45, 1.35),",
        "            max(float(stair_spec.get(\"width_m\", 0.0) or 0.0), door_half * 2.0 + 0.45, 1.35),",
        1,
    )
    text = replace_once(
        text,
        "    if plan.porch:\n        width = min(\n",
        "    if plan.porch:\n        porch_spec = detail_spec.get(\"porches\") or {}\n        width = min(\n",
        "styled porch",
    )
    text = text.replace(
        "            max(2.2, frontage * 0.26),",
        "            max(float(porch_spec.get(\"width_m\", 0.0) or 0.0), 2.2, frontage * 0.26),",
        1,
    )
    text = text.replace("        depth = 1.10\n", "        depth = max(0.45, float(porch_spec.get(\"depth_m\", 1.10) or 1.10))\n", 1)
    text = replace_once(
        text,
        "    if plan.balcony_count:\n        floor_height = min(\n",
        "    if plan.balcony_count:\n        balcony_spec = detail_spec.get(\"balconies\") or {}\n        floor_height = min(\n",
        "styled balcony",
    )
    text = text.replace(
        "                max(2.4, frontage * 0.28),",
        "                max(float(balcony_spec.get(\"width_m\", 0.0) or 0.0), 2.4, frontage * 0.28),",
        1,
    )
    text = text.replace("            depth = 1.0\n", "            depth = max(0.45, float(balcony_spec.get(\"depth_m\", 1.0) or 1.0))\n", 1)
    text = text.replace(
        "            rail_y0, rail_y1 = y, y + 0.95\n",
        "            rail_y0, rail_y1 = y, y + max(0.55, float(balcony_spec.get(\"railing_height_m\", 0.95) or 0.95))\n",
        1,
    )
    text = text.replace(
        "            posts = max(3, int(math.ceil(width / 1.2)) + 1)\n",
        "            post_spacing = max(0.45, float(balcony_spec.get(\"post_spacing_m\", 1.2) or 1.2))\n            posts = max(3, int(math.ceil(width / post_spacing)) + 1)\n",
        1,
    )
    text = replace_once(
        text,
        "    if plan.chimney_count:\n        for index in range(plan.chimney_count):\n",
        "    if plan.chimney_count:\n        chimney_spec = detail_spec.get(\"chimneys\") or {}\n        chimney_width = max(0.20, float(chimney_spec.get(\"width_m\", 0.48) or 0.48))\n        chimney_depth = max(0.20, float(chimney_spec.get(\"depth_m\", 0.40) or 0.40))\n        chimney_height = max(0.35, float(chimney_spec.get(\"height_m\", 1.15) or 1.15))\n        for index in range(plan.chimney_count):\n",
        "styled chimney",
    )
    text = text.replace("                width=0.48,\n                depth=0.40,", "                width=chimney_width,\n                depth=chimney_depth,", 1)
    text = text.replace("                y1=base_y + 1.15,", "                y1=base_y + chimney_height,", 1)
    text = text.replace("                width=0.55,\n                depth=0.47,", "                width=chimney_width + 0.07,\n                depth=chimney_depth + 0.07,", 1)
    text = text.replace("                y0=base_y + 1.15,\n                y1=base_y + 1.23,", "                y0=base_y + chimney_height,\n                y1=base_y + chimney_height + 0.08,", 1)
    text = replace_once(
        text,
        "    if plan.gutters:\n        gutter = 0.085\n",
        "    if plan.gutters:\n        rainwater_spec = detail_spec.get(\"rainwater\") or {}\n        gutter = max(0.04, float(rainwater_spec.get(\"gutter_width_m\", 0.085) or 0.085))\n        downspout_width = max(0.035, float(rainwater_spec.get(\"downspout_width_m\", 0.075) or 0.075))\n",
        "styled rainwater",
    )
    text = text.replace("                width=0.075,\n                depth=0.075,", "                width=downspout_width,\n                depth=downspout_width,", 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_house_style_compatibility() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "house_style_catalogue.py"
    text = path.read_text(encoding="utf-8")
    if "_LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER" not in text:
        marker = "HOUSE_STYLE_PRESET_AUTO = \"auto\"\n"
        if marker not in text:
            # The exact constant placement moved in older branches; inserting after imports
            # is harmless and keeps the vendored JSON bytes untouched.
            marker = "from typing import Any, Mapping, Sequence\n"
        addition = marker + """

_LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER = {
    "mediterranean_europe": "western_europe",
    "eastern_europe_balkans": "eastern_europe",
    "north_africa": "africa",
    "west_africa": "africa",
    "east_africa": "africa",
    "central_southern_africa": "africa",
}
_LEGACY_DEFAULT_STYLE_IDENTIFIERS = frozenset({
    "western_europe", "eastern_europe_balkans", "west_africa",
})
"""
        text = replace_once(text, marker, addition, "legacy style constants")
    # Upstream files intentionally do not contain CWR's old legacy_identifier keys.
    # Teach the loader the compatibility aliases in code instead of mutating vendored JSON.
    text = text.replace(
        "legacy_identifier=str(document.get(\"legacy_identifier\", \"\")).casefold(),",
        "legacy_identifier=str(document.get(\"legacy_identifier\") or _LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER.get(str(document.get(\"identifier\", path.stem)).casefold(), \"\")).casefold(),",
    )
    text = text.replace(
        "legacy_default=bool(document.get(\"legacy_default\", False)),",
        "legacy_default=bool(document.get(\"legacy_default\", False) or str(document.get(\"identifier\", path.stem)).casefold() in _LEGACY_DEFAULT_STYLE_IDENTIFIERS),",
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    patch_procedural_buildings()
    patch_upgrade_details()
    patch_house_style_compatibility()


if __name__ == "__main__":
    main()
