from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected patch context missing in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one patch context in {path}, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


# 1. Country material policy: allow class/material-specific facade colours and
# replace generic wood with a vertical-board renderer when the material says so.
policy = ROOT / "src" / "cwr_worldgen" / "country_utility_material_policy.py"
replace_once(
    policy,
    '''def apply_country_utility_materials(\n    choice,\n''',
    '''def _material_colour_distribution(source: object, wall_material: str):\n    if not isinstance(source, Mapping):\n        return None\n    folded = str(wall_material or "").casefold().strip()\n    for key, values in source.items():\n        if str(key).casefold().strip() == folded:\n            return values\n    return None\n\n\ndef _facade_colour_distribution(\n    materials: Mapping[str, Any],\n    block: Mapping[str, Any],\n    wall_material: str,\n):\n    # Building-class colour pools win first (a Swedish barn should be red far\n    # more often than an ordinary rendered house), then a material-specific\n    # pool, then the country's ordinary facade distribution.\n    if block:\n        values = block.get("facade_colour_distribution")\n        if values:\n            return values\n        values = _material_colour_distribution(\n            block.get("wall_material_colour_distributions"), wall_material\n        )\n        if values:\n            return values\n    values = _material_colour_distribution(\n        materials.get("wall_material_colour_distributions"), wall_material\n    )\n    if values:\n        return values\n    return materials.get("facade_colour_distribution")\n\n\ndef apply_country_utility_materials(\n    choice,\n''',
)
replace_once(
    policy,
    '''    if not primary_colour:\n        primary_colour = _weighted_colour(\n            materials.get("facade_colour_distribution"),\n            signature + ":facade-colour",\n            palette[0] if palette else "",\n        )\n''',
    '''    if not primary_colour:\n        primary_colour = _weighted_colour(\n            _facade_colour_distribution(materials, block, wall),\n            signature + ":facade-colour",\n            palette[0] if palette else "",\n        )\n''',
)
replace_once(
    policy,
    '''def _utility_wall_base(region: str, facade: str, wall_material: str, palette: tuple[str, ...]):\n    utility_kind = _utility_kind(wall_material)\n''',
    '''def _utility_wall_base(region: str, facade: str, wall_material: str, palette: tuple[str, ...]):\n    material_text = str(wall_material or "").casefold()\n    if (\n        not material_text.startswith("utility ")\n        and "vertical" in material_text\n        and any(token in material_text for token in ("timber", "wood"))\n    ):\n        base = (\n            _textures._colour_from_name(palette[0], default=(148, 104, 70))\n            if palette else (148, 104, 70)\n        )\n        return "cwr_vertical_timber", base\n    utility_kind = _utility_kind(wall_material)\n''',
)
replace_once(
    policy,
    '''def _utility_render_wall(kind: str, base, rng: random.Random, size: int):\n    if not str(kind).startswith("utility_"):\n        return _ORIGINAL_RENDER_WALL(kind, base, rng, size)\n''',
    '''def _utility_render_wall(kind: str, base, rng: random.Random, size: int):\n    if str(kind) == "cwr_vertical_timber":\n        # Swedish painted timber is vertical board-on-board/clapboard-like\n        # cladding, not the generic modeler's broad horizontal wood courses.\n        # Keep it chunky enough for CWA while adding subtle board-to-board\n        # variation and sparse weathering instead of a flat colour slab.\n        pixels = []\n        board_width = max(16, int(round(size * 0.085)))\n        for y in range(size):\n            for x in range(size):\n                board = x // board_width\n                phase = x % board_width\n                board_shift = ((board * 17) % 9) - 4\n                colour = [\n                    _textures._clamp(channel + board_shift + rng.randint(-4, 4))\n                    for channel in base\n                ]\n                if phase < 2:\n                    colour = [_textures._clamp(int(c * 0.57)) for c in colour]\n                elif phase < 4:\n                    colour = [_textures._clamp(int(c * 1.06)) for c in colour]\n                # Sparse vertical grain and a very occasional butt joint keep\n                # the material readable without turning it into stripy noise.\n                if (y + board * 23) % 79 == 0 and phase > 4:\n                    colour = [_textures._clamp(int(c * 0.91)) for c in colour]\n                if y % max(96, int(size * 0.62)) < 2:\n                    colour = [_textures._clamp(int(c * 0.88)) for c in colour]\n                pixels.append(tuple(colour))\n        return pixels\n    if not str(kind).startswith("utility_"):\n        return _ORIGINAL_RENDER_WALL(kind, base, rng, size)\n''',
)

# 2. Opening dimensions: barn doors are useful vehicle doors, not hangar doors.
opening = ROOT / "src" / "cwr_worldgen" / "opening_dimension_policy.py"
replace_once(
    opening,
    '''_ORIGINAL_VISUAL_LOD = None\n''',
    '''_ORIGINAL_VISUAL_LOD = None\n_BARN_UTILITY_DOOR_MAX_WIDTH_M = 3.20\n_BARN_UTILITY_DOOR_MAX_HEIGHT_M = 3.30\n''',
)
replace_once(
    opening,
    '''def _front_door_uv(key) -> tuple[float, float, float, float]:\n    door = _door_metadata(key)\n    width = _number(door.get("width_m"), 0.95) or 0.95\n    height = _number(door.get("height_m"), 2.05) or 2.05\n''',
    '''def _front_door_uv(key) -> tuple[float, float, float, float]:\n    door = _door_metadata(key)\n    role = str(door.get("utility_role", "") or "").casefold()\n    width = _number(door.get("width_m"), 0.95) or 0.95\n    height = _number(door.get("height_m"), 2.05) or 2.05\n    if role:\n        width = _number(door.get("utility_width_m"), width) or width\n        height = _number(door.get("utility_height_m"), height) or height\n''',
)
replace_once(
    opening,
    '''        door = dict(metadata.get("door") or {})\n        door["utility_width_m"] = _number(source_door.get("utility_width_m"), 0.0)\n        door["utility_height_m"] = _number(source_door.get("utility_height_m"), 0.0)\n        door["utility_role"] = str(source_door.get("utility_role", "") or "")\n''',
    '''        door = dict(metadata.get("door") or {})\n        utility_width = _number(source_door.get("utility_width_m"), 0.0)\n        utility_height = _number(source_door.get("utility_height_m"), 0.0)\n        utility_role = str(source_door.get("utility_role", "") or "").casefold()\n        if utility_role == "barn":\n            if utility_width > 0.0:\n                utility_width = min(utility_width, _BARN_UTILITY_DOOR_MAX_WIDTH_M)\n            if utility_height > 0.0:\n                utility_height = min(utility_height, _BARN_UTILITY_DOOR_MAX_HEIGHT_M)\n        door["utility_width_m"] = utility_width\n        door["utility_height_m"] = utility_height\n        door["utility_role"] = utility_role\n''',
)

# 3. Texture bridge: front atlases must use utility dimensions too, and cache
# identity must notice them. Add a narrow renderer revision only for the new
# vertical-timber pixels so unrelated modeler textures remain reusable.
bridge = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_texture_bridge.py"
replace_once(
    bridge,
    '''    facade, material, palette = split_texture_token(token)\n    return json.dumps(\n        {\n            "facade": facade,\n            "material": material,\n            "palette": list(palette),\n            "variant": int(texture_variant),\n        },\n''',
    '''    facade, material, palette = split_texture_token(token)\n    material_text = str(material or "").casefold()\n    vertical_timber_revision = int(\n        not material_text.startswith("utility ")\n        and "vertical" in material_text\n        and any(token in material_text for token in ("timber", "wood"))\n    )\n    return json.dumps(\n        {\n            "facade": facade,\n            "material": material,\n            "palette": list(palette),\n            "variant": int(texture_variant),\n            "vertical_timber_renderer_revision": vertical_timber_revision,\n        },\n''',
)
replace_once(
    bridge,
    '''    if layout:\n        result.update({\n            "width_m": round(_number(door.get("width_m"), 0.0), 4),\n            "height_m": round(_number(door.get("height_m"), 0.0), 4),\n        })\n''',
    '''    if layout:\n        role = str(door.get("utility_role", "") or "").casefold()\n        width = _number(door.get("width_m"), 0.0)\n        height = _number(door.get("height_m"), 0.0)\n        if role:\n            width = _number(door.get("utility_width_m"), width) or width\n            height = _number(door.get("utility_height_m"), height) or height\n        result.update({\n            "width_m": round(width, 4),\n            "height_m": round(height, 4),\n            "utility_role": role,\n        })\n''',
)
replace_once(
    bridge,
    '''    door_w_m = max(0.0, _number(door.get("width_m"), 0.0))\n    door_h_m = max(0.0, _number(door.get("height_m"), 0.0))\n''',
    '''    door_w_m = max(0.0, _number(door.get("width_m"), 0.0))\n    door_h_m = max(0.0, _number(door.get("height_m"), 0.0))\n    utility_role = str(door.get("utility_role", "") or "").casefold()\n    if utility_role:\n        door_w_m = max(0.0, _number(door.get("utility_width_m"), door_w_m))\n        door_h_m = max(0.0, _number(door.get("utility_height_m"), door_h_m))\n''',
)

# 4. Permanent country population tool: keep this Sweden tuning after future
# upstream style syncs instead of relying on a one-off edited JSON file.
populate = ROOT / "tools" / "populate_country_visual_balance.py"
replace_once(
    populate,
    '''def populate(repo_root: Path) -> tuple[int, int]:\n''',
    '''def _weighted(field: str, values: list[tuple[str, int]]) -> list[dict[str, object]]:\n    return [{field: value, "weight": weight} for value, weight in values]\n\n\ndef _tune_sweden(document: dict) -> None:\n    document["parent_region_identifier"] = "northern_europe"\n    document["parent_region_name"] = "Northern Europe"\n    document["detail_revision"] = "2026-09-sweden-barn-house-visuals-v3"\n    provenance = document.setdefault("data_provenance", {})\n    provenance["architectural_basis"] = (\n        "curated national tuning over the Northern Europe regional baseline; "\n        "Sweden itself is defined only in country_styles"\n    )\n\n    for context_name, context in (document.get("contexts") or {}).items():\n        selection = context.get("selection") or {}\n        families = selection.get("family_distributions") or {}\n        rural = str(context_name).casefold() == "rural"\n        families["residential"] = (\n            [{"lt": 70, "style": "swedish_wood"}, {"lt": 95, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n            if rural else\n            [{"lt": 48, "style": "swedish_wood"}, {"lt": 90, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n        )\n        families["agricultural"] = (\n            [{"lt": 84, "style": "swedish_wood"}, {"lt": 100, "style": "western_brick"}]\n            if rural else\n            [{"lt": 72, "style": "swedish_wood"}, {"lt": 100, "style": "western_brick"}]\n        )\n        selection["family_distributions"] = families\n        context["selection"] = selection\n\n        details = context.get("architectural_details") or {}\n        materials = details.get("materials") or {}\n        materials["common_wall_material_distribution"] = _weighted(\n            "material",\n            [\n                ("painted vertical timber cladding", 68 if rural else 45),\n                ("stucco/render", 24 if rural else 45),\n                ("brick", 8 if rural else 10),\n            ],\n        )\n        materials["wall_material_colour_distributions"] = {\n            "painted vertical timber cladding": _weighted(\n                "colour",\n                [\n                    ("falun red", 38 if rural else 22),\n                    ("ochre yellow", 24 if rural else 22),\n                    ("white", 12 if rural else 18),\n                    ("cream", 8 if rural else 12),\n                    ("grey", 6 if rural else 12),\n                    ("dark green", 5),\n                    ("natural timber", 7 if rural else 9),\n                ],\n            ),\n            "stucco/render": _weighted(\n                "colour",\n                [\n                    ("cream", 38 if rural else 35),\n                    ("white", 32 if rural else 35),\n                    ("grey", 20 if rural else 22),\n                    ("ochre yellow", 8 if rural else 6),\n                    ("falun red", 2),\n                ],\n            ),\n        }\n\n        overrides = materials.get("building_class_overrides") or {}\n        barn = overrides.get("barn") or {}\n        barn["facade_colour_distribution"] = _weighted(\n            "colour",\n            [\n                ("falun red", 72 if rural else 62),\n                ("ochre yellow", 10 if rural else 12),\n                ("natural timber", 8 if rural else 8),\n                ("dark green", 4 if rural else 5),\n                ("grey", 3 if rural else 5),\n                ("white", 2 if rural else 5),\n                ("cream", 1 if rural else 3),\n            ],\n        )\n        overrides["barn"] = barn\n        shed = overrides.get("shed") or {}\n        shed["facade_colour_distribution"] = _weighted(\n            "colour",\n            [\n                ("falun red", 48 if rural else 34),\n                ("ochre yellow", 16),\n                ("natural timber", 14),\n                ("grey", 8 if rural else 12),\n                ("dark green", 6),\n                ("white", 5 if rural else 10),\n                ("cream", 3 if rural else 8),\n            ],\n        )\n        overrides["shed"] = shed\n        materials["building_class_overrides"] = overrides\n        details["materials"] = materials\n        context["architectural_details"] = details\n\n\ndef populate(repo_root: Path) -> tuple[int, int]:\n''',
)
replace_once(
    populate,
    '''        if document.get("identifier") == "se_sweden":\n            document["parent_region_identifier"] = "northern_europe"\n            document["parent_region_name"] = "Northern Europe"\n            provenance = document.setdefault("data_provenance", {})\n            provenance["architectural_basis"] = (\n                "curated national tuning over the Northern Europe regional baseline; "\n                "Sweden itself is defined only in country_styles"\n            )\n''',
    '''        if document.get("identifier") == "se_sweden":\n            _tune_sweden(document)\n''',
)

# Run the permanent migration now so the bundled Sweden JSON matches future syncs.
import importlib.util
spec = importlib.util.spec_from_file_location("populate_country_visual_balance", populate)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.populate(ROOT)

# 5. Regressions.
opening_test = ROOT / "tests" / "test_modeler_opening_dimensions.py"
replace_once(
    opening_test,
    '''def _token(*, utility_width: float = 0.0, utility_height: float = 0.0) -> str:\n''',
    '''def _token(\n    *, utility_width: float = 0.0, utility_height: float = 0.0, utility_role: str = ""\n) -> str:\n''',
)
replace_once(
    opening_test,
    '''            "utility_height_m": utility_height,\n            "corner_clearance_m": 0.70,\n''',
    '''            "utility_height_m": utility_height,\n            "utility_role": utility_role,\n            "corner_clearance_m": 0.70,\n''',
)
replace_once(
    opening_test,
    '''def test_enterable_windows_use_modeler_width_height_sill_and_edge_margin() -> None:\n''',
    '''def test_barn_utility_door_is_capped_below_hangar_scale() -> None:\n    key = pb.BuildingVariantKey(\n        family="agricultural",\n        building_class="barn",\n        roof_style="gabled",\n        width_m=6.0,\n        length_m=36.0,\n        height_m=6.0,\n        regional_style="swedish_wood",\n        door_width_m=0.95,\n        door_height_m=2.10,\n        door_corner_clearance_m=0.70,\n        texture_style_token=_token(\n            utility_width=3.20, utility_height=3.30, utility_role="barn"\n        ),\n    )\n    half_width, height, _pivot = pb._door_dimensions(key)\n    assert half_width * 2.0 == pytest.approx(3.20, abs=1.0e-6)\n    assert height == pytest.approx(3.30, abs=1.0e-6)\n\n\ndef test_enterable_windows_use_modeler_width_height_sill_and_edge_margin() -> None:\n''',
)

country_test = ROOT / "tests" / "test_global_country_visual_balance.py"
country_test.write_text(
    country_test.read_text(encoding="utf-8")
    + '''\n\ndef test_sweden_barns_are_red_dominant_and_stucco_avoids_timber_red_bias() -> None:\n    profile = next(p for p in load_country_profiles() if p.identifier == "se_sweden")\n    rural = profile.contexts["rural"]\n    materials = rural["architectural_details"]["materials"]\n    barn_colours = materials["building_class_overrides"]["barn"]["facade_colour_distribution"]\n    weights = {str(entry["colour"]): float(entry["weight"]) for entry in barn_colours}\n    assert weights["falun red"] >= 65\n    assert weights["falun red"] == max(weights.values())\n\n    barn_base = StyleChoice(\n        region_identifier="northern_europe", region_name="Northern Europe",\n        facade_style="swedish_wood", roof_style="gabled", context="rural",\n        family="agricultural", building_class="barn",\n        country_code="SE", country_name="Sweden",\n        country_profile_identifier="se_sweden",\n        wall_material="utility painted timber board cladding",\n        roof_material="utility sheet metal roof",\n        colour_palette=("falun red", "ochre yellow", "white", "cream", "grey", "dark green", "natural timber"),\n    )\n    sampled = []\n    for index in range(300):\n        tuned = apply_country_utility_materials(\n            barn_base, {}, seed="sweden-red-barns",\n            width_m=6.0 + index * 0.031, length_m=18.0 + (index % 41) * 0.17,\n        )\n        sampled.append(tuned.colour_palette[0])\n    assert sampled.count("falun red") / len(sampled) >= 0.58\n\n    stucco_base = StyleChoice(\n        region_identifier="northern_europe", region_name="Northern Europe",\n        facade_style="western_stucco", roof_style="gabled", context="rural",\n        family="residential", building_class="residential",\n        country_code="SE", country_name="Sweden",\n        country_profile_identifier="se_sweden",\n        wall_material="stucco/render", roof_material="clay/concrete tile",\n        colour_palette=("falun red", "ochre yellow", "white", "cream", "grey", "dark green", "natural timber"),\n    )\n    stucco_colours = []\n    for index in range(240):\n        tuned = apply_country_utility_materials(\n            stucco_base, {"building:material": "stucco"},\n            seed="sweden-stucco-colours",\n            width_m=8.0 + index * 0.021, length_m=10.0 + (index % 37) * 0.11,\n        )\n        stucco_colours.append(tuned.colour_palette[0])\n    neutral = sum(colour in {"cream", "white", "grey"} for colour in stucco_colours)\n    assert neutral / len(stucco_colours) >= 0.78\n    assert stucco_colours.count("falun red") / len(stucco_colours) <= 0.06\n''',
    encoding="utf-8",
    newline="\n",
)

texture_test = ROOT / "tests" / "test_osm_house_modeler_texture_bridge.py"
texture_test.write_text(
    texture_test.read_text(encoding="utf-8")
    + '''\n\ndef test_painted_vertical_timber_cladding_uses_vertical_board_renderer() -> None:\n    token = _token("swedish_wood", "painted vertical timber cladding", "ochre yellow")\n    facade, material, palette = split_texture_token(token)\n    kind, base = upstream_textures._choose_wall_base(facade, facade, material, palette)\n    assert kind == "cwr_vertical_timber"\n    pixels = upstream_textures._render_wall(kind, base, random.Random(1234), 256)\n    # A seam column is intentionally much darker than a board interior. This\n    # catches a regression back to the old broad horizontal wood courses.\n    seam = sum(sum(pixels[y * 256]) for y in range(256)) / (256 * 3)\n    interior = sum(sum(pixels[y * 256 + 10]) for y in range(256)) / (256 * 3)\n    assert seam < interior * 0.78\n''',
    encoding="utf-8",
    newline="\n",
)

print("Applied Sweden barn/house visual corrections")
