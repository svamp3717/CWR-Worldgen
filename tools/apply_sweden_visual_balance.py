from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match in {path}: {old[:80]!r}; got {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


# 1. Porches are disabled globally. Keep the upstream/country detail metadata for
# provenance, but never turn it into CWA geometry.
replace_once(
    ROOT / "src/cwr_worldgen/osm_house_modeler_upgrade.py",
    '''    porch = (\n        pedestrian\n        and key.family in {"residential", "townhouse"}\n        and key.width_m >= 4.8\n        and _chance(key, "porch", porch_p)\n    )\n''',
    '''    # Porches/canopies do not read well in CWA's low-resolution visual LOD.\n    # Keep their country metadata available for provenance, but never generate\n    # porch geometry.\n    porch = False\n''',
)
replace_once(
    ROOT / "src/cwr_worldgen/osm_house_modeler_runtime.py",
    '    porch = enabled("porches") and pedestrian and key.family in {"residential", "townhouse"}\n',
    '    # Country profiles may describe porches, but CWR intentionally suppresses\n    # their visual geometry because the current implementation does not read well\n    # in-game.\n    porch = False\n',
)

# 2. Make per-country material/colour distributions first-class data. Sweden is
# the first profile to use these extra fields; other countries continue to use
# their existing upstream-compatible lists unchanged.
policy_path = ROOT / "src/cwr_worldgen/country_utility_material_policy.py"
policy = policy_path.read_text(encoding="utf-8")
marker = '\n\ndef apply_country_utility_materials(\n'
if policy.count(marker) != 1:
    raise RuntimeError("country utility policy function marker changed")
colour_helper = r'''

def _weighted_colour(values: object, seed: str, fallback: str) -> str:
    """Pick a weighted facade colour from explicit country data."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return str(fallback or "")
    choices: list[tuple[str, float]] = []
    for entry in values:
        if isinstance(entry, Mapping):
            colour = str(entry.get("colour", entry.get("color", "")) or "").strip()
            try:
                weight = max(0.0, float(entry.get("weight", 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
        else:
            colour = str(entry or "").strip()
            weight = 1.0
        if colour and weight > 0.0:
            choices.append((colour, weight))
    if not choices:
        return str(fallback or "")
    total = sum(weight for _colour, weight in choices)
    unit = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big") / 2**64
    target = unit * total
    running = 0.0
    for colour, weight in choices:
        running += weight
        if target < running:
            return colour
    return choices[-1][0]
'''
policy = policy.replace(marker, colour_helper + marker, 1)
start = policy.index('def apply_country_utility_materials(\n')
end = policy.index('\n\ndef _utility_kind', start)
new_apply = r'''def apply_country_utility_materials(
    choice,
    tags: Mapping[str, str],
    *,
    seed: str,
    width_m: float,
    length_m: float,
):
    """Apply material and facade-colour pools explicitly stored in the profile."""
    materials, geometry = _context_details(choice)
    override_name = _override_name(choice, tags)
    overrides = materials.get("building_class_overrides") or {}
    block = {}
    if override_name and isinstance(overrides, Mapping):
        candidate = overrides.get(override_name) or {}
        if isinstance(candidate, Mapping):
            block = candidate

    signature = ":".join((
        str(seed),
        str(getattr(choice, "country_profile_identifier", "") or getattr(choice, "region_identifier", "")),
        str(getattr(choice, "context", "")),
        str(getattr(choice, "building_class", "")),
        str(getattr(choice, "family", "")),
        str(getattr(choice, "facade_style", "")),
        f"{float(width_m):.2f}",
        f"{float(length_m):.2f}",
    ))

    wall = str(getattr(choice, "wall_material", "") or "")
    roof = str(getattr(choice, "roof_material", "") or "")
    if not str(tags.get("building:material", "") or "").strip():
        if block:
            wall = _weighted_pick(block.get("wall_materials"), signature + ":wall", wall)
        else:
            wall = _weighted_pick(
                materials.get("common_wall_material_distribution"),
                signature + ":wall",
                wall,
            )
    if block and not str(tags.get("roof:material", "") or "").strip():
        roof = _weighted_pick(block.get("roof_materials"), signature + ":roof", roof)

    palette = tuple(str(value) for value in getattr(choice, "colour_palette", ()) if str(value).strip())
    explicit_colour = str(
        tags.get("building:colour")
        or tags.get("building:color")
        or ""
    ).strip()
    primary_colour = explicit_colour
    if not primary_colour:
        primary_colour = _weighted_colour(
            materials.get("facade_colour_distribution"),
            signature + ":facade-colour",
            palette[0] if palette else "",
        )
    if primary_colour:
        primary_key = primary_colour.casefold()
        palette = (primary_colour,) + tuple(
            value for value in palette if value.casefold() != primary_key
        )

    thickness = float(getattr(choice, "wall_thickness_m", 0.22) or 0.22)
    if wall != getattr(choice, "wall_material", ""):
        try:
            thickness = float(_styles._wall_thickness_m(geometry, wall))
        except (AttributeError, TypeError, ValueError):
            pass
    return replace(
        choice,
        wall_material=wall,
        roof_material=roof,
        wall_thickness_m=thickness,
        colour_palette=palette,
    )
'''
policy = policy[:start] + new_apply + policy[end:]
policy_path.write_text(policy, encoding="utf-8", newline="\n")

# 3. Roof material is authoritative. A wall/facade palette must not repaint a
# standing-seam/slate/tile roof red just because Falun red is a facade option.
replace_once(
    ROOT / "src/cwr_worldgen/osm_house_modeler_texture_bridge.py",
    '''    palette = tuple(v for v in (parts[2].split(",") if len(parts) > 2 else ()) if v)\n    kind, base = _upstream._choose_roof_base(shape, material)\n    if palette:\n        base = _upstream._colour_from_name(palette[0], default=base)\n    rng = random.Random(_seed(f"roof:{roof_style}:{texture_variant}"))\n''',
    '''    kind, base = _upstream._choose_roof_base(shape, material)\n    # Wall/facade colours are not roof colours. Keep roof material authoritative\n    # and seed only from roof semantics so changing a facade colour cannot even\n    # perturb the roof noise pattern.\n    rng = random.Random(_seed(f"roof:{shape}|{material}:{texture_variant}"))\n''',
)

# 4. The texture semantics changed, so old red-biased PAAs must not be restored.
replace_once(
    ROOT / "src/cwr_worldgen/opening_dimension_policy.py",
    '        metadata["texture_renderer_revision"] = 3\n',
    '        metadata["texture_renderer_revision"] = 4\n',
)

# 5. Final P3D reuse must never change building class or primary appearance.
# The previous performance cap could map ordinary residences to cabin assets and
# heavily skew material frequencies. Reuse stays aggressive inside an appearance
# bucket, but the bucket itself is now preserved.
budget_path = ROOT / "src/cwr_worldgen/building_asset_budget_policy.py"
budget = budget_path.read_text(encoding="utf-8")
needle = '''    pool = same_mode\n\n    compatible = set(library._compatible_families(requested.family))\n'''
replacement = '''    pool = same_mode\n\n    # Performance reuse must not rewrite the architectural identity selected by\n    # the country/style system. In the Lundby80 diagnostic PBO the old broad\n    # family-compatible reuse mapped many ordinary residences to cabin assets and\n    # collapsed a balanced timber/stucco/brick request mix into mostly brick.\n    # Preserve class and the primary visible material/colour/roof group, then\n    # continue using the mature physical-fit chooser inside that group.\n    def appearance_group(key):\n        palette = tuple(getattr(key, "colour_palette", ()) or ())\n        primary_colour = str(palette[0]) if palette else ""\n        return (\n            str(getattr(key, "family", "")),\n            str(getattr(key, "building_class", "")),\n            str(getattr(key, "outbuilding_kind", "")),\n            str(getattr(key, "wall_material", "")),\n            primary_colour,\n            str(getattr(key, "roof_material", "")),\n            str(getattr(key, "roof_style", "")),\n            int(getattr(key, "facade_storeys", 1) or 1),\n        )\n\n    requested_group = appearance_group(requested)\n    same_appearance = [candidate for candidate in pool if appearance_group(candidate) == requested_group]\n    if not same_appearance:\n        return None\n    pool = same_appearance\n\n    compatible = set(library._compatible_families(requested.family))\n'''
if budget.count(needle) != 1:
    raise RuntimeError("building budget reuse insertion point changed")
budget = budget.replace(needle, replacement, 1)
budget_path.write_text(budget, encoding="utf-8", newline="\n")

# 6. Sweden-specific data tuning. The profile now states explicit weighted
# residential materials and facade colours, and makes brick a rare barn finish.
sweden_path = ROOT / "src/cwr_worldgen/country_styles/SE_Sweden.json"
doc = json.loads(sweden_path.read_text(encoding="utf-8"))
doc["detail_revision"] = "2026-09-sweden-visual-balance-v2"
for context_name, context in doc["contexts"].items():
    materials = context["architectural_details"]["materials"]
    if context_name == "rural":
        materials["common_wall_material_distribution"] = [
            {"material": "painted vertical timber cladding", "weight": 52},
            {"material": "stucco/render", "weight": 33},
            {"material": "brick", "weight": 15},
        ]
        materials["facade_colour_distribution"] = [
            {"colour": "falun red", "weight": 24},
            {"colour": "ochre yellow", "weight": 24},
            {"colour": "cream", "weight": 16},
            {"colour": "white", "weight": 12},
            {"colour": "grey", "weight": 10},
            {"colour": "natural timber", "weight": 10},
            {"colour": "dark green", "weight": 4},
        ]
    else:
        materials["common_wall_material_distribution"] = [
            {"material": "painted vertical timber cladding", "weight": 30},
            {"material": "stucco/render", "weight": 50},
            {"material": "brick", "weight": 20},
        ]
        materials["facade_colour_distribution"] = [
            {"colour": "falun red", "weight": 12},
            {"colour": "ochre yellow", "weight": 16},
            {"colour": "cream", "weight": 22},
            {"colour": "white", "weight": 22},
            {"colour": "grey", "weight": 20},
            {"colour": "natural timber", "weight": 5},
            {"colour": "dark green", "weight": 3},
        ]
    barn = materials["building_class_overrides"]["barn"]
    barn["wall_materials"] = [
        {"material": "utility painted timber board cladding", "weight": 68},
        {"material": "utility corrugated metal cladding", "weight": 20},
        {"material": "utility rendered masonry wall", "weight": 6},
        {"material": "utility structural brick", "weight": 6},
    ]
    barn["selection_note"] = (
        "Sweden-specific barn pool: painted timber dominates, corrugated metal is "
        "secondary, and structural brick is intentionally rare. Explicit OSM "
        "building:material/roof:material tags override this default."
    )
    materials["visual_balance_revision"] = "2026-09-sweden-red-brick-balance-v2"
    materials["facade_colour_provenance"] = (
        "Weighted facade colours are stored explicitly so Falun red is a common "
        "Swedish cue rather than the automatic first colour on every building."
    )
sweden_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

# 7. Tests: update the old forced-porch expectation and add regressions for
# colour/material balance and safe final reuse.
fidelity_path = ROOT / "tests/test_osm_house_modeler_geometry_fidelity.py"
fidelity = fidelity_path.read_text(encoding="utf-8")
old = '''    spec = upgrade.detail_spec_from_key(placement.requested)\n    assert spec["porches"]["enabled"]\n    assert spec["chimneys"]["enabled"]\n'''
new = '''    spec = upgrade.detail_spec_from_key(placement.requested)\n    # Source country metadata may still describe/force a porch, but CWR no\n    # longer emits porch geometry because it does not read well in-game.\n    assert spec["porches"]["enabled"]\n    assert not upgrade.detail_plan_for_key(\n        placement.requested, foundation_depth=placement.requested.foundation_depth_m\n    ).porch\n    assert spec["chimneys"]["enabled"]\n'''
if fidelity.count(old) != 1:
    raise RuntimeError("fidelity porch assertion changed")
fidelity_path.write_text(fidelity.replace(old, new, 1), encoding="utf-8", newline="\n")

budget_test_path = ROOT / "tests/test_building_asset_budget_policy.py"
budget_test = budget_test_path.read_text(encoding="utf-8")
if "from dataclasses import replace\n" not in budget_test:
    budget_test = budget_test.replace(
        "from __future__ import annotations\n\n",
        "from __future__ import annotations\n\nfrom dataclasses import replace\n",
        1,
    )
append = r'''

def test_final_budget_never_changes_building_class_or_primary_material_group() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="FinalBudgetAppearance",
        maximum_variants=1,
    )
    house_key = replace(
        _variant(),
        building_class="residential",
        colour_palette=("cream", "falun red"),
    )
    cabin_key = replace(
        _variant(overhang=0.33),
        building_class="cabin",
        wall_material="painted vertical timber cladding",
        roof_material="standing-seam metal",
        colour_palette=("ochre yellow", "falun red"),
        texture_style_token="swedish_wood|painted vertical timber cladding~|ochre yellow,falun red",
    )
    first = library.register_placement(_placement(house_key))
    second = library.register_placement(_placement(cabin_key))

    assert first.selected.building_class == "residential"
    assert second.selected.building_class == "cabin"
    assert second.selected.wall_material == "painted vertical timber cladding"
    assert second.selected.colour_palette[0] == "ochre yellow"
    # Fidelity groups may exceed a tiny synthetic numerical cap rather than\n    # silently changing architecture/material identity.
    assert len(library._usage) == 2
'''
if "test_final_budget_never_changes_building_class_or_primary_material_group" not in budget_test:
    budget_test += append
budget_test_path.write_text(budget_test, encoding="utf-8", newline="\n")

sweden_test = ROOT / "tests/test_sweden_visual_balance.py"
sweden_test.write_text(r'''from __future__ import annotations

from collections import Counter

from cwr_worldgen.country_utility_material_policy import apply_country_utility_materials
from cwr_worldgen.osm_house_modeler_styles import StyleChoice, load_country_profiles
from cwr_worldgen.osm_house_modeler_texture_bridge import modeler_roof_texture_image


def _choice(*, building_class: str = "residential", family: str = "residential") -> StyleChoice:
    return StyleChoice(
        region_identifier="sweden",
        region_name="Sweden",
        facade_style="swedish_wood",
        roof_style="gabled",
        context="rural",
        family=family,
        building_class=building_class,
        country_code="SE",
        country_name="Sweden",
        country_profile_identifier="se_sweden",
        wall_material="brick",
        roof_material="standing-seam metal",
        colour_palette=(
            "falun red", "ochre yellow", "white", "cream",
            "dark green", "grey", "black", "natural timber",
        ),
    )


def test_sweden_profile_has_explicit_visual_balance_data_in_both_contexts() -> None:
    sweden = next(profile for profile in load_country_profiles() if profile.identifier == "se_sweden")
    for context in ("rural", "town_city"):
        materials = sweden.contexts[context]["architectural_details"]["materials"]
        assert materials["common_wall_material_distribution"]
        colours = materials["facade_colour_distribution"]
        red = next(item["weight"] for item in colours if item["colour"] == "falun red")
        assert red <= 24
        barn = materials["building_class_overrides"]["barn"]["wall_materials"]
        brick = next(item["weight"] for item in barn if item["material"] == "utility structural brick")
        assert brick <= 6


def test_sweden_weighted_facade_colours_are_not_stuck_on_falun_red() -> None:
    colours = Counter()
    materials = Counter()
    base = _choice()
    for index in range(240):
        tuned = apply_country_utility_materials(
            base,
            {},
            seed="SwedenBalance",
            width_m=6.0 + index * 0.07,
            length_m=8.0 + (index % 17) * 0.11,
        )
        colours[tuned.colour_palette[0]] += 1
        materials[tuned.wall_material] += 1
    assert len(colours) >= 5
    assert colours["falun red"] < 90
    assert materials["brick"] < 70
    assert materials["painted vertical timber cladding"] > materials["brick"]


def test_sweden_barn_brick_is_rare_and_osm_overrides_still_win() -> None:
    base = _choice(building_class="barn", family="agricultural")
    walls = Counter()
    for index in range(300):
        tuned = apply_country_utility_materials(
            base,
            {},
            seed="SwedenBarnBalance",
            width_m=8.0 + index * 0.09,
            length_m=15.0 + (index % 23) * 0.13,
        )
        walls[tuned.wall_material] += 1
    assert walls["utility structural brick"] < 35
    assert walls["utility painted timber board cladding"] > 150

    explicit = apply_country_utility_materials(
        base,
        {"building:material": "brick", "building:colour": "white", "roof:material": "tile"},
        seed="SwedenExplicit",
        width_m=12.0,
        length_m=24.0,
    )
    assert explicit.wall_material == "brick"
    assert explicit.roof_material == "tile"
    assert explicit.colour_palette[0] == "white"


def test_roof_material_is_not_repainted_by_facade_palette() -> None:
    red = modeler_roof_texture_image("gabled|standing-seam metal|falun red,white", size=128)
    yellow = modeler_roof_texture_image("gabled|standing-seam metal|ochre yellow,white", size=128)
    assert red.tobytes() == yellow.tobytes()
''', encoding="utf-8", newline="\n")

print("Applied Sweden visual balance, porch suppression, and safe reuse fixes.")
