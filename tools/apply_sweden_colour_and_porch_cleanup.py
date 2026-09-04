from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# Keep the permanent country-population tool authoritative so future upstream
# style syncs reproduce the same Sweden tuning instead of restoring the older
# grey-heavy balance.
populate_path = ROOT / "tools" / "populate_country_visual_balance.py"
populate = populate_path.read_text(encoding="utf-8")
populate = populate.replace(
    'document["detail_revision"] = "2026-09-sweden-barn-house-visuals-v3"',
    'document["detail_revision"] = "2026-09-sweden-colour-balance-v4"',
)
populate = replace_once(
    populate,
    '''        families["residential"] = (\n            [{"lt": 70, "style": "swedish_wood"}, {"lt": 95, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n            if rural else\n            [{"lt": 48, "style": "swedish_wood"}, {"lt": 90, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n        )''',
    '''        families["residential"] = (\n            [{"lt": 78, "style": "swedish_wood"}, {"lt": 96, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n            if rural else\n            [{"lt": 58, "style": "swedish_wood"}, {"lt": 92, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]\n        )''',
    label="Sweden residential style distribution",
)
populate = replace_once(
    populate,
    '''                ("painted vertical timber cladding", 68 if rural else 45),\n                ("stucco/render", 24 if rural else 45),\n                ("brick", 8 if rural else 10),''',
    '''                ("painted vertical timber cladding", 74 if rural else 52),\n                ("stucco/render", 22 if rural else 40),\n                ("brick", 4 if rural else 8),''',
    label="Sweden ordinary wall materials",
)
populate = replace_once(
    populate,
    '''                    ("falun red", 38 if rural else 22),\n                    ("ochre yellow", 24 if rural else 22),\n                    ("white", 12 if rural else 18),\n                    ("cream", 8 if rural else 12),\n                    ("grey", 6 if rural else 12),\n                    ("dark green", 5),\n                    ("natural timber", 7 if rural else 9),''',
    '''                    ("falun red", 48 if rural else 30),\n                    ("ochre yellow", 30 if rural else 28),\n                    ("white", 8 if rural else 16),\n                    ("cream", 5 if rural else 10),\n                    ("grey", 2 if rural else 5),\n                    ("dark green", 3 if rural else 4),\n                    ("natural timber", 4 if rural else 7),''',
    label="Sweden timber colours",
)
populate = replace_once(
    populate,
    '''                    ("cream", 38 if rural else 35),\n                    ("white", 32 if rural else 35),\n                    ("grey", 20 if rural else 22),\n                    ("ochre yellow", 8 if rural else 6),\n                    ("falun red", 2),''',
    '''                    ("cream", 42 if rural else 35),\n                    ("white", 35 if rural else 36),\n                    ("grey", 8 if rural else 10),\n                    ("ochre yellow", 13 if rural else 17),\n                    ("falun red", 2),''',
    label="Sweden stucco colours",
)
# Explicit ordinary fallback colours cover brick/other ordinary materials as
# well as any future Swedish wall type without a material-specific pool.
needle = '        overrides = materials.get("building_class_overrides") or {}\n'
replacement = '''        materials["facade_colour_distribution"] = _weighted(\n            "colour",\n            [\n                ("falun red", 38 if rural else 24),\n                ("ochre yellow", 28 if rural else 24),\n                ("white", 12 if rural else 20),\n                ("cream", 8 if rural else 14),\n                ("natural timber", 5),\n                ("dark green", 4),\n                ("grey", 3 if rural else 6),\n                ("black", 2 if rural else 3),\n            ],\n        )\n\n        overrides = materials.get("building_class_overrides") or {}\n'''
populate = replace_once(populate, needle, replacement, label="Sweden fallback colours")
populate = replace_once(
    populate,
    '''                ("falun red", 72 if rural else 62),\n                ("ochre yellow", 10 if rural else 12),\n                ("natural timber", 8 if rural else 8),\n                ("dark green", 4 if rural else 5),\n                ("grey", 3 if rural else 5),\n                ("white", 2 if rural else 5),\n                ("cream", 1 if rural else 3),''',
    '''                ("falun red", 78 if rural else 68),\n                ("ochre yellow", 8 if rural else 10),\n                ("natural timber", 7 if rural else 8),\n                ("dark green", 3 if rural else 4),\n                ("grey", 1 if rural else 2),\n                ("white", 2 if rural else 5),\n                ("cream", 1 if rural else 3),''',
    label="Sweden barn colours",
)
populate = replace_once(
    populate,
    '''                ("falun red", 48 if rural else 34),\n                ("ochre yellow", 16),\n                ("natural timber", 14),\n                ("grey", 8 if rural else 12),\n                ("dark green", 6),\n                ("white", 5 if rural else 10),\n                ("cream", 3 if rural else 8),''',
    '''                ("falun red", 58 if rural else 44),\n                ("ochre yellow", 18),\n                ("natural timber", 12),\n                ("grey", 3 if rural else 6),\n                ("dark green", 4 if rural else 5),\n                ("white", 3 if rural else 9),\n                ("cream", 2 if rural else 6),''',
    label="Sweden shed colours",
)
populate_path.write_text(populate, encoding="utf-8", newline="\n")

# Re-emit the country JSON from the permanent population tool.
import subprocess
subprocess.run(["python", str(populate_path), "--repo-root", str(ROOT)], check=True)

# Porch selection has already been hard-disabled. Remove the unreachable deck,
# canopy and post geometry too so no future refactor can accidentally revive a
# porch floor without deliberately reintroducing the feature.
upgrade_path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"
upgrade = upgrade_path.read_text(encoding="utf-8")
start = upgrade.find("    if plan.porch:\n")
end = upgrade.find("    if plan.balcony_count:", start)
if start < 0 or end < 0:
    raise RuntimeError("could not locate the dead porch geometry block")
upgrade = (
    upgrade[:start]
    + "    # Porch geometry is intentionally absent in CWA. Country metadata may\n"
      "    # still describe porches, but no deck/floor, canopy or posts are emitted.\n\n"
    + upgrade[end:]
)
upgrade_path.write_text(upgrade, encoding="utf-8", newline="\n")

# Force regeneration of P3Ds that may have been cached before porches were
# disabled. Texture caches do not need invalidation for this geometry-only change.
opening_path = ROOT / "src" / "cwr_worldgen" / "opening_dimension_policy.py"
opening = opening_path.read_text(encoding="utf-8")
opening = replace_once(
    opening,
    '_BUILDING_MODEL_CACHE_V51 = "procedural-building-model-v51-modeler-opening-dimensions"',
    '_BUILDING_MODEL_CACHE_V51 = "procedural-building-model-v51-modeler-opening-dimensions"\n_BUILDING_MODEL_CACHE_V52 = "procedural-building-model-v52-no-porch-geometry"',
    label="P3D cache revision declaration",
)
opening = replace_once(
    opening,
    'if namespace in {_BUILDING_MODEL_CACHE_V49, _BUILDING_MODEL_CACHE_V50}:\n            namespace = _BUILDING_MODEL_CACHE_V51',
    'if namespace in {_BUILDING_MODEL_CACHE_V49, _BUILDING_MODEL_CACHE_V50, _BUILDING_MODEL_CACHE_V51}:\n            namespace = _BUILDING_MODEL_CACHE_V52',
    label="P3D cache revision mapping",
)
opening_path.write_text(opening, encoding="utf-8", newline="\n")

# Add focused regressions for the visual balance and complete porch removal.
test_path = ROOT / "tests" / "test_sweden_colour_and_porch_cleanup.py"
test_path.write_text(r'''from __future__ import annotations

import json
from pathlib import Path


def _weights(values, field):
    return {str(entry[field]): float(entry["weight"]) for entry in values}


def _sweden():
    path = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "country_styles" / "SE_Sweden.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_sweden_ordinary_colours_prefer_red_and_yellow_over_grey():
    document = _sweden()
    for context_name, context in document["contexts"].items():
        materials = context["architectural_details"]["materials"]
        ordinary = _weights(materials["facade_colour_distribution"], "colour")
        timber = _weights(
            materials["wall_material_colour_distributions"]["painted vertical timber cladding"],
            "colour",
        )
        assert ordinary["falun red"] + ordinary["ochre yellow"] >= 48
        assert ordinary["grey"] <= 6
        assert timber["falun red"] + timber["ochre yellow"] >= 58
        assert timber["grey"] <= 5


def test_sweden_rural_houses_are_mostly_painted_timber():
    materials = _sweden()["contexts"]["rural"]["architectural_details"]["materials"]
    walls = _weights(materials["common_wall_material_distribution"], "material")
    assert walls["painted vertical timber cladding"] == 74
    assert walls["stucco/render"] == 22
    assert walls["brick"] == 4


def test_sweden_barns_and_sheds_do_not_drift_grey():
    document = _sweden()
    for context_name, context in document["contexts"].items():
        overrides = context["architectural_details"]["materials"]["building_class_overrides"]
        barn = _weights(overrides["barn"]["facade_colour_distribution"], "colour")
        shed = _weights(overrides["shed"]["facade_colour_distribution"], "colour")
        assert barn["falun red"] >= (78 if context_name == "rural" else 68)
        assert barn["grey"] <= 2
        assert shed["falun red"] >= (58 if context_name == "rural" else 44)
        assert shed["grey"] <= 6


def test_porch_floor_canopy_and_posts_are_not_in_the_generator():
    root = Path(__file__).parents[1]
    source = (root / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py").read_text(encoding="utf-8")
    assert "if plan.porch:" not in source
    assert "porch_canopy_texture" not in source
    assert "Porch geometry is intentionally absent" in source


def test_no_porch_p3ds_use_a_new_cache_revision():
    root = Path(__file__).parents[1]
    source = (root / "src" / "cwr_worldgen" / "opening_dimension_policy.py").read_text(encoding="utf-8")
    assert "procedural-building-model-v52-no-porch-geometry" in source
''', encoding="utf-8", newline="\n")

print("Applied Sweden colour rebalance and complete porch cleanup")
