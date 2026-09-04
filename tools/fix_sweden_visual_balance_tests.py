from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}; got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


# The country-material adapter runs after the core style resolver. When OSM has
# explicit material tags, the StyleChoice already contains those mapped values;
# this layer's job is to avoid replacing them with country defaults.
sweden = ROOT / "tests/test_sweden_visual_balance.py"
replace_once(
    sweden,
    '''    explicit = apply_country_utility_materials(\n        base,\n        {"building:material": "brick", "building:colour": "white", "roof:material": "tile"},\n        seed="SwedenExplicit",\n        width_m=12.0,\n        length_m=24.0,\n    )\n    assert explicit.wall_material == "brick"\n    assert explicit.roof_material == "tile"\n    assert explicit.colour_palette[0] == "white"\n''',
    '''    resolved = __import__("dataclasses").replace(\n        base, wall_material="brick", roof_material="tile"\n    )\n    explicit = apply_country_utility_materials(\n        resolved,\n        {"building:material": "brick", "building:colour": "white", "roof:material": "tile"},\n        seed="SwedenExplicit",\n        width_m=12.0,\n        length_m=24.0,\n    )\n    assert explicit.wall_material == "brick"\n    assert explicit.roof_material == "tile"\n    assert explicit.colour_palette[0] == "white"\n''',
)

# Residential buildings are now intentionally allowed to receive a weighted
# country facade colour/material. The invariant here is simply that they never
# receive a utility-only material token.
utility = ROOT / "tests/test_country_utility_materials.py"
replace_once(
    utility,
    '''    assert selected == choice\n\n\ndef test_explicit_osm_material_tags_override_country_utility_defaults() -> None:\n''',
    '''    assert selected.building_class == choice.building_class\n    assert not selected.wall_material.startswith("utility ")\n    assert not selected.roof_material.startswith("utility ")\n    assert set(selected.colour_palette) == set(choice.colour_palette)\n\n\ndef test_explicit_osm_material_tags_override_country_utility_defaults() -> None:\n''',
)

print("Adjusted Sweden visual-balance tests to the resolved-style contract.")
