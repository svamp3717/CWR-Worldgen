from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPGRADE = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"
OPENING = ROOT / "src" / "cwr_worldgen" / "opening_dimension_policy.py"
TEST = ROOT / "tests" / "test_single_chimney_policy.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: expected source block not found")
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected source block exactly once, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_upgrade() -> None:
    text = UPGRADE.read_text(encoding="utf-8")
    old = '''    chimney_count = int(\n        key.family in {"residential", "townhouse"}\n        and key.roof_style not in {"flat", "dome", "onion"}\n        and _chance(key, "chimney", chimney_p)\n    )\n    if (\n        chimney_count\n        and key.width_m >= 12.0\n        and _chance(key, "chimney-second", 0.18)\n    ):\n        chimney_count = 2\n'''
    new = '''    # CWA procedural buildings use at most one chimney. Multiple stacks look\n    # exaggerated at the game's visual scale and are not worth multiplying\n    # secondary roof geometry. Country/style data may still control whether a\n    # chimney exists, but never the count above one.\n    chimney_count = int(\n        key.family in {"residential", "townhouse"}\n        and key.roof_style not in {"flat", "dome", "onion"}\n        and _chance(key, "chimney", chimney_p)\n    )\n'''
    text = replace_once(text, old, new, "remove second-chimney lottery")

    old = '''    if plan.chimney_count:\n        chimney_spec = detail_spec.get("chimneys") or {}\n        chimney_texture = feature_material(chimney_spec.get("material"), "masonry", foundation_texture or roof_texture or detail_texture)\n        chimney_width = max(0.20, float(chimney_spec.get("width_m", 0.48) or 0.48))\n        chimney_depth = max(0.20, float(chimney_spec.get("depth_m", 0.40) or 0.40))\n        chimney_height = max(0.35, float(chimney_spec.get("height_m", 1.15) or 1.15))\n        for index in range(plan.chimney_count):\n            offset = (\n                index - (plan.chimney_count - 1) * 0.5\n            ) * min(2.4, key.length_m * 0.22)\n'''
    new = '''    # Treat one chimney as a hard geometry invariant as well as a style-policy\n    # rule. This protects generated P3Ds even if a future caller constructs an\n    # ArchitecturalDetailPlan with an out-of-range chimney_count.\n    chimney_count = min(1, max(0, int(plan.chimney_count)))\n    if chimney_count:\n        chimney_spec = detail_spec.get("chimneys") or {}\n        chimney_texture = feature_material(chimney_spec.get("material"), "masonry", foundation_texture or roof_texture or detail_texture)\n        chimney_width = max(0.20, float(chimney_spec.get("width_m", 0.48) or 0.48))\n        chimney_depth = max(0.20, float(chimney_spec.get("depth_m", 0.40) or 0.40))\n        chimney_height = max(0.35, float(chimney_spec.get("height_m", 1.15) or 1.15))\n        for index in range(chimney_count):\n            offset = (\n                index - (chimney_count - 1) * 0.5\n            ) * min(2.4, key.length_m * 0.22)\n'''
    text = replace_once(text, old, new, "hard-cap chimney geometry")
    UPGRADE.write_text(text, encoding="utf-8", newline="\n")


def patch_cache() -> None:
    text = OPENING.read_text(encoding="utf-8")
    old = '''_BUILDING_MODEL_CACHE_V51 = "procedural-building-model-v51-modeler-opening-dimensions"\n_BUILDING_MODEL_CACHE_V52 = "procedural-building-model-v52-no-porch-geometry"\n'''
    new = '''_BUILDING_MODEL_CACHE_V51 = "procedural-building-model-v51-modeler-opening-dimensions"\n_BUILDING_MODEL_CACHE_V52 = "procedural-building-model-v52-no-porch-geometry"\n_BUILDING_MODEL_CACHE_V53 = "procedural-building-model-v53-single-chimney"\n'''
    text = replace_once(text, old, new, "add v53 cache namespace")
    old = '''        if namespace in {_BUILDING_MODEL_CACHE_V49, _BUILDING_MODEL_CACHE_V50, _BUILDING_MODEL_CACHE_V51}:\n            namespace = _BUILDING_MODEL_CACHE_V52\n'''
    new = '''        if namespace in {\n            _BUILDING_MODEL_CACHE_V49,\n            _BUILDING_MODEL_CACHE_V50,\n            _BUILDING_MODEL_CACHE_V51,\n            _BUILDING_MODEL_CACHE_V52,\n        }:\n            namespace = _BUILDING_MODEL_CACHE_V53\n'''
    text = replace_once(text, old, new, "route building cache to v53")
    OPENING.write_text(text, encoding="utf-8", newline="\n")


def write_test() -> None:
    TEST.write_text('''from __future__ import annotations\n\nfrom dataclasses import replace\n\nfrom cwr_worldgen import osm_house_modeler_upgrade as upgrade\nfrom cwr_worldgen.procedural_buildings import BuildingVariantKey\n\n\ndef _key(*, width_m: float = 16.0, texture_variant: int = 0) -> BuildingVariantKey:\n    return BuildingVariantKey(\n        family="residential",\n        roof_style="gabled",\n        width_m=width_m,\n        length_m=18.0,\n        height_m=7.0,\n        regional_style="swedish_wood",\n        texture_variant=texture_variant,\n    )\n\n\ndef test_style_planner_never_requests_multiple_chimneys() -> None:\n    # Exercise enough deterministic seeds and large widths to cover the old\n    # width>=12 m / 18% second-chimney branch.\n    counts = {\n        upgrade.detail_plan_for_key(_key(width_m=width, texture_variant=variant)).chimney_count\n        for width in (6.0, 12.0, 18.0, 30.0)\n        for variant in range(256)\n    }\n    assert counts <= {0, 1}\n    assert 1 in counts\n\n\ndef test_non_house_families_still_do_not_gain_chimneys() -> None:\n    key = _key()\n    assert upgrade.detail_plan_for_key(replace(key, family="industrial")).chimney_count == 0\n    assert upgrade.detail_plan_for_key(replace(key, family="agricultural")).chimney_count == 0\n\n\ndef test_geometry_source_has_hard_single_chimney_clamp() -> None:\n    source = (upgrade.__file__ and open(upgrade.__file__, encoding="utf-8").read())\n    assert 'chimney_count = min(1, max(0, int(plan.chimney_count)))' in source\n    assert 'chimney-second' not in source\n    assert 'chimney_count = 2' not in source\n''', encoding="utf-8", newline="\n")


def main() -> int:
    patch_upgrade()
    patch_cache()
    write_test()
    print("Applied global single-chimney policy and P3D cache revision v53")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
