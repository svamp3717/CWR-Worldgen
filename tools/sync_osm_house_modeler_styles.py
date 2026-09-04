from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected patch context not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def sync_catalogues(repo_root: Path, source_root: Path) -> None:
    package = repo_root / "src" / "cwr_worldgen"
    for directory_name in ("house_styles", "country_styles"):
        target = package / directory_name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_root / directory_name, target)

    shutil.copy2(
        source_root / "src" / "osm_house_modeler" / "styles.py",
        package / "osm_house_modeler_styles.py",
    )

    # The upstream country catalogue is the baseline. CWR deliberately extends
    # every country/context with explicit class-specific material distributions
    # for barns, sheds, garages, warehouses, hangars and industrial buildings.
    # Reapply that data migration immediately after every upstream refresh so a
    # sync cannot silently erase the local country extensions.
    subprocess.run(
        [sys.executable, str(repo_root / "tools" / "populate_country_utility_materials.py")],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(repo_root / "tools" / "populate_country_visual_balance.py")],
        cwd=repo_root,
        check=True,
    )

    replace_once(
        repo_root / "pyproject.toml",
        '"cwr_worldgen" = ["data/road_types.json", "data/eden_gravel/*.paa", "data/eden_gravel/*.txt", "data/gravel_reference.png", "house_styles/*.json"]',
        '"cwr_worldgen" = ["data/road_types.json", "data/eden_gravel/*.paa", "data/eden_gravel/*.txt", "data/gravel_reference.png", "house_styles/*.json", "country_styles/*.json"]',
    )

    replace_once(
        package / "house_style_catalogue.py",
        '''def get_house_style_context(\n    region_identifier: str | None, settlement_context: str = "rural"\n) -> HouseStyleContext | None:\n    profile = get_region_profile(region_identifier)\n    if profile is None:\n        return None\n    contexts = profile.contexts or {}\n    return contexts.get(_context_key(settlement_context)) or contexts.get("rural")\n''',
        '''def get_country_style_context(\n    country_identifier: str | None, settlement_context: str = "rural"\n) -> HouseStyleContext | None:\n    """Resolve one modeler country profile into CWR's compatibility context."""\n\n    if not country_identifier:\n        return None\n    from .osm_house_modeler_styles import find_country_profile, load_country_profiles\n\n    profiles = load_country_profiles()\n    if not profiles:\n        return None\n    try:\n        profile = find_country_profile(profiles, country_identifier)\n    except ValueError:\n        return None\n    context_name = _context_key(settlement_context)\n    raw = profile.contexts.get(context_name) or profile.contexts.get("rural")\n    if not isinstance(raw, Mapping):\n        return None\n    try:\n        return _normalise_context(\n            raw,\n            filename=f"country_styles/{profile.iso_alpha2}_{profile.display_name}",\n            context_name=context_name,\n        )\n    except ValueError:\n        return None\n\n\ndef get_house_style_context(\n    region_identifier: str | None, settlement_context: str = "rural"\n) -> HouseStyleContext | None:\n    profile = get_region_profile(region_identifier)\n    if profile is None:\n        return get_country_style_context(region_identifier, settlement_context)\n    contexts = profile.contexts or {}\n    return contexts.get(_context_key(settlement_context)) or contexts.get("rural")\n''',
    )

    buildings = package / "procedural_buildings.py"
    replace_once(
        buildings,
        'from .building_semantics import detect_region, is_actual_church\n',
        'from .building_semantics import detect_region, is_actual_church\nfrom .osm_house_modeler_styles import choose_country, load_country_profiles\n',
    )
    replace_once(
        buildings,
        '        self.region_identifier: str | None = None\n        self.detected_house_style_identifier: str | None = None\n',
        '        self.region_identifier: str | None = None\n        self.country_style_identifier: str | None = None\n        self.detected_house_style_identifier: str | None = None\n',
    )
    replace_once(
        buildings,
        '''        profile = detect_region(\n            (projection.south, projection.west, projection.north, projection.east),\n            tag_sources,\n        )\n        self.region_identifier = profile.identifier if profile is not None else None\n        self.detected_house_style_identifier = (\n            profile.house_style_identifier if profile is not None else None\n        )\n        override_profile = house_style_preset_profile(self.house_style_preset)\n        self.house_style_identifier = (\n            override_profile.house_style_identifier\n            if override_profile is not None\n            else self.detected_house_style_identifier\n        )\n''',
        '''        bbox = (projection.south, projection.west, projection.north, projection.east)\n        profile = detect_region(bbox, tag_sources)\n        latitude = (projection.south + projection.north) * 0.5\n        longitude = (projection.west + projection.east) * 0.5\n        country_tags: dict[str, str] = {}\n        country_keys = (\n            "addr:country", "country", "country_code", "is_in:country_code",\n            "ISO3166-1:alpha2", "ISO3166-1:alpha3",\n        )\n        for tags in tag_sources:\n            for name in country_keys:\n                value = tags.get(name)\n                if value and name not in country_tags:\n                    country_tags[name] = str(value)\n        country_profiles = load_country_profiles()\n        country_profile = (\n            choose_country(country_profiles, longitude, latitude, country_tags)\n            if country_profiles else None\n        )\n        self.country_style_identifier = (\n            country_profile.identifier if country_profile is not None else None\n        )\n        self.region_identifier = (\n            country_profile.parent_region_identifier\n            if country_profile is not None\n            else profile.identifier if profile is not None else None\n        )\n        self.detected_house_style_identifier = (\n            country_profile.identifier\n            if country_profile is not None\n            else profile.house_style_identifier if profile is not None else None\n        )\n        override_profile = house_style_preset_profile(self.house_style_preset)\n        self.house_style_identifier = (\n            override_profile.house_style_identifier\n            if override_profile is not None\n            else self.detected_house_style_identifier\n        )\n''',
    )

    upgrade = package / "osm_house_modeler_upgrade.py"
    replace_once(
        upgrade,
        '''    # A balcony without a corresponding collision/opening treatment is a lie in\n    # an enterable P3D. Keep balconies on exterior-only variants until a balcony\n    # door is also represented in Geometry/Memory/animation LODs.\n    balcony_count = 0\n    if (\n        not key.interiors\n        and key.family in {"residential", "townhouse", "urban"}\n''',
        '''    # Balconies are intentionally visual secondary architecture on both closed\n    # and enterable variants. Enterable buildings do not need a dedicated balcony\n    # door; the balcony may simply be an inaccessible exterior feature.\n    balcony_count = 0\n    if (\n        key.family in {"residential", "townhouse", "urban"}\n''',
    )

    upgrade_tests = repo_root / "tests" / "test_osm_house_modeler_upgrade.py"
    test_text = upgrade_tests.read_text(encoding="utf-8")
    test_text = test_text.replace(
        '    assert plan.balcony_count == 0\n',
        '    # Enterable variants may carry visual-only balconies without a balcony door.\n',
        1,
    )
    if "test_enterable_balconies_are_allowed_without_balcony_doors" not in test_text:
        test_text += '''\n\ndef test_enterable_balconies_are_allowed_without_balcony_doors() -> None:\n    found = None\n    for variant in range(256):\n        key = pb.BuildingVariantKey(\n            "residential", "gabled", 10.0, 14.0, 6.0,\n            foundation_depth_m=0.5, regional_style="sweden_red",\n            texture_variant=variant, interiors=True,\n        )\n        plan = detail_plan_for_key(key, foundation_depth=0.5)\n        if plan.balcony_count:\n            found = plan\n            break\n    assert found is not None\n    assert found.balcony_count >= 1\n'''
    upgrade_tests.write_text(test_text, encoding="utf-8")

    (repo_root / "tests" / "test_house_style_catalogue.py").write_text(
        '''from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nfrom cwr_worldgen.house_style_catalogue import (\n    HOUSE_STYLE_PRESET_AUTO,\n    HOUSE_STYLE_PRESET_IDENTIFIERS,\n    REGION_PROFILES,\n    get_house_style_context,\n    house_style_preset_profile,\n    normalise_house_style_preset,\n    select_regional_style,\n)\nfrom cwr_worldgen.osm_house_modeler_styles import (\n    choose_country,\n    load_country_profiles,\n    load_profiles,\n)\n\n\ndef test_modeler_region_catalogue_replaces_old_compact_files() -> None:\n    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 24))\n    assert len(load_profiles()) == 23\n    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"\n    assert not sweden_file.exists()\n    assert all(profile.identifier != "sweden" for profile in load_profiles())\n\n\ndef test_country_catalogue_contains_all_modeler_profiles() -> None:\n    countries = load_country_profiles()\n    assert len(countries) == 249\n    sweden = choose_country(countries, 15.0, 62.0, {})\n    assert sweden is not None\n    assert sweden.iso_alpha2 == "SE"\n    assert sweden.parent_region_identifier == "northern_europe"\n    assert sweden.detail_level == "country-expanded-curated"\n    context = get_house_style_context(sweden.identifier, "rural")\n    assert context is not None\n    assert context.selection["default_style"]\n    assert context.roof_defaults["residential"]\n\n\ndef test_explicit_country_code_overrides_coordinate_guess() -> None:\n    countries = load_country_profiles()\n    sweden = choose_country(countries, -100.0, 40.0, {"addr:country": "SE"})\n    assert sweden is not None and sweden.iso_alpha2 == "SE"\n\n\ndef test_country_context_drives_existing_cwr_style_selector() -> None:\n    styles = {\n        select_regional_style(\n            "se_sweden", "residential", {"building": "house", "name": f"House {index}"},\n            10.0, 16.0, settlement_context="rural",\n        )\n        for index in range(80)\n    }\n    assert "swedish_wood" in styles\n\n\ndef test_house_style_preset_catalogue_exposes_23_regions_without_sweden_duplicate() -> None:\n    assert HOUSE_STYLE_PRESET_AUTO == "auto"\n    assert HOUSE_STYLE_PRESET_IDENTIFIERS == tuple(\n        profile.house_style_identifier for profile in REGION_PROFILES\n    )\n    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 23\n    assert "sweden" not in HOUSE_STYLE_PRESET_IDENTIFIERS\n    assert normalise_house_style_preset("AUTO") == "auto"\n    assert normalise_house_style_preset("east_asia") == "east_asia"\n    assert house_style_preset_profile("auto") is None\n    assert house_style_preset_profile("east_asia").display_name == "East Asia"\n''',
        encoding="utf-8",
    )

    notice = repo_root / "THIRD_PARTY_NOTICES.md"
    notice_text = notice.read_text(encoding="utf-8")
    notice_text = notice_text.replace(
        "It ports the architectural-detail behavior needed by the\nworld generator: entrance stairs for exterior-only models, porches/canopies,\nbalconies on non-enterable variants, chimneys, gutters and downspouts. Enterable\nvariants keep CWR's collision-aware openings, floors, stairs and animated doors.\n",
        "It ports the architectural-detail behavior needed by the\nworld generator: entrance stairs for exterior-only models, porches/canopies,\nbalconies (including visual-only balconies on enterable variants), chimneys,\ngutters and downspouts. Enterable variants keep CWR's collision-aware openings,\nfloors, stairs and animated entrance doors; no dedicated balcony door is required.\n\nThe 24 regional house-style profiles and the base set of 249 country profiles\noriginate from osm-house-modeler commit\n`74c8049466875dc94409493bc77bfcad56e38a8d`. CWR extends every country/context\nwith explicit barn, shed, garage, warehouse, hangar and industrial wall/roof\nmaterial pools under revision `2026-09-country-utility-materials-v1`. The\nupstream country/region style-selection engine remains pinned to that source\ncommit.\n",
    )
    notice.write_text(notice_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to an osm-house-modeler checkout")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sync_catalogues(args.repo_root.resolve(), args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
