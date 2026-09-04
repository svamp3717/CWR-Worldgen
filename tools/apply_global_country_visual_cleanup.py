from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match in {path}: got {count} for {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


# Permanent country visual-balance migration. It is intentionally kept in tools/
# because sync_osm_house_modeler_styles.py must reapply it after upstream refreshes.
populator = ROOT / "tools" / "populate_country_visual_balance.py"
populator.write_text(r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

GLOBAL_VISUAL_BALANCE_REVISION = "2026-09-global-country-visual-balance-v1"


def _dedupe(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _distribution(values: object, field: str) -> list[dict[str, object]]:
    # Do not invent national frequency statistics. The profile's curated list is
    # authoritative; the generic baseline simply gives each listed option an
    # explicit equal chance. Countries can override this with hand-tuned weights
    # (Sweden already does).
    return [{field: value, "weight": 100} for value in _dedupe(values)]


def populate(repo_root: Path) -> tuple[int, int]:
    package = repo_root / "src" / "cwr_worldgen"
    country_dir = package / "country_styles"
    paths = sorted(country_dir.glob("*.json"))
    if len(paths) != 249:
        raise RuntimeError(f"expected 249 country profiles, found {len(paths)}")

    context_count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("identifier") == "se_sweden":
            document["parent_region_identifier"] = "northern_europe"
            document["parent_region_name"] = "Northern Europe"
            provenance = document.setdefault("data_provenance", {})
            provenance["architectural_basis"] = (
                "curated national tuning over the Northern Europe regional baseline; "
                "Sweden itself is defined only in country_styles"
            )

        contexts = document.get("contexts") or {}
        if not isinstance(contexts, dict):
            raise RuntimeError(f"{path.name}: contexts must be an object")
        for context_name, context in contexts.items():
            if not isinstance(context, dict):
                raise RuntimeError(f"{path.name}: context {context_name!r} must be an object")
            details = context.get("architectural_details") or {}
            materials = details.get("materials") or {}
            if not isinstance(materials, dict):
                raise RuntimeError(f"{path.name}: {context_name} materials must be an object")

            wall_values = _dedupe(materials.get("common_wall_materials"))
            if wall_values and not materials.get("common_wall_material_distribution"):
                materials["common_wall_material_distribution"] = _distribution(
                    wall_values, "material"
                )

            colour_values = _dedupe(materials.get("typical_colour_palette"))
            if colour_values and not materials.get("facade_colour_distribution"):
                materials["facade_colour_distribution"] = _distribution(
                    colour_values, "colour"
                )

            materials["global_visual_balance_revision"] = GLOBAL_VISUAL_BALANCE_REVISION
            materials["global_visual_balance_provenance"] = (
                "Explicit ordinary-wall and facade-colour distributions derived from "
                "this country/context's curated lists. Hand-tuned distributions take "
                "precedence; otherwise listed options use an equal-weight baseline. "
                "Explicit OSM building:material/building:colour tags remain authoritative."
            )
            context_count += 1

        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # Sweden is a country profile, not a special 24th regional style. Keeping both
    # caused two authorities for the same country and made sync behavior brittle.
    sweden_region = package / "house_styles" / "24_sweden.json"
    sweden_region.unlink(missing_ok=True)
    return len(paths), context_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    countries, contexts = populate(args.repo_root.resolve())
    print(f"Applied global visual balance to {countries} countries / {contexts} contexts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8", newline="\n")

# Door textures are always solid in CWR, even if upstream selected a glazed door.
bridge = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_texture_bridge.py"
replace_once(
    bridge,
    '''        _upstream._render_door(spec, family, outbuilding_kind, rng, native), int(size)\n''',
    '''        _upstream._render_door(\n            spec, family, outbuilding_kind, rng, native, no_glass=True\n        ),\n        int(size),\n''',
)

# Force regeneration of fronts/doors whose style token previously allowed glazing.
replace_once(
    ROOT / "src" / "cwr_worldgen" / "opening_dimension_policy.py",
    '        metadata["texture_renderer_revision"] = 4\n',
    '        metadata["texture_renderer_revision"] = 5\n',
)

# The country material policy is global now, not merely a utility-building patch.
policy = ROOT / "src" / "cwr_worldgen" / "country_utility_material_policy.py"
replace_once(
    policy,
    '''"""Consume explicit per-country utility material pools and render their finishes.\n\nSelection in this module is intentionally data-driven. It does not invent a barn\nor warehouse material at runtime: the selected country/context JSON must contain\n``architectural_details.materials.building_class_overrides``. Explicit OSM\n``building:material`` and ``roof:material`` tags remain authoritative.\n"""\n''',
    '''"""Consume explicit per-country material, colour and utility-building pools.\n\nSelection in this module is intentionally data-driven. Ordinary walls/facade\ncolours and utility classes come from the selected country/context JSON. Explicit\nOSM ``building:material``, ``building:colour`` and ``roof:material`` tags remain\nauthoritative.\n"""\n''',
)

# Regional catalogue documentation should no longer claim Sweden is one of its
# bundled presets.
replace_once(
    ROOT / "src" / "cwr_worldgen" / "house_style_catalogue.py",
    '''    ``house_style_identifier`` when it needs the precise 24-region catalogue.\n''',
    '''    ``house_style_identifier`` when it needs the bundled regional catalogue.\n''',
)

# Make future upstream style syncs reapply both local country extensions and the
# Sweden-country-only rule.
sync = ROOT / "tools" / "sync_osm_house_modeler_styles.py"
replace_once(
    sync,
    '''    subprocess.run(\n        [sys.executable, str(repo_root / "tools" / "populate_country_utility_materials.py")],\n        cwd=repo_root,\n        check=True,\n    )\n\n''',
    '''    subprocess.run(\n        [sys.executable, str(repo_root / "tools" / "populate_country_utility_materials.py")],\n        cwd=repo_root,\n        check=True,\n    )\n    subprocess.run(\n        [sys.executable, str(repo_root / "tools" / "populate_country_visual_balance.py")],\n        cwd=repo_root,\n        check=True,\n    )\n\n''',
)
sync_text = sync.read_text(encoding="utf-8")
sync_text = sync_text.replace(
    '    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 25))\\n    assert len(load_profiles()) == 24\\n    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"\\n    document = json.loads(sweden_file.read_text(encoding="utf-8"))\\n    assert document.get("detail_revision")\\n    assert "architectural_details" in document["contexts"]["rural"]\\n    assert "exterior_details" in document["contexts"]["rural"]["architectural_details"]\\n',
    '    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 24))\\n    assert len(load_profiles()) == 23\\n    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"\\n    assert not sweden_file.exists()\\n    assert all(profile.identifier != "sweden" for profile in load_profiles())\\n',
)
sync_text = sync_text.replace(
    '    assert sweden.parent_region_identifier == "sweden"\\n',
    '    assert sweden.parent_region_identifier == "northern_europe"\\n',
)
sync_text = sync_text.replace(
    'def test_house_style_preset_catalogue_still_exposes_24_regions() -> None:\\n',
    'def test_house_style_preset_catalogue_exposes_23_regions_without_sweden_duplicate() -> None:\\n',
)
sync_text = sync_text.replace(
    '    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 24\\n',
    '    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 23\\n    assert "sweden" not in HOUSE_STYLE_PRESET_IDENTIFIERS\\n',
)
sync.write_text(sync_text, encoding="utf-8", newline="\n")

# Update the current house-style tests to the same contract.
house_tests = ROOT / "tests" / "test_house_style_catalogue.py"
text = house_tests.read_text(encoding="utf-8")
text = text.replace("import json\n", "")
old_block = '''def test_modeler_region_catalogue_replaces_old_compact_files() -> None:\n    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 25))\n    assert len(load_profiles()) == 24\n    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"\n    document = json.loads(sweden_file.read_text(encoding="utf-8"))\n    assert document.get("detail_revision")\n    assert "architectural_details" in document["contexts"]["rural"]\n    assert "exterior_details" in document["contexts"]["rural"]["architectural_details"]\n'''
new_block = '''def test_modeler_region_catalogue_excludes_sweden_country_duplicate() -> None:\n    assert [profile.map_region_number for profile in REGION_PROFILES] == list(range(1, 24))\n    assert len(load_profiles()) == 23\n    sweden_file = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json"\n    assert not sweden_file.exists()\n    assert all(profile.identifier != "sweden" for profile in load_profiles())\n'''
if old_block not in text:
    raise RuntimeError("house-style Sweden region test block changed")
text = text.replace(old_block, new_block, 1)
text = text.replace(
    '    assert sweden.parent_region_identifier == "sweden"\n',
    '    assert sweden.parent_region_identifier == "northern_europe"\n',
    1,
)
text = text.replace(
    'def test_house_style_preset_catalogue_still_exposes_24_regions() -> None:\n',
    'def test_house_style_preset_catalogue_exposes_23_regions_without_sweden_duplicate() -> None:\n',
    1,
)
text = text.replace(
    '    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 24\n',
    '    assert len(HOUSE_STYLE_PRESET_IDENTIFIERS) == 23\n    assert "sweden" not in HOUSE_STYLE_PRESET_IDENTIFIERS\n',
    1,
)
house_tests.write_text(text, encoding="utf-8", newline="\n")

# Door regression test: verify the bridge forces the upstream renderer into its
# no-glass mode for a door type that explicitly says "glazed".
texture_tests = ROOT / "tests" / "test_osm_house_modeler_texture_bridge.py"
text = texture_tests.read_text(encoding="utf-8")
text = text.replace(
    '    _seed,\n    _wall_material_image,\n',
    '    _door_image_cached,\n    _seed,\n    _wall_material_image,\n',
    1,
)
text = text.replace(
    '    modeler_front_texture_image,\n',
    '    modeler_door_texture_image,\n    modeler_front_texture_image,\n',
    1,
)
if "test_modeler_doors_force_solid_no_glass" not in text:
    text += r'''


def test_modeler_doors_force_solid_no_glass(monkeypatch) -> None:
    token = _token("swedish_wood", "painted timber", "cream", door_material="timber")
    seen: list[bool] = []
    original = upstream_textures._render_door

    def captured(spec, family, outbuilding_kind, rng, size, *, no_glass=False):
        seen.append(bool(no_glass))
        return original(
            spec, family, outbuilding_kind, rng, size, no_glass=no_glass
        )

    monkeypatch.setattr(upstream_textures, "_render_door", captured)
    _door_image_cached.cache_clear()
    modeler_door_texture_image(
        128,
        family="residential",
        regional_style=token,
        texture_variant=11,
    )
    assert seen == [True]
'''
texture_tests.write_text(text, encoding="utf-8", newline="\n")

# All-country regression coverage plus explicit Sweden country-only resolution.
global_tests = ROOT / "tests" / "test_global_country_visual_balance.py"
global_tests.write_text(r'''from __future__ import annotations

import json
from pathlib import Path

from cwr_worldgen.country_utility_material_policy import apply_country_utility_materials
from cwr_worldgen.osm_house_modeler_styles import (
    StyleChoice,
    choose_style,
    load_country_profiles,
    load_profiles,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTRY_DIR = ROOT / "src" / "cwr_worldgen" / "country_styles"


def _values(entries, key: str) -> set[str]:
    return {str(entry[key]).casefold() for entry in entries}


def test_all_249_country_contexts_have_explicit_ordinary_visual_distributions() -> None:
    paths = sorted(COUNTRY_DIR.glob("*.json"))
    assert len(paths) == 249
    context_count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for context in document["contexts"].values():
            materials = context["architectural_details"]["materials"]
            walls = [str(value) for value in materials.get("common_wall_materials", []) if str(value)]
            wall_distribution = materials.get("common_wall_material_distribution", [])
            if walls:
                assert wall_distribution, path.name
                assert _values(wall_distribution, "material") == {value.casefold() for value in walls}
                assert all(float(entry["weight"]) > 0 for entry in wall_distribution)

            colours = [str(value) for value in materials.get("typical_colour_palette", []) if str(value)]
            colour_distribution = materials.get("facade_colour_distribution", [])
            if colours:
                assert colour_distribution, path.name
                assert _values(colour_distribution, "colour") == {value.casefold() for value in colours}
                assert all(float(entry["weight"]) > 0 for entry in colour_distribution)
                if len(colour_distribution) >= 3:
                    total = sum(float(entry["weight"]) for entry in colour_distribution)
                    assert max(float(entry["weight"]) for entry in colour_distribution) / total <= 0.55

            assert materials["global_visual_balance_revision"] == "2026-09-global-country-visual-balance-v1"
            context_count += 1
    assert context_count == 498


def _choice_for(country_identifier: str, context: str = "rural") -> StyleChoice:
    profile = next(p for p in load_country_profiles() if p.identifier == country_identifier)
    raw = profile.contexts[context]
    materials = raw["architectural_details"]["materials"]
    walls = materials.get("common_wall_materials") or ["stucco/render"]
    roofs = materials.get("common_roof_materials") or ["tile"]
    palette = tuple(str(value) for value in materials.get("typical_colour_palette") or ["cream"])
    return StyleChoice(
        region_identifier=profile.parent_region_identifier,
        region_name=profile.parent_region_identifier,
        facade_style=str(raw["selection"].get("default_style", "default")),
        roof_style=str(raw["roof_defaults"].get("residential", "gabled")),
        context=context,
        family="residential",
        building_class="residential",
        country_code=profile.iso_alpha2,
        country_name=profile.display_name,
        country_profile_identifier=profile.identifier,
        wall_material=str(walls[0]),
        roof_material=str(roofs[0]),
        colour_palette=palette,
    )


def test_representative_countries_are_not_pinned_to_first_colour_or_material() -> None:
    for country_identifier in ("se_sweden", "ke_kenya", "jp_japan"):
        base = _choice_for(country_identifier)
        colours: set[str] = set()
        walls: set[str] = set()
        for index in range(180):
            tuned = apply_country_utility_materials(
                base,
                {},
                seed=f"global-balance-{country_identifier}",
                width_m=6.0 + index * 0.11,
                length_m=8.0 + (index % 29) * 0.13,
            )
            colours.add(tuned.colour_palette[0])
            walls.add(tuned.wall_material)
        assert len(colours) >= 2, country_identifier
        assert len(walls) >= 2, country_identifier


def test_sweden_is_country_style_only_and_uses_northern_europe_parent() -> None:
    assert not (ROOT / "src" / "cwr_worldgen" / "house_styles" / "24_sweden.json").exists()
    regions = load_profiles()
    assert len(regions) == 23
    assert all(profile.identifier != "sweden" for profile in regions)

    sweden = next(profile for profile in load_country_profiles() if profile.identifier == "se_sweden")
    assert sweden.parent_region_identifier == "northern_europe"

    choice = choose_style(
        regions,
        15.0,
        62.0,
        {"building": "house", "addr:country": "SE"},
        12345,
        width_m=10.0,
        length_m=14.0,
        seed="sweden-country-only",
    )
    assert choice.country_profile_identifier == "se_sweden"
    assert choice.region_identifier == "northern_europe"
''', encoding="utf-8", newline="\n")

# Third-party notice: upstream still has 24 regions, CWR deliberately bundles 23
# because Sweden is now solely a country profile. Also document global balancing
# and porch suppression accurately.
notice = ROOT / "THIRD_PARTY_NOTICES.md"
notice.write_text('''# Third-party notices\n\n## OSM House Modeler\n\nThis branch adapts procedural building-detail concepts and geometry behavior from\n[`svamp3717/osm-house-modeler`](https://github.com/svamp3717/osm-house-modeler),\nversion 0.13.6.\n\nThe upstream project declares its license as MIT in `pyproject.toml`. The adapted\nCWR implementation keeps CWR's existing GPL-3.0-or-later package licensing and\npreserves the upstream source/version reference in\n`src/cwr_worldgen/osm_house_modeler_upgrade.py`.\n\nThe integration intentionally retains CWR's OFP/CWA-specific P3D/MLOD, Geometry,\nRoadway, Memory and Paths LOD implementation instead of copying the standalone\nOBJ/viewer pipeline. It ports entrance stairs, balconies, chimneys, gutters and\ndownspouts. Porch/canopy metadata is retained from the source profiles, but CWR\nintentionally suppresses porch geometry because it does not read well in-game.\nEnterable variants keep CWR's collision-aware openings, floors, stairs and animated\nentrance doors; no dedicated balcony door is required. Generated CWR door textures\nare intentionally solid and do not render glazed/window panels.\n\nThe upstream source commit `74c8049466875dc94409493bc77bfcad56e38a8d` provides\n24 regional house-style profiles and the base set of 249 country profiles. CWR\ndeliberately bundles 23 regional profiles: upstream `24_sweden.json` is excluded so\nSweden is defined solely by `country_styles/SE_Sweden.json`, layered on the\n`northern_europe` regional baseline like other countries.\n\nCWR extends every country/context with explicit barn, shed, garage, warehouse,\nhangar and industrial wall/roof material pools under revision\n`2026-09-country-utility-materials-v1`. It also stores explicit ordinary wall\nmaterial and facade-colour distributions under revision\n`2026-09-global-country-visual-balance-v1`; hand-tuned country weights take\nprecedence, while otherwise the country's own curated list receives a balanced\nequal-weight baseline. Explicit OSM `building:material`, `building:colour` and\n`roof:material` tags remain authoritative.\n\nThe upstream country/region style-selection engine and procedural material texture\ngenerator remain pinned to the same source commit. The vendored texture\nimplementation is stored as `src/cwr_worldgen/osm_house_modeler_textures.py`; CWR\nwraps its wall, roof, foundation, window, door and secondary-material pixels in a\nCWA/PAA bridge while retaining the existing MLOD asset pipeline.\n''', encoding="utf-8", newline="\n")

# Apply the permanent population tool to the current checked-out catalogues.
subprocess.run([sys.executable, str(populator), "--repo-root", str(ROOT)], check=True)
print("Applied global country visual cleanup")
