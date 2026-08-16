from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cwr_worldgen.gui import (
    WIZARD_STEPS,
    FROZEN_CLI_MARKER,
    APPEARANCE_PRESETS,
    RECOMMENDED_APPEARANCE_PRESET,
    RESISTANCE_APPEARANCE_PRESET,
    PINE_NOGOVA_APPEARANCE_PRESET,
    NOGOVA_FOREST_BLOCK_MODEL,
    NOGOVA_FOREST_STEEP_MODEL,
    NOGOVA_PINE_FOREST_BLOCK_MODEL,
    NOGOVA_PINE_FOREST_STEEP_MODEL,
    NOGOVA_LEAF_SINGLE_TREE_MODEL,
    NOGOVA_PINE_SINGLE_TREE_MODEL,
    WorldgenGui,
    application_base_dir,
    build_fetch_command,
    cli_command_prefix,
    build_gui_osm_asset_mapping_document,
    build_milestone9_command,
    build_wizard_pipeline_commands,
    default_gui_path,
    default_gui_values,
    defaults_with_recent_source,
    existing_source_preview_path,
    format_wizard_world_size,
    increment_trailing_number,
    main as gui_main,
    load_gui_state,
    quote_command,
    resolve_gui_path,
    save_gui_state,
    slugify_world_name,
    suggested_world_values,
    validate_wizard_step,
    write_gui_osm_asset_mapping,
)


class FrozenGuiDispatchTests(unittest.TestCase):
    def test_frozen_cli_prefix_reenters_executable_without_python_dash_m(self) -> None:
        with (
            patch("cwr_worldgen.gui.sys.executable", r"C:\\Worldgen\\cwr-worldgen-gui.exe"),
            patch("cwr_worldgen.gui.sys.frozen", True, create=True),
        ):
            self.assertEqual(
                cli_command_prefix(),
                [r"C:\\Worldgen\\cwr-worldgen-gui.exe", FROZEN_CLI_MARKER],
            )
            command = build_milestone9_command({
                "source_dir": "source",
                "output": "build",
                "name": "map",
                "display_name": "Map",
            })
            self.assertEqual(command[:3], [r"C:\\Worldgen\\cwr-worldgen-gui.exe", FROZEN_CLI_MARKER, "milestone9"])
            self.assertNotIn("-m", command[:4])

    def test_explicit_python_launcher_stays_compatible_with_source_mode(self) -> None:
        self.assertEqual(cli_command_prefix("python"), ["python", "-m", "cwr_worldgen"])

    def test_frozen_defaults_and_relative_paths_live_beside_executable(self) -> None:
        with TemporaryDirectory() as temporary:
            app_dir = Path(temporary).resolve()
            executable = app_dir / "CWR-Worldgen.exe"
            with (
                patch("cwr_worldgen.gui.sys.executable", str(executable)),
                patch("cwr_worldgen.gui.sys.frozen", True, create=True),
            ):
                self.assertEqual(application_base_dir(), app_dir)
                self.assertEqual(resolve_gui_path("relative-output"), app_dir / "relative-output")
                self.assertEqual(default_gui_path(Path("build") / "map"), str(app_dir / "build" / "map"))
                defaults = default_gui_values()
                self.assertEqual(defaults["output"], str(app_dir / "build" / "my_world"))
                self.assertEqual(defaults["source_dir"], str(app_dir / "source-data" / "my_world"))
                suggested = suggested_world_values(
                    "Stockholm Test", source_mode="new", source_dir=str(app_dir / "unused")
                )
                self.assertEqual(suggested["output"], str(app_dir / "build" / "stockholm_test"))
                self.assertEqual(suggested["source_dir"], str(app_dir / "source-data" / "stockholm_test"))

    def test_private_frozen_marker_dispatches_to_cli_without_opening_gui(self) -> None:
        with patch("cwr_worldgen.cli.main", return_value=23) as cli_main:
            result = gui_main([FROZEN_CLI_MARKER, "inspect-sources", "--source-dir", "bundle"])
        self.assertEqual(result, 23)
        cli_main.assert_called_once_with(["inspect-sources", "--source-dir", "bundle"])


class DeployControlStateTests(unittest.TestCase):
    class _Var:
        def __init__(self, value: bool) -> None:
            self.value = value

        def get(self) -> bool:
            return self.value

    class _Widget:
        def __init__(self) -> None:
            self.state = ""

        def configure(self, *, state: str) -> None:
            self.state = state

    def test_deploy_folder_picker_tracks_copy_checkbox(self) -> None:
        enabled = self._Var(False)
        entry = self._Widget()
        button = self._Widget()
        fake_gui = type("FakeGui", (), {})()
        fake_gui.vars = {"deploy_to_mod_folder": enabled}
        fake_gui.entry_widgets = {"deploy_mod_dir": entry}
        fake_gui.browse_buttons = {"deploy_mod_dir": button}

        WorldgenGui._update_deploy_controls(fake_gui)
        self.assertEqual(entry.state, "disabled")
        self.assertEqual(button.state, "disabled")

        enabled.value = True
        WorldgenGui._update_deploy_controls(fake_gui)
        self.assertEqual(entry.state, "normal")
        self.assertEqual(button.state, "normal")


class GuiCommandTests(unittest.TestCase):
    def test_build_command_contains_repeatable_asset_roots_and_flags(self) -> None:
        command = build_milestone9_command({
            "source_dir": "source-data/map",
            "output": "build/map",
            "name": "cwr_map",
            "display_name": "Map",
            "profile": "cwr-ce",
            "ground_textures": "everon",
            "forest_profile": "everon",
            "pbo_backend": "poseidon",
            "poseidon_tools": "C:/tools/PoseidonTools.exe",
            "asset_roots": ["DTA_unpacked", "O.pbo"],
            "strict_assets": True,
            "forest_clusters": True,
            "forest_undergrowth": False,
            "overture_buildings": False,
            "rocky_forest_rocks_per_patch": "4",
            "rocky_forest_spread": "16",
            "lake_shore_smoothing_cells": "10",
            "lake_shore_max_slope": "7.5",
            "building_ground_clearance": "0.10",
            "building_foundation_depth": "0.50",
            "procedural_building_interiors": True,
            "advanced_args": "--max-barrier-objects 123",
        }, python="python")
        self.assertEqual(command[:4], ["python", "-m", "cwr_worldgen", "milestone9"])
        self.assertEqual(command.count("--asset-root"), 2)
        self.assertIn("--strict-assets", command)
        self.assertIn("--pbo-backend", command)
        self.assertIn("poseidon", command)
        self.assertIn("--poseidon-tools", command)
        self.assertIn("--no-forest-undergrowth", command)
        self.assertIn("--no-overture-buildings", command)
        self.assertIn("--rocky-forest-rocks-per-patch", command)
        self.assertIn("--rocky-forest-spread", command)
        self.assertIn("--lake-shore-smoothing-cells", command)
        self.assertIn("--lake-shore-max-slope", command)
        self.assertIn("--building-ground-clearance", command)
        self.assertIn("--building-foundation-depth", command)
        self.assertIn("--procedural-building-interiors", command)
        self.assertNotIn("--no-forest-clusters", command)
        self.assertNotIn("--verify-regeneration", command)
        self.assertEqual(command[-2:], ["--max-barrier-objects", "123"])

    def test_build_command_accepts_desert_ground_texture_profile(self) -> None:
        values = default_gui_values()
        values["ground_textures"] = "desert"
        command = build_milestone9_command(values, python="python")
        self.assertIn("--ground-textures", command)
        self.assertEqual(command[command.index("--ground-textures") + 1], "desert")

    def test_v5_appearance_presets_are_native_and_recommended_by_default(self) -> None:
        values = default_gui_values()
        self.assertEqual(values["appearance_preset"], RECOMMENDED_APPEARANCE_PRESET)
        self.assertEqual(
            APPEARANCE_PRESETS,
            (
                "Nogova textures + Everon trees (recommended)",
                "Nogova Resistance leaf forests",
                "Nogova Resistance pine forests",
                "Malden classic",
                "Everon classic",
                "Desert ground textures",
                "Generated ground textures",
                "Custom",
            ),
        )

    def test_obsolete_terrainfit_test_preset_is_removed(self) -> None:
        self.assertFalse(any("terrain-fit" in preset.casefold() for preset in APPEARANCE_PRESETS))
        self.assertFalse(any("10m" in preset.casefold() for preset in APPEARANCE_PRESETS))

    def test_recommended_preset_keeps_everon_forest_models(self) -> None:
        values = default_gui_values()
        command = build_milestone9_command(values, python="python")
        self.assertNotIn("--forest-block-model", command)
        self.assertNotIn("--forest-steep-model", command)

    def test_pine_nogova_preset_uses_conifer_polygon_models(self) -> None:
        values = default_gui_values()
        values["appearance_preset"] = PINE_NOGOVA_APPEARANCE_PRESET
        values["forest_polygon_sink_fraction"] = "0.25"
        values["advanced_args"] = "--forest-block-model custom-block.p3d"
        command = build_milestone9_command(values, python="python")
        self.assertEqual(command[-4:], [
            "--forest-block-model", NOGOVA_PINE_FOREST_BLOCK_MODEL,
            "--forest-steep-model", NOGOVA_PINE_FOREST_STEEP_MODEL,
        ])
        self.assertEqual(command.count("--forest-single-tree-model"), 1)
        self.assertEqual(command.count("--forest-polygon-sink-fraction"), 1)
        self.assertEqual(
            command[command.index("--forest-polygon-sink-fraction") + 1], "0.25"
        )
        self.assertEqual(NOGOVA_PINE_FOREST_BLOCK_MODEL, r"o\tree\les_nw_jehl_ctver_pruhozi.p3d")
        self.assertEqual(NOGOVA_PINE_FOREST_STEEP_MODEL, r"o\tree\les_nw_jehl_trojuhelnik.p3d")

    def test_resistance_preset_overrides_forest_geometry_after_advanced_args(self) -> None:
        values = default_gui_values()
        values["appearance_preset"] = RESISTANCE_APPEARANCE_PRESET
        values["forest_polygon_sink_fraction"] = "0.75"
        values["advanced_args"] = "--forest-block-model custom-block.p3d"
        command = build_milestone9_command(values, python="python")
        self.assertEqual(command[-4:], [
            "--forest-block-model", NOGOVA_FOREST_BLOCK_MODEL,
            "--forest-steep-model", NOGOVA_FOREST_STEEP_MODEL,
        ])
        self.assertEqual(command.count("--forest-single-tree-model"), 1)
        self.assertEqual(command.count("--forest-polygon-sink-fraction"), 1)
        self.assertEqual(
            command[command.index("--forest-polygon-sink-fraction") + 1], "0.75"
        )
        self.assertNotIn("--forest-block-max-burial", command)
        self.assertNotIn("--forest-steep-max-burial", command)

    def test_overture_checkbox_uses_requested_random_generation_label(self) -> None:
        gui_source = (
            Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "gui.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '("Use Overture buildings before randomly generated", "overture_buildings", True)',
            gui_source,
        )

    def test_procedural_building_interiors_are_opt_in(self) -> None:
        values = default_gui_values()
        self.assertFalse(values["procedural_building_interiors"])
        self.assertNotIn(
            "--procedural-building-interiors",
            build_milestone9_command(values, python="python"),
        )
        values["procedural_building_interiors"] = True
        self.assertIn(
            "--procedural-building-interiors",
            build_milestone9_command(values, python="python"),
        )

    def test_stock_nogova_bridges_are_default_with_procedural_opt_in(self) -> None:
        values = default_gui_values()
        self.assertTrue(values["bridges"])
        self.assertFalse(values["procedural_bridges"])
        default_command = build_milestone9_command(values, python="python")
        self.assertNotIn("--procedural-bridges", default_command)
        self.assertNotIn("--stock-bridges", default_command)
        self.assertIn("--bridge-module-length", default_command)
        self.assertEqual(default_command[default_command.index("--bridge-module-length") + 1], "30")
        values["procedural_bridges"] = True
        self.assertIn(
            "--procedural-bridges",
            build_milestone9_command(values, python="python"),
        )

    def test_custom_osm_asset_mapping_is_forwarded_and_validated(self) -> None:
        with TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "asset-map.json"
            mapping.write_text('{"schema": 1, "rules": []}', encoding="utf-8")
            values = default_gui_values()
            values["osm_asset_map"] = str(mapping)
            command = build_milestone9_command(values, python="python")
            self.assertIn("--osm-asset-map", command)
            self.assertEqual(command[command.index("--osm-asset-map") + 1], str(mapping))
            validate_wizard_step(2, values)
            values["osm_asset_map"] = str(Path(temporary) / "missing.json")
            with self.assertRaisesRegex(ValueError, "mapping JSON"):
                validate_wizard_step(2, values)

    def test_visual_osm_mapping_editor_builds_and_validates_schema_one_document(self) -> None:
        values = default_gui_values()
        values["osm_asset_mapping_enabled"] = True
        values["osm_asset_mapping_inherit_defaults"] = True
        values["osm_asset_mapping_global_models"] = r"O/Hous/Nahrobek1.p3d" + "\n" + r"O\Hous\Nahrobek2.p3d"
        values["osm_asset_mapping_global_textures"] = "Eden/tn.paa"
        values["osm_asset_mapping_rules"] = json.dumps([{
            "id": "custom-graves",
            "layers": ["sites"],
            "geometry": "polygon",
            "match": {"site": ["cemetery"]},
            "exclude": {},
            "models": [r"O\Hous\Nahrobek3.p3d"],
            "textures": [],
            "description": "GUI-created rule",
            "enabled": True,
        }])
        document = build_gui_osm_asset_mapping_document(values)
        self.assertTrue(document["inherit_defaults"])
        self.assertEqual(document["global"]["models"], [r"O\Hous\Nahrobek1.p3d", r"O\Hous\Nahrobek2.p3d"])
        self.assertEqual(document["rules"][0]["id"], "custom-graves")
        with TemporaryDirectory() as temporary:
            path = write_gui_osm_asset_mapping(Path(temporary) / "gui-map.json", values)
            self.assertTrue(path.is_file())
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], 1)

    def test_gui_state_remembers_visual_osm_mapping_between_restarts(self) -> None:
        rule_json = json.dumps([{"id": "custom", "layers": ["roads"], "geometry": "line", "match": {}, "exclude": {}, "models": [], "textures": [], "enabled": False}])
        defaults = defaults_with_recent_source(default_gui_values(), {
            "osm_asset_mapping_enabled": True,
            "osm_asset_mapping_inherit_defaults": False,
            "osm_asset_mapping_rules": rule_json,
            "osm_asset_mapping_global_models": r"O\Hous\Nahrobek1.p3d",
            "osm_asset_mapping_global_textures": r"Eden\tn.paa",
        })
        self.assertTrue(defaults["osm_asset_mapping_enabled"])
        self.assertFalse(defaults["osm_asset_mapping_inherit_defaults"])
        self.assertEqual(defaults["osm_asset_mapping_rules"], rule_json)

    def test_asset_checking_is_off_unless_explicitly_enabled(self) -> None:
        values = default_gui_values()
        command = build_milestone9_command(values, python="python")
        self.assertFalse(values["strict_assets"])
        self.assertNotIn("--strict-assets", command)
        self.assertEqual(format_wizard_world_size(values), "6.4 km (256×256 at 25 m)")

    def test_vegetation_defaults_emit_dense_interior_hillside_and_wetland_options(self) -> None:
        values = default_gui_values()
        command = build_milestone9_command(values, python="python")
        self.assertEqual(values["forest_undergrowth_max_objects"], "120000")
        self.assertEqual(values["forest_undergrowth_spacing"], "30")
        self.assertEqual(values["forest_polygon_sink_fraction"], "0.5")
        self.assertEqual(values["max_steep_hill_bush_objects"], "80000")
        self.assertEqual(values["steep_hill_bush_spacing"], "24")
        self.assertEqual(values["max_wetland_reed_objects"], "100000")
        self.assertEqual(values["wetland_reed_spacing"], "18")
        self.assertEqual(values["max_residential_infill_buildings"], "1500")
        self.assertEqual(values["residential_infill_spacing"], "68")
        self.assertTrue(values["overture_buildings"])
        self.assertIn("--forest-undergrowth-max-objects", command)
        self.assertIn("--forest-polygon-sink-fraction", command)
        self.assertIn("--max-residential-infill-buildings", command)
        self.assertIn("--residential-infill-spacing", command)
        self.assertNotIn("--no-overture-buildings", command)
        self.assertIn("--max-steep-hill-bush-objects", command)
        self.assertIn("--max-wetland-reed-objects", command)
        self.assertNotIn("--no-steep-hill-bushes", command)
        self.assertNotIn("--no-wetland-reeds", command)

    def test_gui_state_remembers_last_downloaded_source_after_restart(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "downloaded-source"
            source.mkdir()
            (source / "source.json").write_text("{}", encoding="utf-8")
            state_path = root / "gui-state.json"
            save_gui_state(state_path, {"last_downloaded_source": str(source)})
            state = load_gui_state(state_path)
            defaults = defaults_with_recent_source(default_gui_values(), state)
            self.assertEqual(defaults["source_mode"], "existing")
            self.assertEqual(defaults["source_dir"], str(source.resolve()))
            self.assertEqual(defaults["fetch_source_dir"], str(source.resolve()))

    def test_existing_source_preview_prefers_manifest_reference_image(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            reference = source / "reference"
            reference.mkdir(parents=True)
            expected = reference / "custom-map.png"
            expected.write_bytes(b"not-an-image-but-path-selection-does-not-decode")
            (reference / "opentopomap.png").write_bytes(b"fallback")
            (source / "source.json").write_text(
                json.dumps({"reference_map": {"path": "reference/custom-map.png"}}),
                encoding="utf-8",
            )
            self.assertEqual(existing_source_preview_path(source), expected.resolve())

    def test_existing_source_preview_falls_back_to_reference_folder_without_tiles(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            reference = source / "reference"
            tiles = reference / "tiles" / "12" / "1"
            tiles.mkdir(parents=True)
            (tiles / "2.png").write_bytes(b"tile")
            expected = reference / "opentopomap.png"
            expected.write_bytes(b"preview")
            (source / "source.json").write_text("{}", encoding="utf-8")
            self.assertEqual(existing_source_preview_path(source), expected)

    def test_existing_source_preview_rejects_manifest_path_outside_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            (source / "source.json").write_text(
                json.dumps({"reference_map": {"path": "../outside.png"}}),
                encoding="utf-8",
            )
            self.assertIsNone(existing_source_preview_path(source))

    def test_gui_state_suggests_next_world_name_after_restart(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "downloaded-source"
            source.mkdir()
            (source / "source.json").write_text("{}", encoding="utf-8")
            state_path = root / "gui-state.json"
            save_gui_state(
                state_path,
                {
                    "last_downloaded_source": str(source),
                    "last_world_name": "cwr_test10",
                    "last_world_display_name": "Test 10",
                    "last_world_output": str(Path("build") / "test10"),
                },
            )
            defaults = defaults_with_recent_source(
                default_gui_values(), load_gui_state(state_path)
            )
            self.assertEqual(defaults["name"], "cwr_test11")
            self.assertEqual(defaults["display_name"], "Test 11")
            self.assertEqual(defaults["output"], str(Path("build") / "test11"))
            self.assertEqual(defaults["source_mode"], "existing")
            self.assertEqual(defaults["source_dir"], str(source.resolve()))

    def test_gui_state_remembers_existing_deployment_mod_folder(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            mod_folder = root / "@MyMod"
            mod_folder.mkdir()
            state_path = root / "gui-state.json"
            save_gui_state(
                state_path,
                {"last_deploy_mod_dir": str(mod_folder), "deploy_to_mod_folder": True},
            )
            defaults = defaults_with_recent_source(
                default_gui_values(), load_gui_state(state_path)
            )
            self.assertTrue(defaults["deploy_to_mod_folder"])
            self.assertEqual(defaults["deploy_mod_dir"], str(mod_folder.resolve()))

    def test_build_command_deploys_to_existing_mod_folder(self) -> None:
        with TemporaryDirectory() as temporary:
            mod_folder = Path(temporary) / "@MyMod"
            mod_folder.mkdir()
            values = default_gui_values()
            values["deploy_to_mod_folder"] = True
            values["deploy_mod_dir"] = str(mod_folder)
            command = build_milestone9_command(values, python="python")
            self.assertIn("--deploy-mod-dir", command)
            self.assertEqual(command[command.index("--deploy-mod-dir") + 1], str(mod_folder))
            validate_wizard_step(1, values)

    def test_gui_state_ignores_missing_or_malformed_recent_sources(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "gui-state.json"
            state_path.write_text("not-json", encoding="utf-8")
            self.assertEqual(load_gui_state(state_path), {})
            defaults = defaults_with_recent_source(
                default_gui_values(),
                {"last_downloaded_source": str(root / "missing")},
            )
            self.assertEqual(defaults["source_mode"], "new")

    def test_bus_stop_signs_default_on_and_can_be_disabled(self) -> None:
        values = {
            "source_dir": "source-data/map",
            "output": "build/map",
            "name": "cwr_map",
            "display_name": "Map",
            "bus_stop_signs": False,
        }
        self.assertIn("--no-bus-stop-signs", build_milestone9_command(values, python="python"))
        values["bus_stop_signs"] = True
        command = build_milestone9_command(values, python="python")
        self.assertIn("--bus-stop-signs", command)
        self.assertNotIn("--no-bus-stop-signs", command)

    def test_legacy_bus_stop_profile_key_still_enables_signs(self) -> None:
        values = {
            "source_dir": "source-data/map",
            "output": "build/map",
            "name": "cwr_map",
            "display_name": "Map",
            "bus_stops": True,
        }
        self.assertIn("--bus-stop-signs", build_milestone9_command(values, python="python"))

    def test_invalid_world_name_is_rejected_in_gui_preflight(self) -> None:
        values = default_gui_values()
        values["name"] = "Bad World Name"
        with self.assertRaisesRegex(ValueError, "world name must be"):
            build_milestone9_command(values, python="python")

    def test_regeneration_verification_is_explicit_opt_in(self) -> None:
        command = build_milestone9_command({
            "source_dir": "source-data/map",
            "output": "build/map",
            "name": "cwr_map",
            "display_name": "Map",
            "verify_regeneration": True,
        }, python="python")
        self.assertIn("--verify-regeneration", command)

    def test_build_command_requires_core_fields(self) -> None:
        with self.assertRaises(ValueError):
            build_milestone9_command({}, python="python")

    def test_fetch_map_url(self) -> None:
        command = build_fetch_command({
            "source_dir": "source-data/map",
            "selection_mode": "map_url",
            "map_url": "https://www.opentopomap.org/#map=13/1/2",
            "cells": 256,
            "cell_size": 25,
            "refresh": True,
        }, python="python")
        self.assertIn("--map-url", command)
        self.assertIn("--refresh", command)

    def test_fetch_command_defaults_to_recommended_256_cell_area(self) -> None:
        command = build_fetch_command({
            "source_dir": "source-data/map",
            "selection_mode": "map_url",
            "map_url": "https://www.opentopomap.org/#map=13/1/2",
        }, python="python")
        self.assertEqual(command[command.index("--cells") + 1], "256")
        self.assertEqual(command[command.index("--cell-size") + 1], "25.0")


    def test_wizard_defaults_are_safe_and_nogova_first(self) -> None:
        values = default_gui_values()
        self.assertEqual(values["source_mode"], "new")
        self.assertEqual(values["selection_mode"], "bbox")
        self.assertEqual(values["ground_textures"], "nogova")
        self.assertEqual(values["forest_profile"], "everon")
        self.assertFalse(values["verify_regeneration"])
        self.assertFalse(values["strict_assets"])
        self.assertTrue(values["include_minor_roads"])
        self.assertTrue(values["bus_stop_signs"])
        self.assertEqual(values["rocky_forest_rocks_per_patch"], "3")
        self.assertEqual(values["rocky_forest_spread"], "18")
        self.assertEqual(values["lake_shore_smoothing_cells"], "8")
        self.assertEqual(values["lake_shore_max_slope"], "8")
        self.assertEqual(values["building_ground_clearance"], "0.10")
        self.assertEqual(values["building_foundation_depth"], "0.50")
        self.assertEqual(values["max_road_objects"], "1024000")
        self.assertEqual(values["max_buildings"], "100000")
        self.assertEqual(values["max_forest_objects"], "500000")
        self.assertEqual(int(str(values["fetch_cells"])), 256)
        self.assertEqual(float(str(values["fetch_cell_size"])), 25.0)

    def test_default_wizard_values_validate_through_appearance_step(self) -> None:
        values = default_gui_values()
        values["cells"] = values["fetch_cells"]
        values["cell_size"] = values["fetch_cell_size"]
        for step in (0, 1, 2):
            validate_wizard_step(step, values, [])

    def test_existing_source_mode_requires_manifest(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary:
            values = default_gui_values()
            values.update({
                "source_mode": "existing",
                "source_dir": temporary,
                "cells": values["fetch_cells"],
                "cell_size": values["fetch_cell_size"],
            })
            with self.assertRaises(ValueError):
                validate_wizard_step(0, values)
            Path(temporary, "source.json").write_text(
                json.dumps({
                    "selection": {
                        "bbox_south_west_north_east": [59.0, 17.0, 59.1, 17.1],
                        "cells": 512,
                        "cell_size_metres": 25,
                    },
                    "osm": {"raw_json": "osm/raw.json", "query": "osm/query.txt"},
                    "elevation": {"heightmap": "elevation/heightmap.png"},
                }),
                encoding="utf-8",
            )
            validate_wizard_step(0, values)

    def test_existing_source_review_uses_manifest_grid_instead_of_fetch_defaults(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary:
            Path(temporary, "source.json").write_text(
                json.dumps({
                    "selection": {
                        "bbox_south_west_north_east": [59.0, 17.0, 59.2, 17.2],
                        "cells": 1024,
                        "cell_size_metres": 25,
                    },
                    "osm": {"raw_json": "osm/raw.json", "query": "osm/query.txt"},
                    "elevation": {"heightmap": "elevation/heightmap.png"},
                }),
                encoding="utf-8",
            )
            values = default_gui_values()
            values.update({"source_mode": "existing", "source_dir": temporary})
            self.assertEqual(format_wizard_world_size(values), "25.6 km (1024×1024 at 25 m)")

    def test_world_slug_is_cwa_friendly(self) -> None:
        self.assertEqual(slugify_world_name("Lake Mälaren 2026!"), "lake_m_laren_2026")
        self.assertEqual(slugify_world_name("123"), "world_123")

    def test_increment_trailing_number_preserves_width(self) -> None:
        self.assertEqual(increment_trailing_number("test10"), "test11")
        self.assertEqual(increment_trailing_number("test009"), "test010")
        self.assertEqual(increment_trailing_number("test"), "test2")

    def test_name_suggestion_preserves_existing_source_bundle(self) -> None:
        values = suggested_world_values(
            "foresttest3",
            source_mode="existing",
            source_dir="source-data/foresttest2",
        )
        self.assertEqual(values["name"], "cwr_foresttest3")
        self.assertEqual(values["output"], str(Path("build") / "foresttest3"))
        self.assertEqual(values["source_dir"], "source-data/foresttest2")

    def test_name_suggestion_updates_source_folder_for_new_area(self) -> None:
        values = suggested_world_values(
            "foresttest3",
            source_mode="new",
            source_dir="source-data/foresttest2",
        )
        self.assertEqual(values["source_dir"], str(Path("source-data") / "foresttest3"))


    def test_wizard_new_area_build_pipeline_fetches_then_builds(self) -> None:
        values = default_gui_values()
        fetch_values = {
            "source_dir": values["source_dir"],
            "selection_mode": values["selection_mode"],
            "south": values["south"],
            "west": values["west"],
            "north": values["north"],
            "east": values["east"],
            "cells": values["fetch_cells"],
            "cell_size": values["fetch_cell_size"],
        }
        commands = build_wizard_pipeline_commands(
            source_mode="new",
            fetch_values=fetch_values,
            build_values=values,
            python="python",
        )
        self.assertEqual(len(commands), 2)
        self.assertIn("fetch-sources", commands[0])
        self.assertIn("milestone9", commands[1])

    def test_wizard_pipeline_skips_fetch_for_existing_bundle(self) -> None:
        values = default_gui_values()
        commands = build_wizard_pipeline_commands(
            source_mode="existing",
            fetch_values={},
            build_values=values,
            python="python",
        )
        self.assertEqual(len(commands), 1)
        self.assertIn("milestone9", commands[0])

    def test_wizard_has_progressive_build_sequence(self) -> None:
        self.assertEqual(
            [step.title for step in WIZARD_STEPS],
            ["Choose map area", "Name the world", "Choose appearance", "Build"],
        )

    def test_primary_action_does_not_turn_a_double_click_into_stop(self) -> None:
        app = object.__new__(WorldgenGui)
        app.process = object()
        app._pipeline_active = True
        stop_calls: list[bool] = []
        app._stop_process = lambda: stop_calls.append(True)

        WorldgenGui._primary_action(app)

        self.assertEqual(stop_calls, [])

    def test_command_preview_quotes_spaces(self) -> None:
        rendered = quote_command(["python", "-m", "cwr_worldgen", "--value", "a b"])
        self.assertIn("a b", rendered)


class AdvancedSettingHighlightTests(unittest.TestCase):
    class _Var:
        def __init__(self, value: object) -> None:
            self.value = value

        def get(self) -> object:
            return self.value

    class _Widget:
        def __init__(self) -> None:
            self.style = ""

        def configure(self, *, style: str) -> None:
            self.style = style

    def test_numeric_formatting_equal_to_default_stays_unbolded(self) -> None:
        self.assertTrue(WorldgenGui._advanced_value_matches_default("0.50", "0.5"))
        self.assertTrue(WorldgenGui._advanced_value_matches_default("8.0", "8"))
        self.assertFalse(WorldgenGui._advanced_value_matches_default("0.25", "0.5"))

    def test_advanced_setting_style_bolds_changed_values_and_restores_default(self) -> None:
        label = self._Widget()
        checkbox = self._Widget()
        fake_gui = type("FakeGui", (), {})()
        fake_gui.vars = {
            "forest_polygon_sink_fraction": self._Var("0.25"),
            "cache_refresh": self._Var(False),
        }
        fake_gui._advanced_default_values = {
            "forest_polygon_sink_fraction": "0.5",
            "cache_refresh": False,
        }
        fake_gui.advanced_setting_widgets = {
            "forest_polygon_sink_fraction": [
                (label, "TLabel", "AdvancedChanged.TLabel")
            ],
            "cache_refresh": [
                (checkbox, "TCheckbutton", "AdvancedChanged.TCheckbutton")
            ],
        }
        fake_gui._advanced_value_matches_default = WorldgenGui._advanced_value_matches_default

        WorldgenGui._update_advanced_setting_styles(fake_gui)
        self.assertEqual(label.style, "AdvancedChanged.TLabel")
        self.assertEqual(checkbox.style, "TCheckbutton")

        fake_gui.vars["forest_polygon_sink_fraction"].value = "0.500"
        fake_gui.vars["cache_refresh"].value = True
        WorldgenGui._update_advanced_setting_styles(fake_gui)
        self.assertEqual(label.style, "TLabel")
        self.assertEqual(checkbox.style, "AdvancedChanged.TCheckbutton")


if __name__ == "__main__":
    unittest.main()
