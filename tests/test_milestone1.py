# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from cwr_worldgen.generator import WorldSpec, build_milestone1
from cwr_worldgen.paa import inspect_paa
from cwr_worldgen.templates import render_config, validate_cwa_config
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.wrp import inspect_rvw4


class Milestone1Tests(unittest.TestCase):
    def test_programmatic_world_default_uses_25m_cells(self) -> None:
        spec = WorldSpec()
        self.assertEqual(spec.cells, 256)
        self.assertEqual(spec.cell_size, 25.0)
        self.assertEqual(spec.world_size, 6400.0)

    def test_builds_complete_self_contained_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone1(Path(temp) / "build")
            self.assertTrue(result.wrp_path.is_file())
            self.assertTrue(result.texture_path.is_file())
            self.assertTrue(result.pbo_path.is_file())
            self.assertTrue(result.mission_path.is_file())
            self.assertTrue(result.intro_mission_path.is_file())
            self.assertTrue(result.intro_script_path.is_file())
            self.assertTrue(result.preview_path.is_file())
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.report_path.is_file())

            summary = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertEqual((summary.width, summary.height), (256, 256))
            self.assertLess(summary.minimum_height, 0)
            self.assertGreater(summary.maximum_height, 0)
            self.assertEqual(
                set(summary.texture_paths),
                {r"cwr_milestone1\data\d.paa", r"cwr_milestone1\data\g.paa"},
            )
            self.assertEqual(summary.texture_slots[0], r"cwr_milestone1\data\d.paa")
            self.assertEqual(summary.texture_slots[1], r"cwr_milestone1\data\g.paa")
            self.assertEqual(summary.texture_index_counts[0], 0)
            self.assertEqual(summary.object_count, 0)
            self.assertEqual(summary.object_ids, ())
            self.assertTrue(summary.has_object_terminator)

            for texture_path in result.texture_paths:
                texture = inspect_paa(texture_path)
                self.assertEqual(texture.magic, 0xFF01)
                self.assertEqual((texture.width, texture.height), (128, 128))
                self.assertEqual(texture.mipmap_count, 6)
                self.assertEqual((texture.minimum_mip_width, texture.minimum_mip_height), (4, 4))
                self.assertIn("AVGC", texture.tags)
                self.assertIn("OFFS", texture.tags)

            pbo = {entry.name.casefold(): entry.data for entry in read_pbo(result.pbo_path)}
            self.assertEqual(
                set(pbo),
                {"config.cpp", "cwr_milestone1.wrp", r"data\d.paa", r"data\g.paa"},
            )
            self.assertEqual(pbo["cwr_milestone1.wrp"], result.wrp_path.read_bytes())
            self.assertEqual(pbo[r"data\g.paa"], result.texture_path.read_bytes())
            config = pbo["config.cpp"].decode("ascii")
            self.assertIn("class DefaultWorld", config)
            self.assertIn("class Intro: DefaultWorld", config)
            self.assertIn("class cwr_milestone1: Intro", config)
            self.assertNotIn("class Abel;", config)
            self.assertNotIn("class CfgVehicles", config)
            validate_cwa_config(config)
            self.assertIn("class CfgWorldList", config)
            self.assertNotIn("class CfgAddons", config)
            self.assertIn('worlds[] = {"cwr_milestone1"};', config)
            self.assertIn("units[] = {};", config)
            self.assertIn('cutscenes[] = {"intro"};', config)
            self.assertNotIn("cutscenes[] = {};", config)
            self.assertIn('icon = "\\cwr_milestone1\\data\\g.paa";', config)

            mission = result.mission_path.read_text(encoding="ascii")
            self.assertIn('vehicle="SoldierWB";', mission)
            self.assertIn('addOns[]={"cwr_milestone1"};', mission)
            self.assertIn("leader=1;", mission)
            self.assertRegex(mission, r"position\[\]=\{[^,]+,0\.000,[^}]+\};")

            intro = result.intro_mission_path.read_text(encoding="ascii")
            self.assertIn("class Intro", intro)
            self.assertIn('vehicle="SoldierWB";', intro)
            self.assertIn('addOns[]={"cwr_milestone1"};', intro)
            self.assertIn("camCreate", result.intro_script_path.read_text(encoding="ascii"))

            report = result.report_path.read_text(encoding="utf-8")
            self.assertNotIn("[FAIL]", report)
            self.assertIn("Failures: 0", report)

    def test_config_declares_world_owner_for_unsaved_editor_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone1(Path(temp) / "build")
            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            self.assertRegex(
                config,
                r'(?s)class CfgPatches\s*\{.*class cwr_milestone1\s*\{.*worlds\[\]\s*=\s*\{"cwr_milestone1"\};',
            )
            mission = result.mission_path.read_text(encoding="ascii")
            self.assertIn('addOns[]={"cwr_milestone1"};', mission)
            self.assertIn('vehicle="SoldierWB";', mission)

    def test_world_exit_has_generated_menu_intro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone1(Path(temp) / "build")
            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            self.assertIn('cutscenes[] = {"intro"};', config)
            self.assertNotIn("cutscenes[] = {};", config)
            self.assertEqual(result.intro_mission_path.parent.name, "intro.cwr_milestone1")
            self.assertTrue(result.intro_mission_path.is_file())
            self.assertTrue(result.intro_script_path.is_file())

    def test_render_config_registers_procedural_door_buildings(self) -> None:
        spec = WorldSpec(name="door_world", display_name="Door World")
        config = render_config(
            spec,
            milestone=9,
            animated_building_models=(
                r"door_world\g\b_a1b2_deadbeef.p3d",
                r"door_world\g\b_a1b2_cafefeed.p3d",
            ),
        )
        validate_cwa_config(config)
        self.assertIn("class CfgVehicles", config)
        self.assertIn("class CWR_door_world_ProceduralDoorHouse\n", config)
        self.assertNotIn("class CWR_door_world_ProceduralDoorHouse: House", config)
        self.assertIn('simulation = "house";', config)
        self.assertIn("scope = 0;", config)
        self.assertIn("class Land_b_a1b2_deadbeef: CWR_door_world_ProceduralDoorHouse", config)
        self.assertRegex(config, r"class Land_b_a1b2_deadbeef: CWR_door_world_ProceduralDoorHouse\s*\{\s*scope = 1;")
        self.assertIn('model = "\\door_world\\g\\b_a1b2_deadbeef.p3d";', config)
        self.assertIn('selection = "door1";', config)
        self.assertIn('axis = "door1_axis";', config)
        self.assertIn('position = "door1_action";', config)
        self.assertIn('displayName = "Open door";', config)
        self.assertIn('displayName = "Close door";', config)
        self.assertEqual(config.count("radius = 4.0;"), 2)
        self.assertIn('statement = "this animate [""Door1"", 1]";', config)
        self.assertIn('statement = "this animate [""Door1"", 0]";', config)

    def test_rejects_cwa_unsupported_bare_class_declaration(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported bare class declaration"):
            validate_cwa_config("class CfgWorlds { class Abel; };")

    def test_rejects_cwa_inheritance_from_undeclared_external_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared base class House"):
            validate_cwa_config("class CfgVehicles { class DoorHouse: House {}; };")

    def test_rvw4_wire_layout_has_serializer_terminator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone1(Path(temp) / "build")
            data = result.wrp_path.read_bytes()
            self.assertEqual(data[:4], b"4WVR")
            self.assertEqual(struct.unpack_from("<ii", data, 4), (256, 256))
            expected_size = 12 + 256 * 256 * 2 + 256 * 256 * 2 + 512 * 32 + 128
            self.assertEqual(len(data), expected_size)
            self.assertEqual(data[-128:], bytes(128))
            self.assertNotIn(b"data3d\\smrk.p3d", data)
            self.assertNotIn(b"landtext\\mo.pac", data)

    def test_build_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = build_milestone1(Path(temp) / "first")
            second = build_milestone1(Path(temp) / "second")
            first_hash = hashlib.sha256(first.pbo_path.read_bytes()).digest()
            second_hash = hashlib.sha256(second.pbo_path.read_bytes()).digest()
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.texture_path.read_bytes(), second.texture_path.read_bytes())

    def test_cwr_ce_profile_uses_fixed_rvw4_height_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            spec = WorldSpec(profile="cwr-ce")
            result = build_milestone1(Path(temp) / "build", spec)
            summary = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertAlmostEqual(summary.minimum_height, -5.0, places=3)
            self.assertAlmostEqual(summary.maximum_height, 5.0, places=3)

    def test_rejects_unsafe_or_too_long_world_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build_milestone1(Path(temp) / "build", WorldSpec(name="Bad/World"))
            with self.assertRaises(ValueError):
                build_milestone1(Path(temp) / "build", WorldSpec(name="a" * 21))


if __name__ == "__main__":
    unittest.main()
