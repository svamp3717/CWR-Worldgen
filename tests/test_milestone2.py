# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from cwr_worldgen.generator import build_milestone2
from cwr_worldgen.images import regrid_cell_centres_to_vertices
from cwr_worldgen.model import HeightmapSpec
from cwr_worldgen.paa import inspect_paa
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.wrp import inspect_rvw4, quantize_elevations


def _write_radial_u16_png(path: Path, size: int = 64) -> None:
    values: list[int] = []
    centre = (size - 1) / 2.0
    maximum_distance = math.hypot(centre, centre)
    for y in range(size):
        for x in range(size):
            distance = math.hypot(x - centre, y - centre) / maximum_distance
            value = max(0.0, 1.0 - distance)
            values.append(int(round(value * 65535.0)))
    image = Image.new("I;16", (size, size))
    image.putdata(values)
    image.save(path)


def _write_float_tiff(path: Path, width: int, height: int, values: list[float]) -> None:
    image = Image.new("F", (width, height))
    image.putdata(values)
    image.save(path, format="TIFF")


class Milestone2Tests(unittest.TestCase):
    def test_legacy_cell_centres_are_shifted_onto_wrp_vertices(self) -> None:
        self.assertEqual(
            regrid_cell_centres_to_vertices(
                (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
                3,
                3,
            ),
            (0.0, 5.0, 15.0, 15.0, 20.0, 30.0, 45.0, 50.0, 60.0),
        )

    def test_final_terrain_grid_matches_exact_rvw4_height_values(self) -> None:
        final = quantize_elevations((10.024, 10.026, -1.024, -1.026), 0.05)
        self.assertEqual(final, (10.0, 10.05, -1.0, -1.05))
        self.assertTrue(all(abs(value / 0.05 - round(value / 0.05)) < 1e-9 for value in final))

    def test_programmatic_heightmap_default_uses_25m_cells(self) -> None:
        spec = HeightmapSpec(heightmap_path=Path("heightmap.png"))
        self.assertEqual(spec.cells, 256)
        self.assertEqual(spec.cell_size, 25.0)
        self.assertEqual(spec.world_size, 6400.0)

    def test_builds_heightmap_world_with_four_embedded_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            _write_radial_u16_png(heightmap)
            spec = HeightmapSpec(heightmap_path=heightmap, cells=128)
            result = build_milestone2(root / "build", spec)

            summary = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertEqual((summary.width, summary.height), (128, 128))
            self.assertLess(summary.minimum_height, 0)
            self.assertGreater(summary.maximum_height, 100)
            self.assertEqual(
                set(summary.texture_paths),
                {
                    r"cwr_milestone2\data\d.paa",
                    r"cwr_milestone2\data\w.paa",
                    r"cwr_milestone2\data\s.paa",
                    r"cwr_milestone2\data\g.paa",
                    r"cwr_milestone2\data\r.paa",
                },
            )
            self.assertEqual(summary.texture_slots[0], r"cwr_milestone2\data\d.paa")
            self.assertNotEqual(summary.texture_slots[0], summary.texture_slots[1])
            self.assertEqual(summary.texture_index_counts[0], 0)
            self.assertGreaterEqual(sum(count > 0 for count in summary.texture_index_counts[1:]), 3)
            self.assertEqual(summary.object_count, 0)
            self.assertTrue(summary.has_object_terminator)

            self.assertEqual({path.name for path in result.texture_paths}, {"d.paa", "w.paa", "s.paa", "g.paa", "r.paa"})
            for texture_path in result.texture_paths:
                texture = inspect_paa(texture_path)
                self.assertEqual(texture.magic, 0xFF01)
                self.assertEqual((texture.width, texture.height), (128, 128))
                self.assertEqual((texture.minimum_mip_width, texture.minimum_mip_height), (4, 4))
                self.assertIn("AVGC", texture.tags)
                self.assertIn("OFFS", texture.tags)

            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            self.assertIn('worlds[] = {"cwr_milestone2"};', config)
            self.assertIn('units[] = {};', config)
            self.assertIn('class cwr_milestone2: Intro', config)
            self.assertNotIn('class CfgVehicles', config)
            self.assertIn('cutscenes[] = {"intro"};', config)
            self.assertNotIn("cutscenes[] = {};", config)
            self.assertIn('icon = "\\cwr_milestone2\\data\\g.paa";', config)

            mission = result.mission_path.read_text(encoding="ascii")
            self.assertIn('vehicle="SoldierWB";', mission)
            self.assertIn('leader=1;', mission)
            self.assertRegex(mission, r'position\[\]=\{[^,]+,0\.000,[^}]+\};')

            self.assertTrue(result.intro_mission_path.is_file())
            self.assertTrue(result.intro_script_path.is_file())
            intro = result.intro_mission_path.read_text(encoding="ascii")
            self.assertIn("class Intro", intro)
            self.assertIn('vehicle="SoldierWB";', intro)
            self.assertIn('addOns[]={"cwr_milestone2"};', intro)
            self.assertIn("camCreate", result.intro_script_path.read_text(encoding="ascii"))

            entries = {entry.name.casefold(): entry.data for entry in read_pbo(result.pbo_path)}
            self.assertEqual(
                set(entries),
                {
                    "config.cpp",
                    "cwr_milestone2.wrp",
                    r"data\d.paa",
                    r"data\w.paa",
                    r"data\s.paa",
                    r"data\g.paa",
                    r"data\r.paa",
                },
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["milestone"], 2)
            self.assertGreater(manifest["spawn"]["y"], spec.sea_level)
            self.assertEqual(len(manifest["materials"]), 4)
            self.assertTrue(result.preview_path.is_file())
            self.assertTrue(result.height_preview_path and result.height_preview_path.is_file())
            self.assertTrue(result.material_preview_path and result.material_preview_path.is_file())
            self.assertIn("Failures: 0", result.report_path.read_text(encoding="utf-8"))

    def test_float_tiff_can_be_interpreted_as_metres(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.tiff"
            width = height = 16
            values = []
            for y in range(height):
                for x in range(width):
                    values.append(-5.0 + (x + y) / (width + height - 2) * 105.0)
            _write_float_tiff(heightmap, width, height, values)
            spec = HeightmapSpec(
                heightmap_path=heightmap,
                cells=32,
                heightmap_grid="game-terrain-vertices",
                input_mode="meters",
                rock_height=80.0,
            )
            result = build_milestone2(root / "build", spec)
            summary = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertAlmostEqual(summary.minimum_height, -5.0, delta=0.1)
            self.assertAlmostEqual(summary.maximum_height, 100.0, delta=0.1)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["heightmap"]["source_mode"], "F")
            self.assertEqual(manifest["world"]["input_mode"], "meters")

    def test_material_mask_overrides_automatic_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            height = Image.new("L", (8, 8), 128)
            height.save(heightmap)
            mask_path = root / "materials.png"
            mask = Image.new("L", (8, 8))
            pixels = []
            for y in range(8):
                for x in range(8):
                    if x < 4 and y < 4:
                        pixels.append(0)
                    elif x >= 4 and y < 4:
                        pixels.append(85)
                    elif x < 4:
                        pixels.append(170)
                    else:
                        pixels.append(255)
            mask.putdata(pixels)
            mask.save(mask_path)

            spec = HeightmapSpec(
                heightmap_path=heightmap,
                material_mask_path=mask_path,
                cells=32,
                elevation_minimum=10.0,
                elevation_maximum=20.0,
                sea_level=0.0,
                rock_height=1000.0,
                rock_slope_degrees=89.0,
            )
            result = build_milestone2(root / "build", spec)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            counts = [material["cells"] for material in manifest["materials"]]
            self.assertEqual(counts, [256, 256, 256, 256])
            self.assertIsNotNone(manifest["material_mask"])

    def test_rejects_heightmap_with_no_playable_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "water.png"
            Image.new("L", (16, 16), 0).save(heightmap)
            spec = HeightmapSpec(
                heightmap_path=heightmap,
                cells=32,
                elevation_minimum=-20.0,
                elevation_maximum=-5.0,
            )
            with self.assertRaisesRegex(ValueError, "no playable spawn cell"):
                build_milestone2(root / "build", spec)

    def test_rejects_elevations_outside_rvw4_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "too-high.tiff"
            _write_float_tiff(heightmap, 8, 8, [2000.0] * 64)
            spec = HeightmapSpec(heightmap_path=heightmap, cells=32, input_mode="meters")
            with self.assertRaisesRegex(ValueError, "cannot be represented"):
                build_milestone2(root / "build", spec)

    def test_test_mission_uses_stock_player_unit_and_world_addon(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            _write_radial_u16_png(heightmap, 32)
            result = build_milestone2(root / "build", HeightmapSpec(heightmap_path=heightmap, cells=64))
            mission = result.mission_path.read_text(encoding="ascii")
            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            self.assertIn('units[] = {};', config)
            self.assertNotIn('class CfgVehicles', config)
            self.assertIn('cutscenes[] = {"intro"};', config)
            self.assertNotIn("cutscenes[] = {};", config)
            self.assertIn('icon = "\\cwr_milestone2\\data\\g.paa";', config)
            self.assertIn('class cwr_milestone2: Intro', config)
            self.assertIn('vehicle="SoldierWB";', mission)
            self.assertIn('addOns[]={"cwr_milestone2"};', mission)
            self.assertIn('player="PLAYER COMMANDER";', mission)
            self.assertIn('leader=1;', mission)

    def test_world_exit_has_generated_menu_intro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            _write_radial_u16_png(heightmap, 32)
            result = build_milestone2(root / "build", HeightmapSpec(heightmap_path=heightmap, cells=64))
            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            self.assertIn('cutscenes[] = {"intro"};', config)
            self.assertNotIn("cutscenes[] = {};", config)
            self.assertEqual(result.intro_mission_path.parent.name, "intro.cwr_milestone2")
            self.assertTrue(result.intro_mission_path.is_file())
            self.assertTrue(result.intro_script_path.is_file())

    def test_heightmap_build_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            _write_radial_u16_png(heightmap, 32)
            spec = HeightmapSpec(heightmap_path=heightmap, cells=64)
            first = build_milestone2(root / "first", spec)
            second = build_milestone2(root / "second", spec)
            self.assertEqual(
                hashlib.sha256(first.pbo_path.read_bytes()).digest(),
                hashlib.sha256(second.pbo_path.read_bytes()).digest(),
            )
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.preview_path.read_bytes(), second.preview_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
