# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

from cwr_worldgen.location_example import square_bbox
from cwr_worldgen.map_picker import (
    bbox_dimensions_metres,
    latlon_to_world_pixel,
    plan_area_selection,
    plan_center_selection,
    plan_initial_selection,
    resize_area_selection,
    world_pixel_to_latlon,
    zoom_for_bbox,
)


class MapPickerTests(unittest.TestCase):
    def test_web_mercator_round_trip(self) -> None:
        latitude, longitude = 59.42766, 16.89457
        x, y = latlon_to_world_pixel(latitude, longitude, 12)
        actual_latitude, actual_longitude = world_pixel_to_latlon(x, y, 12)
        self.assertAlmostEqual(actual_latitude, latitude, places=10)
        self.assertAlmostEqual(actual_longitude, longitude, places=10)

    def test_exact_default_bbox_does_not_round_up_to_512_cells(self) -> None:
        bbox = square_bbox(59.45, 17.0, 6400.0)
        plan = plan_area_selection(bbox)
        self.assertEqual(plan.cells, 256)
        self.assertEqual(plan.world_size_metres, 6400.0)

    def test_rounded_wizard_bbox_preserves_explicit_256_cell_grid(self) -> None:
        bbox = tuple(float(f"{value:.7f}") for value in square_bbox(59.45, 17.0, 6400.0))
        self.assertEqual(plan_area_selection(bbox).cells, 512)
        plan = plan_initial_selection(bbox, cells=256, cell_size_metres=25.0)
        self.assertEqual(plan.cells, 256)
        self.assertEqual(plan.world_size_metres, 6400.0)

    def test_initial_grid_hint_does_not_shrink_genuinely_larger_bbox(self) -> None:
        bbox = square_bbox(59.45, 17.0, 12800.0)
        plan = plan_initial_selection(bbox, cells=256, cell_size_metres=25.0)
        self.assertEqual(plan.cells, 512)
        self.assertEqual(plan.world_size_metres, 12800.0)

    def test_selection_snaps_to_classic_safe_grid(self) -> None:
        bbox = square_bbox(59.42766, 16.89457, 6200.0)
        plan = plan_area_selection(bbox)
        self.assertEqual(plan.cells, 256)
        self.assertEqual(plan.world_size_metres, 6400.0)
        self.assertEqual(plan.severity, "safe")
        width, height = bbox_dimensions_metres(plan.bbox)
        self.assertAlmostEqual(width, 6400.0, delta=5.0)
        self.assertAlmostEqual(height, 6400.0, delta=5.0)

    def test_selection_warns_above_classic_ofp_size(self) -> None:
        bbox = square_bbox(59.42766, 16.89457, 7000.0)
        plan = plan_area_selection(bbox)
        self.assertEqual(plan.cells, 512)
        self.assertEqual(plan.world_size_metres, 12800.0)
        self.assertEqual(plan.severity, "warning")
        self.assertTrue(plan.requires_warning)

    def test_selection_marks_area_beyond_generator_limit(self) -> None:
        bbox = square_bbox(0.0, 0.0, 110000.0)
        plan = plan_area_selection(bbox)
        self.assertFalse(plan.supported)
        self.assertEqual(plan.severity, "unsupported")
        self.assertEqual(plan.cells, 2048)

    def test_center_selection_uses_25m_classic_default(self) -> None:
        plan = plan_center_selection(59.42766, 16.89457)
        self.assertEqual(plan.cells, 256)
        self.assertEqual(plan.cell_size_metres, 25.0)
        self.assertEqual(plan.world_size_metres, 6400.0)
        width, height = bbox_dimensions_metres(plan.bbox)
        self.assertAlmostEqual(width, 6400.0, delta=5.0)
        self.assertAlmostEqual(height, 6400.0, delta=5.0)

    def test_double_size_preserves_center_and_cell_size(self) -> None:
        first = plan_center_selection(59.42766, 16.89457, cells=256, cell_size_metres=25.0)
        doubled = resize_area_selection(first, 2)
        self.assertEqual(doubled.cells, 512)
        self.assertEqual(doubled.cell_size_metres, 25.0)
        self.assertEqual(doubled.world_size_metres, 12800.0)
        first_center = ((first.bbox[0] + first.bbox[2]) / 2.0, (first.bbox[1] + first.bbox[3]) / 2.0)
        doubled_center = ((doubled.bbox[0] + doubled.bbox[2]) / 2.0, (doubled.bbox[1] + doubled.bbox[3]) / 2.0)
        self.assertAlmostEqual(first_center[0], doubled_center[0], places=10)
        self.assertAlmostEqual(first_center[1], doubled_center[1], places=10)

    def test_zoom_fits_bbox(self) -> None:
        bbox = square_bbox(59.42766, 16.89457, 12800.0)
        zoom = zoom_for_bbox(bbox, 900, 600)
        self.assertGreaterEqual(zoom, 2)
        self.assertLessEqual(zoom, 18)


if __name__ == "__main__":
    unittest.main()
