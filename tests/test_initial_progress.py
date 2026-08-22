# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmRaster,
    parse_overpass_json,
    prepare_spatial_index,
    rasterize_osm,
)
from cwr_worldgen.progress import format_duration, progress_range, report_progress, reset_progress_session
from cwr_worldgen.terrain_solver import solve_terrain_constraints


class InitialProgressTests(unittest.TestCase):
    def test_large_osm_parse_reports_element_counts(self) -> None:
        elements = [
            {
                "type": "way",
                "id": index,
                "tags": {"highway": "residential"},
                "geometry": [
                    {"lat": 0.1, "lon": 0.1},
                    {"lat": 0.1 + index * 1e-7, "lon": 0.2},
                ],
            }
            for index in range(1, 241)
        ]
        events: list[tuple[int, str]] = []
        dataset = parse_overpass_json(
            json.dumps({"generator": "progress-test", "elements": elements}).encode("utf-8"),
            progress_callback=lambda percent, stage: events.append((percent, stage)),
        )
        self.assertEqual(dataset.element_count, 240)
        self.assertGreaterEqual(len(events), 20)
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], 100)
        self.assertTrue(any("240/240" in stage for _percent, stage in events))
        self.assertEqual([percent for percent, _stage in events], sorted(percent for percent, _stage in events))

    def test_raster_and_spatial_index_report_feature_progress(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        osm = {
            "generator": "raster-progress",
            "elements": [
                {
                    "type": "way",
                    "id": index,
                    "tags": {"highway": "residential"},
                    "geometry": [
                        {"lat": 0.1 + index * 0.01, "lon": 0.1},
                        {"lat": 0.1 + index * 0.01, "lon": 0.9},
                    ],
                }
                for index in range(1, 21)
            ],
        }
        dataset = parse_overpass_json(json.dumps(osm).encode("utf-8"))
        raster_events: list[tuple[int, str]] = []
        raster = rasterize_osm(
            dataset,
            projection,
            cells=8,
            include_minor_roads=True,
            progress_callback=lambda percent, stage: raster_events.append((percent, stage)),
        )
        self.assertEqual(raster.cells, 8)
        self.assertTrue(any("Rasterizing roads" in stage for _percent, stage in raster_events))
        self.assertTrue(any("Downsampling" in stage for _percent, stage in raster_events))
        self.assertEqual(raster_events[-1][0], 100)

        index_events: list[tuple[int, str]] = []
        spatial = prepare_spatial_index(
            dataset,
            projection,
            use_cache=False,
            progress_callback=lambda percent, stage: index_events.append((percent, stage)),
        )
        self.assertGreater(len(spatial.road_segments), 0)
        self.assertTrue(any("Indexing projected roads" in stage for _percent, stage in index_events))
        self.assertEqual(index_events[-1][0], 100)

    def test_constraint_solver_reports_iterations_and_feature_stages(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="solver-progress",
            element_count=0,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(),
        )
        raster = OsmRaster(
            cells=4,
            water=(False,) * 16,
            forest=(False,) * 16,
            farmland=(False,) * 16,
            urban=(False,) * 16,
            roads=(False,) * 16,
            buildings=(False,) * 16,
            high_resolution=4,
            coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_solver_progress",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=4,
            cell_size=25.0,
            solver_iterations=6,
            strict_assets=False,
        )
        events: list[tuple[int, str]] = []
        solve_terrain_constraints(
            (5.0,) * 16,
            dataset,
            projection,
            raster,
            spec,
            progress_callback=lambda percent, stage: events.append((percent, stage)),
        )
        self.assertTrue(any("coastal" in stage and "water" in stage for _percent, stage in events))
        self.assertTrue(any("6/6 iterations" in stage for _percent, stage in events))
        self.assertEqual(events[-1], (100, "Terrain constraint solution ready"))
        self.assertEqual([percent for percent, _stage in events], sorted(percent for percent, _stage in events))

    def test_progress_range_reserves_initial_percentage_for_milestone_setup(self) -> None:
        output = StringIO()
        reset_progress_session()
        with patch.dict(os.environ, {"CWR_PROGRESS": "1", "CWR_PROGRESS_FORMAT": "machine"}, clear=False):
            with redirect_stdout(output):
                report_progress(10, "setup")
                with progress_range(12, 99):
                    report_progress(0, "core-start")
                    report_progress(100, "core-end")
                report_progress(100, "complete")
        lines = output.getvalue().strip().splitlines()
        expected = ((10, "setup"), (12, "core-start"), (99, "core-end"), (100, "complete"))
        for line, (percent, stage) in zip(lines, expected, strict=True):
            parts = line.split("\t")
            self.assertEqual(parts[:3], ["CWR_PROGRESS", str(percent), stage])
            self.assertRegex(parts[3], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
            self.assertRegex(parts[4], r"^\d{2,}:\d{2}:\d{2}\.\d{3}$")

    def test_duration_format_includes_hours_minutes_seconds_and_milliseconds(self) -> None:
        self.assertEqual(format_duration(3661.234), "01:01:01.234")
        self.assertEqual(format_duration(61.9, milliseconds=False), "00:01:02")


if __name__ == "__main__":
    unittest.main()
