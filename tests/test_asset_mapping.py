# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from cwr_worldgen.asset_mapping import (
    collect_osm_asset_requirements,
    default_osm_asset_mapping,
    load_osm_asset_mapping,
)
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPointFeature,
    OsmPolygonFeature,
)


class OsmAssetMappingTests(unittest.TestCase):
    def _dataset(self) -> OsmDataset:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        road = OsmLineFeature(
            "way/road",
            {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in ((10.0, 50.0), (90.0, 50.0))),
        )
        bus = OsmPointFeature(
            "node/bus",
            {"landmark": "bus_stop"},
            projection.to_latlon((30.0, 50.0)),
        )
        cemetery_ring = tuple(
            projection.to_latlon(point)
            for point in ((10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0), (10.0, 10.0))
        )
        cemetery = OsmPolygonFeature(
            "way/cemetery", {"site": "cemetery"}, (GeoPolygon(cemetery_ring),)
        )
        return OsmDataset(
            source_generator="asset-map-test",
            element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(road,), gravel_roads=(road,), landmarks=(bus,), sites=(cemetery,),
        )

    def _spec(self) -> _Milestone9PlayabilitySpec:
        return _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=16,
            cell_size=6.25,
            strict_assets=False,
        )

    def test_built_in_mapping_keeps_current_bus_stop_and_grave_assets(self) -> None:
        spec = self._spec()
        mapping = default_osm_asset_mapping(spec, 9)
        report = collect_osm_asset_requirements(self._dataset(), mapping)
        self.assertIn(r"o\misc\aut_z_st.p3d", report.selected_models)
        self.assertIn(r"o\hous\nahrobek1.p3d", report.selected_models)
        self.assertIn(rf"{spec.name}\i\gravel25.p3d", report.selected_models)
        self.assertIn(rf"{spec.name}\i\g.paa", report.selected_textures)
        self.assertGreaterEqual(report.matched_feature_count, 3)

    def test_custom_mapping_merges_and_overrides_rules_by_id(self) -> None:
        defaults = default_osm_asset_mapping(self._spec(), 9)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.json"
            path.write_text(json.dumps({
                "schema": 1,
                "inherit_defaults": True,
                "rules": [
                    {
                        "id": "bus-stop-signs",
                        "layers": ["landmarks"],
                        "geometry": "point",
                        "match": {"landmark": "bus_stop"},
                        "models": [r"my_mod\bus_sign.p3d"],
                        "textures": [r"my_mod\bus_sign.paa"],
                    }
                ],
            }), encoding="utf-8")
            mapping = load_osm_asset_mapping(path, defaults)
            report = collect_osm_asset_requirements(self._dataset(), mapping)
        self.assertIn(r"my_mod\bus_sign.p3d", report.selected_models)
        self.assertIn(r"my_mod\bus_sign.paa", report.selected_textures)
        self.assertNotIn(r"o\misc\aut_z_st.p3d", report.selected_models)
        self.assertIn(r"o\hous\nahrobek1.p3d", report.selected_models)


    def test_surface_unpaved_uses_gravel_asset_layer(self) -> None:
        road = OsmLineFeature(
            "way/unpaved",
            {"highway": "service", "surface": "unpaved"},
            ((0.0, 0.0), (0.0, 0.001)),
        )
        dataset = OsmDataset(
            source_generator="test", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = self._spec()
        mapping = default_osm_asset_mapping(spec, 9)
        report = collect_osm_asset_requirements(dataset, mapping)
        self.assertIn(rf"{spec.name}\i\gravel25.p3d", report.selected_models)
        self.assertIn(rf"{spec.name}\i\g.paa", report.selected_textures)
        matched = {match.rule_id for match in report.rule_matches}
        self.assertIn("road-gravel", matched)

    def test_custom_mapping_can_replace_defaults(self) -> None:
        defaults = default_osm_asset_mapping(self._spec(), 9)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.json"
            path.write_text(json.dumps({
                "schema": 1,
                "inherit_defaults": False,
                "global": {"textures": [r"my_mod\shared.paa"]},
                "rules": [
                    {
                        "id": "gravel-service",
                        "layers": ["gravel_roads"],
                        "geometry": "line",
                        "match": {"highway": "service", "surface": "gravel"},
                        "models": [r"my_mod\gravel25.p3d"],
                    }
                ],
            }), encoding="utf-8")
            mapping = load_osm_asset_mapping(path, defaults)
            report = collect_osm_asset_requirements(self._dataset(), mapping)
        self.assertEqual(report.selected_models, (r"my_mod\gravel25.p3d",))
        self.assertEqual(report.selected_textures, (r"my_mod\shared.paa",))

    def test_mapping_rejects_wrong_asset_suffix(self) -> None:
        defaults = default_osm_asset_mapping(self._spec(), 9)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.json"
            path.write_text(json.dumps({
                "schema": 1,
                "rules": [{
                    "id": "bad",
                    "layers": ["roads"],
                    "match": {"highway": "service"},
                    "models": [r"my_mod\not_a_model.txt"],
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong suffix"):
                load_osm_asset_mapping(path, defaults)


if __name__ == "__main__":
    unittest.main()
