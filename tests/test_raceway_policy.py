# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace
import unittest

from cwr_worldgen.asset_mapping import (
    collect_osm_asset_requirements,
    default_osm_asset_mapping,
)
from cwr_worldgen.normalization import _MAJOR_HIGHWAYS as NORMALIZED_MAJOR_HIGHWAYS
from cwr_worldgen.osm import (
    OsmDataset,
    OsmLineFeature,
    road_is_dirt,
    road_is_supported,
    road_model_for_tags,
)


class RacewayRoadPolicyTests(unittest.TestCase):
    @staticmethod
    def _spec() -> SimpleNamespace:
        return SimpleNamespace(
            name="raceway_test",
            procedural_gravel_roads=False,
            paved_road_model=r"o\road\sil25.p3d",
            dirt_road_model=r"o\road\ces25.p3d",
        )

    def test_raceway_is_supported_and_defaults_to_asphalt(self) -> None:
        tags = {"highway": "raceway", "name": "Test Raceway"}
        spec = self._spec()

        self.assertTrue(road_is_supported(tags, include_minor=False))
        self.assertIn("raceway", NORMALIZED_MAJOR_HIGHWAYS)
        self.assertFalse(road_is_dirt(tags))
        self.assertEqual(road_model_for_tags(spec, tags), spec.paved_road_model)

    def test_raceway_selects_the_paved_asset_rule(self) -> None:
        road = OsmLineFeature(
            "way/test-raceway",
            {"highway": "raceway", "name": "Test Raceway"},
            ((50.0, 7.0), (50.001, 7.002)),
        )
        dataset = OsmDataset(
            source_generator="raceway-test",
            element_count=1,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(road,),
        )
        spec = self._spec()
        report = collect_osm_asset_requirements(
            dataset,
            default_osm_asset_mapping(spec, 9),
        )

        self.assertIn(spec.paved_road_model, report.selected_models)
        matches = {match.rule_id: match for match in report.rule_matches}
        self.assertIn("road-paved", matches)
        self.assertIn("way/test-raceway", matches["road-paved"].sample_osm_keys)

    def test_explicit_unpaved_raceway_still_uses_dirt(self) -> None:
        tags = {"highway": "raceway", "surface": "dirt"}
        spec = self._spec()

        self.assertTrue(road_is_supported(tags, include_minor=False))
        self.assertTrue(road_is_dirt(tags))
        self.assertEqual(road_model_for_tags(spec, tags), spec.dirt_road_model)


if __name__ == "__main__":
    unittest.main()
