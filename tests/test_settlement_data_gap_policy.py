# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmPointFeature
from cwr_worldgen.settlement_data_gap_policy import (
    SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES,
    find_settlement_building_gaps,
    notify_missing_settlement_buildings,
)


class SettlementDataGapPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)

    def _dataset(
        self,
        *,
        place_type: str = "village",
        building_offset_metres: float | None = None,
        building_tags: dict[str, str] | None = None,
    ) -> OsmDataset:
        place = OsmPointFeature(
            "node/test-place",
            {"place": place_type, "name": "Example Settlement"},
            self.projection.to_latlon((500.0, 500.0)),
        )
        building_points = ()
        if building_offset_metres is not None:
            building_points = (
                OsmPointFeature(
                    "node/test-building",
                    {"building": "yes", **(building_tags or {})},
                    self.projection.to_latlon((500.0 + building_offset_metres, 500.0)),
                ),
            )
        return OsmDataset(
            source_generator="settlement-gap-test",
            element_count=1 + len(building_points),
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(),
            building_points=building_points,
            places=(place,),
        )

    def test_missing_village_is_reported_when_no_building_source_covers_it(self) -> None:
        gaps = find_settlement_building_gaps(self._dataset(), self.projection)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].place_type, "village")
        self.assertEqual(gaps[0].name, "Example Settlement")
        self.assertEqual(
            gaps[0].radius_metres,
            SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES["village"],
        )

    def test_nearby_osm_or_overture_building_suppresses_the_warning(self) -> None:
        osm_dataset = self._dataset(building_offset_metres=100.0)
        overture_dataset = self._dataset(
            building_offset_metres=100.0,
            building_tags={"cwr:overture_source": "test"},
        )

        self.assertEqual(find_settlement_building_gaps(osm_dataset, self.projection), ())
        self.assertEqual(find_settlement_building_gaps(overture_dataset, self.projection), ())

    def test_building_outside_the_settlement_radius_does_not_hide_a_gap(self) -> None:
        radius = SETTLEMENT_BUILDING_SEARCH_RADIUS_METRES["village"]
        dataset = self._dataset(building_offset_metres=radius + 25.0)

        self.assertEqual(len(find_settlement_building_gaps(dataset, self.projection)), 1)

    def test_notification_mentions_both_sources_only_after_overture_is_available(self) -> None:
        messages: list[str] = []
        callback = lambda _percent, stage: messages.append(stage)
        dataset = self._dataset(place_type="town")

        disabled = notify_missing_settlement_buildings(
            dataset,
            self.projection,
            overture_available=False,
            progress_callback=callback,
        )
        self.assertEqual(disabled, ())
        self.assertEqual(messages, [])

        gaps = notify_missing_settlement_buildings(
            dataset,
            self.projection,
            overture_available=True,
            progress_callback=callback,
        )
        self.assertEqual(len(gaps), 1)
        self.assertTrue(any("no OSM or Overture buildings" in message for message in messages))
        self.assertTrue(any("Example Settlement" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
