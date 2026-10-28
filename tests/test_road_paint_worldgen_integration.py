"""Synthetic town road A/B tests using the real worldgen road-fitter pipeline."""
from pathlib import Path

import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as quality
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _town():
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    style = {"highway": "service", "surface": "dirt"}
    def road(key, points):
        return OsmLineFeature(
            key, style,
            tuple(projection.to_latlon(point) for point in points),
        )
    data = OsmDataset(
        source_generator="town-road-paint-ab",
        element_count=2,
        coastlines=(), water=(), forests=(), farmland=(), urban=(),
        roads=(
            road("way/through", ((300.0, 500.0), (700.0, 500.0))),
            road("way/spur", ((500.0, 350.0), (500.0, 498.0))),
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="town_paint_ab",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )
    return data, projection, spec


def test_town_mouth_becomes_a_true_junction_in_worldgen(monkeypatch):
    data, projection, spec = _town()
    key = playability._road_node_key((500.0, 500.0))
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    baseline = quality._junction_geometry(data, projection, spec)
    assert key not in baseline
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    repaired = quality._junction_geometry(data, projection, spec)
    assert key in repaired
    assert len(repaired[key].directions) == 3


def test_town_fitter_accepts_repaired_road_network(monkeypatch):
    data, projection, spec = _town()
    heights = [0.0] * (spec.cells * spec.cells)
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    baseline = playability.fit_road_objects(data, projection, heights, spec)
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    repaired = playability.fit_road_objects(data, projection, heights, spec)
    assert repaired.objects
    assert not repaired.truncated
    print(
        "Town A/B:",
        "control=", (len(baseline.objects), baseline.junction_cap_objects,
                     baseline.failed_connections, baseline.maximum_connection_gap),
        "repair=", (len(repaired.objects), repaired.junction_cap_objects,
                    repaired.failed_connections, repaired.maximum_connection_gap),
    )
