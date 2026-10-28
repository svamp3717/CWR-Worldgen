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
    from cwr_worldgen import paved_junction_policy as paved

    # Inspect the unfiltered road topology. The stock-paved junction planner
    # intentionally excludes dirt/gravel hubs from its reservation table.
    baseline = paved._ORIGINAL_GEOMETRY(data, projection, spec)
    assert key not in baseline
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    repaired = paved._ORIGINAL_GEOMETRY(data, projection, spec)
    assert key in repaired
    assert len(repaired[key].directions) == 3
    assert key not in quality._junction_geometry(data, projection, spec)


def test_town_fitter_accepts_repaired_road_network(monkeypatch):
    data, projection, spec = _town()
    heights = [0.0] * (spec.cells * spec.cells)
    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    baseline = playability.fit_road_objects(data, projection, heights, spec)
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    repaired = playability.fit_road_objects(data, projection, heights, spec)
    assert repaired.objects
    assert not repaired.truncated
    assert repaired.junction_cap_objects == baseline.junction_cap_objects + 1
    assert repaired.failed_connections <= baseline.failed_connections
    print(
        "Town A/B:",
        "control=", (len(baseline.objects), baseline.junction_cap_objects,
                     baseline.failed_connections, baseline.maximum_connection_gap),
        "repair=", (len(repaired.objects), repaired.junction_cap_objects,
                    repaired.failed_connections, repaired.maximum_connection_gap),
    )



def test_town_four_way_crossing_builds_one_real_junction_cap(monkeypatch):
    from dataclasses import replace
    from cwr_worldgen import paved_junction_policy as paved

    data, projection, spec = _town()
    cross = OsmLineFeature(
        "way/cross", data.roads[0].tags,
        tuple(projection.to_latlon(point) for point in (
            (500.0, 300.0), (500.0, 700.0),
        )),
    )
    data = replace(data, roads=(data.roads[0], cross))
    key = playability._road_node_key((500.0, 500.0))
    heights = [0.0] * (spec.cells * spec.cells)

    monkeypatch.delenv("CWR_ROAD_PAINT_SNAP_METRES", raising=False)
    assert key not in paved._ORIGINAL_GEOMETRY(data, projection, spec)
    baseline = playability.fit_road_objects(data, projection, heights, spec)

    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    geometry = paved._ORIGINAL_GEOMETRY(data, projection, spec)
    assert key in geometry
    assert len(geometry[key].directions) == 4
    repaired = playability.fit_road_objects(data, projection, heights, spec)
    assert repaired.junction_cap_objects == baseline.junction_cap_objects + 1
    assert repaired.failed_connections <= baseline.failed_connections
    assert not repaired.truncated
    print(
        "Town X A/B:",
        "control=", (len(baseline.objects), baseline.junction_cap_objects,
                     baseline.failed_connections),
        "repair=", (len(repaired.objects), repaired.junction_cap_objects,
                    repaired.failed_connections),
    )



def test_duplicate_city_streets_fit_once_without_stacked_pavement(monkeypatch):
    from dataclasses import replace

    data, projection, spec = _town()
    paved_tags = {"highway": "residential", "surface": "asphalt"}

    def street(key, offset):
        return OsmLineFeature(
            key, paved_tags,
            tuple(projection.to_latlon(point) for point in (
                (250.0, 500.0 + offset),
                (500.0, 500.0 + offset),
                (750.0, 500.0 + offset),
            )),
        )

    primary = street("way/city-avenue", 0.0)
    duplicate = street("way/city-avenue-duplicate", 0.30)
    single = replace(data, roads=(primary,))
    doubled = replace(data, roads=(primary, duplicate))
    heights = [0.0] * (spec.cells * spec.cells)

    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    projected = playability._paved_junction_augmented_polylines(
        doubled, projection, spec,
    )
    assert sum(len(line) >= 2 for line in projected) == 1
    single_fit = playability.fit_road_objects(
        single, projection, heights, spec,
    )
    doubled_fit = playability.fit_road_objects(
        doubled, projection, heights, spec,
    )
    assert len(doubled_fit.objects) == len(single_fit.objects)
    assert doubled_fit.junction_cap_objects == single_fit.junction_cap_objects
    assert doubled_fit.failed_connections <= single_fit.failed_connections
    print("City duplicate A/B:", "single=", len(single_fit.objects),
          "duplicate=", len(doubled_fit.objects))



def test_near_duplicate_city_avenue_does_not_split_real_t_junction_twice(monkeypatch):
    from dataclasses import replace

    data, projection, spec = _town()
    tags = {"highway": "residential", "surface": "asphalt"}

    def way(key, points):
        return OsmLineFeature(
            key, tags, tuple(projection.to_latlon(point) for point in points),
        )

    main = way("way/main", ((300.0, 500.0), (700.0, 500.0)))
    double = way("way/duplicate", ((300.0, 500.3), (700.0, 500.3)))
    branch = way("way/branch", ((500.0, 300.0), (500.0, 500.0)))
    single = replace(data, roads=(main, branch))
    double_set = replace(data, roads=(main, double, branch))
    monkeypatch.setenv("CWR_ROAD_PAINT_SNAP_METRES", "2.5")
    projected = playability._paved_junction_augmented_polylines(
        double_set, projection, spec,
    )
    assert len(projected[1]) < 2
    key = playability._road_node_key((500.0, 500.0))
    assert key in {playability._road_node_key(point) for point in projected[0]}
    assert key in {playability._road_node_key(point) for point in projected[2]}

    heights = [0.0] * (spec.cells * spec.cells)
    baseline = playability.fit_road_objects(single, projection, heights, spec)
    result = playability.fit_road_objects(double_set, projection, heights, spec)
    assert result.junction_cap_objects == baseline.junction_cap_objects
    assert result.failed_connections <= baseline.failed_connections
    assert len(result.objects) <= len(baseline.objects) + 1
    print("City T duplicate A/B:", "single=", len(baseline.objects),
          "duplicate=", len(result.objects))
