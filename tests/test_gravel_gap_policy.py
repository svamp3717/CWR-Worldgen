from pathlib import Path

import cwr_worldgen.playability as playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.procedural_infrastructure import GENERATED_GRAVEL_VISUAL_OVERLAP_METRES


def _gravel_feature(projection, osm_key: str, points):
    return OsmLineFeature(
        osm_key,
        {"highway": "track", "surface": "gravel"},
        tuple(projection.to_latlon(point) for point in points),
    )


def test_short_aligned_gap_between_gravel_endpoints_gets_a_small_filler() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    # The junction way ends at x=503, while a separately normalized continuation
    # starts at x=509. This is the visual pattern reported from CWA: a clean hub,
    # then several metres of grass, then the next gravel ribbon.
    main = _gravel_feature(
        projection,
        "way/main",
        ((300.0, 500.0), centre, (503.0, 500.0)),
    )
    branch = _gravel_feature(
        projection,
        "way/branch",
        (centre, (500.0, 700.0)),
    )
    continuation = _gravel_feature(
        projection,
        "way/continuation",
        ((509.0, 500.0), (750.0, 500.0)),
    )
    dataset = OsmDataset(
        source_generator="gravel-endpoint-gap",
        element_count=3,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch, continuation),
    )
    spec = _Milestone9PlayabilitySpec(
        name="gravel_gap",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=True,
    )

    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )
    fillers = [
        obj for obj in report.objects
        if obj.model_path.casefold().endswith(r"\gravel3.p3d")
        and 505.5 <= obj.x <= 506.5
        and abs(obj.z - 500.0) <= 0.1
    ]
    assert len(fillers) == 1
    assert abs(((fillers[0].heading_degrees - 90.0 + 180.0) % 360.0) - 180.0) <= 0.1

    # The 3 m filler plus the lowered visual tips participating in the seams is
    # enough to cover the six-metre source-data hole without inserting a long
    # overlapping slab.
    actual_length = spec.road_segment_length * 3.0 / 25.0
    assert actual_length + 4.0 * GENERATED_GRAVEL_VISUAL_OVERLAP_METRES > 6.0


def test_detached_gravel_endpoint_can_join_the_unused_arm_of_a_nearby_t_hub() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    vertical = _gravel_feature(
        projection,
        "way/vertical",
        ((500.0, 300.0), centre, (500.0, 700.0)),
    )
    left = _gravel_feature(projection, "way/left", ((300.0, 500.0), centre))
    detached_right = _gravel_feature(
        projection,
        "way/right",
        ((507.0, 500.0), (750.0, 500.0)),
    )
    dataset = OsmDataset(
        source_generator="gravel-t-gap",
        element_count=3,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(vertical, left, detached_right),
    )
    spec = _Milestone9PlayabilitySpec(
        name="gravel_gap",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=True,
    )

    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )
    assert any(
        obj.model_path.casefold().endswith(r"\gravel6.p3d")
        and 502.5 <= obj.x <= 504.5
        and abs(obj.z - 500.0) <= 0.1
        for obj in report.objects
    )


def test_short_gravel_gap_repair_does_not_bridge_misaligned_dead_ends() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    first = _gravel_feature(
        projection,
        "way/first",
        ((300.0, 500.0), (500.0, 500.0)),
    )
    # Close in distance, but its endpoint points north rather than back toward
    # the first road. This should remain two separate dead ends.
    second = _gravel_feature(
        projection,
        "way/second",
        ((506.0, 500.0), (506.0, 700.0)),
    )
    dataset = OsmDataset(
        source_generator="gravel-misaligned-gap",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(first, second),
    )
    spec = _Milestone9PlayabilitySpec(
        name="gravel_gap",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=True,
    )

    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )
    assert not any(
        502.0 <= obj.x <= 504.0
        and abs(obj.z - 500.0) <= 0.5
        and obj.model_path.casefold().endswith((r"\gravel3.p3d", r"\gravel6.p3d"))
        for obj in report.objects
    )
