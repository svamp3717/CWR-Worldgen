from pathlib import Path
import math

import cwr_worldgen.gravel_junction_policy as gravel_policy
import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as road_quality
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
)


def _gravel_feature(projection, osm_key: str, points):
    return OsmLineFeature(
        osm_key,
        {"highway": "track", "surface": "gravel"},
        tuple(projection.to_latlon(point) for point in points),
    )


def test_policy_is_layered_on_top_of_general_road_quality_policy() -> None:
    assert road_quality._junction_geometry is gravel_policy._junction_geometry
    assert road_quality._exit_distance is gravel_policy._exit_distance
    assert road_quality._quality_window is gravel_policy._quality_window
    assert playability.fit_road_objects.__module__ == "cwr_worldgen.road_quality_policy"


def test_four_way_diagonal_uses_plus_shape_not_bounding_square() -> None:
    diagonal = (math.sqrt(0.5), math.sqrt(0.5))
    distance = gravel_policy.gravel_hub_exit_distance(
        (0.0, 1.0),
        diagonal,
        extent=3.0,
        half_width=GENERATED_GRAVEL_HALF_WIDTH_METRES,
    )
    assert math.isclose(
        distance,
        GENERATED_GRAVEL_HALF_WIDTH_METRES / math.sqrt(0.5),
        abs_tol=1.0e-9,
    )
    # The previous 3x3 bounding-square approximation exits at ~4.24 m.
    assert distance < 3.30
    assert 4.20 < 3.0 / math.sqrt(0.5) < 4.25


def test_skewed_gravel_cross_uses_real_hub_footprint_and_hidden_overlap() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    vertical = _gravel_feature(
        projection,
        "way/vertical",
        ((500.0, 260.0), centre, (500.0, 740.0)),
    )
    diagonal = _gravel_feature(
        projection,
        "way/diagonal",
        ((330.0, 330.0), centre, (670.0, 670.0)),
    )
    dataset = OsmDataset(
        source_generator="gravel-junction-gap",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(vertical, diagonal),
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

    junctions = road_quality._junction_geometry(dataset, projection, spec)
    junction = junctions[playability._road_node_key(centre)]
    assert math.isclose(junction.half_width, GENERATED_GRAVEL_HALF_WIDTH_METRES, abs_tol=1e-9)
    assert math.isclose(junction.half_length, 3.0, abs_tol=1e-9)

    diagonal_direction = (math.sqrt(0.5), math.sqrt(0.5))
    exit_distance = road_quality._exit_distance(junction, diagonal_direction)
    assert math.isclose(
        exit_distance,
        GENERATED_GRAVEL_HALF_WIDTH_METRES / math.sqrt(0.5),
        abs_tol=1e-6,
    )

    report = playability.fit_road_objects(
        dataset,
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )
    assert report.junction_cap_objects == 1
    assert report.objects[0].model_path.casefold().endswith(r"\gravel_j4.p3d")
    assert report.failed_connections == 0
    assert report.maximum_connection_gap <= spec.road_connection_tolerance

    branch_objects = []
    for obj in report.objects[report.junction_cap_objects :]:
        heading_error = min(
            abs(((obj.heading_degrees - 45.0 + 180.0) % 360.0) - 180.0),
            abs(((obj.heading_degrees - 225.0 + 180.0) % 360.0) - 180.0),
        )
        if heading_error <= 2.0:
            branch_objects.append(obj)
    assert branch_objects

    inner_axis_distance = math.inf
    for obj in branch_objects:
        length = road_quality._piece_length(obj.model_path, spec.road_segment_length)
        axis = playability._model_axis(obj, length)
        inner_axis_distance = min(
            inner_axis_distance,
            math.dist(centre, axis[0]),
            math.dist(centre, axis[1]),
        )

    # The axis itself should now extend well under the true gravel hub. The
    # generated ribbon adds another 0.90 m lowered visual tip, making a naked
    # centreline seam impossible without introducing a separate patch object.
    axis_overlap = exit_distance - inner_axis_distance
    assert axis_overlap >= 0.60
    visual_overlap = axis_overlap + GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
    assert visual_overlap >= 1.45
