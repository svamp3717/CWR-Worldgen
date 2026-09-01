from pathlib import Path
import math

import cwr_worldgen.gravel_family_policy as gravel_policy
import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as road_quality
from cwr_worldgen.gravel_family_policy import (
    GRAVEL_JUNCTION_ARM_EXTENT_METRES,
    GRAVEL_JUNCTION_VARIANTS,
    gravel_junction_template_headings,
    gravel_junction_variant_for_directions,
)
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
    InfrastructureModelKey,
    _road_lods,
)


def _gravel_feature(projection, osm_key: str, points):
    return OsmLineFeature(
        osm_key,
        {"highway": "track", "surface": "gravel"},
        tuple(projection.to_latlon(point) for point in points),
    )


def _heading_direction(heading):
    angle = math.radians(heading)
    return (math.sin(angle), math.cos(angle))


def test_fixed_gravel_junction_catalogue_has_fifteen_reusable_shapes() -> None:
    assert len(GRAVEL_JUNCTION_VARIANTS) == 15
    for variant in GRAVEL_JUNCTION_VARIANTS:
        headings = gravel_junction_template_headings(variant)
        degree = 4 if variant.startswith("x") else 3
        assert len(headings) == degree
        subtype = f"gravel_j{degree}_{variant}"
        visual, _map_geometry, roadway, _land = _road_lods(
            InfrastructureModelKey("road", subtype, 46, 80),
            r"cwr_family\i\g.paa",
        )
        assert visual.faces
        assert roadway.faces
        assert max(math.hypot(x, z) for x, _y, z in visual.points) > 3.5


def test_uploaded_lundby_junction_selects_reusable_60_degree_t_model() -> None:
    bbox = (59.4012347, 16.8211928, 59.4587912, 16.9343614)
    projection = BboxProjection.create(bbox, 6400.0)
    centre = (3943.500087480133, 2295.750436527429)
    southwest = (3840.0, 2096.249771962806)
    northeast = (4098.0000459490875, 2595.7503322819402)
    west_branch = (3898.4999019163347, 2288.9997828219803)
    dataset = OsmDataset(
        source_generator="uploaded-lundby-regression",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            _gravel_feature(projection, "way/main", (southwest, centre, northeast)),
            _gravel_feature(projection, "way/branch", (centre, west_branch)),
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="gravel_family",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=128,
        cell_size=50.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=True,
    )
    junction = road_quality._junction_geometry(dataset, projection, spec)[
        playability._road_node_key(centre)
    ]
    variant, _axis = gravel_junction_variant_for_directions(junction.directions)
    assert variant in {"t60l", "t60r"}
    assert math.isclose(
        junction.half_width,
        GENERATED_GRAVEL_HALF_WIDTH_METRES,
        abs_tol=1e-9,
    )
    assert math.isclose(
        junction.half_length,
        GRAVEL_JUNCTION_ARM_EXTENT_METRES,
        abs_tol=1e-9,
    )
    for direction in junction.directions:
        assert road_quality._exit_distance(junction, direction) >= 3.9
    report = playability.fit_road_objects(
        dataset, projection, [0.0] * (spec.cells * spec.cells), spec
    )
    assert report.junction_cap_objects == 1
    cap = report.objects[0]
    assert cap.model_path.casefold().endswith(rf"\gravel_j3_{variant}.p3d")
    assert report.failed_connections == 0
    assert report.maximum_connection_gap <= spec.road_connection_tolerance
    assert not any(
        obj.model_path.casefold().endswith(r"\gravel3.p3d")
        and obj.y > -0.020
        and math.dist((obj.x, obj.z), centre) < 5.0
        for obj in report.objects
    )


def test_right_angle_t_and_45_degree_cross_select_standard_family_members() -> None:
    t_variant, _axis = gravel_junction_variant_for_directions(
        (_heading_direction(0), _heading_direction(180), _heading_direction(90))
    )
    assert t_variant == "t90"
    x_variant, _axis = gravel_junction_variant_for_directions(
        tuple(_heading_direction(value) for value in (0, 180, 45, 225))
    )
    assert x_variant == "x45"


def test_bucketed_arm_still_keeps_hidden_overlap_for_54_degree_source_branch() -> None:
    directions = tuple(_heading_direction(value) for value in (207.42, 27.25, 261.47))
    variant, axis = gravel_junction_variant_for_directions(directions)
    assert variant in {"t60l", "t60r"}
    junction = road_quality._Junction(
        point=(0.0, 0.0),
        axis=axis,
        half_length=GRAVEL_JUNCTION_ARM_EXTENT_METRES,
        half_width=GENERATED_GRAVEL_HALF_WIDTH_METRES,
        directions=directions,
    )
    branch = _heading_direction(261.47)
    exit_distance = gravel_policy._exit_distance(junction, branch)
    assert exit_distance >= 3.9
    assert 0.70 + GENERATED_GRAVEL_VISUAL_OVERLAP_METRES >= 1.5


def test_skew_three_way_family_rotation_minimizes_all_arm_errors() -> None:
    headings = (0.0, 98.0, 249.0)
    directions = tuple(_heading_direction(value) for value in headings)
    variant, axis = gravel_junction_variant_for_directions(directions)
    assert variant == "t90"
    axis_heading = math.degrees(math.atan2(axis[0], axis[1])) % 360.0
    model_headings = tuple(
        (axis_heading + value) % 360.0
        for value in gravel_junction_template_headings(variant)
    )

    def angular_error(left, right):
        return abs((left - right + 180.0) % 360.0 - 180.0)

    best_maximum = min(
        max(
            angular_error(model, actual)
            for model, actual in zip(model_headings, ordering)
        )
        for ordering in __import__("itertools").permutations(headings)
    )
    assert best_maximum <= 21.0


def test_orthogonal_family_hubs_use_full_length_family_models() -> None:
    assert gravel_policy.gravel_junction_model_path(
        "synthetic", 3, "t90"
    ).endswith(r"\gravel_j3.p3d")
    assert gravel_policy.gravel_junction_model_path(
        "synthetic", 4, "x90"
    ).endswith(r"\gravel_j4.p3d")
    for degree, variant in ((3, "t90"), (4, "x90")):
        subtype = f"gravel_j{degree}"
        visual, _map_geometry, roadway, _land = _road_lods(
            InfrastructureModelKey("road", subtype, 46, 80),
            r"synthetic\i\g.paa",
        )
        visual_radius = max(math.hypot(x, z) for x, _y, z in visual.points)
        roadway_radius = max(math.hypot(x, z) for x, _y, z in roadway.points)
        assert visual_radius >= GRAVEL_JUNCTION_ARM_EXTENT_METRES
        assert roadway_radius >= GRAVEL_JUNCTION_ARM_EXTENT_METRES


def test_short_aligned_gap_between_gravel_endpoints_gets_a_small_filler() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
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
        obj
        for obj in report.objects
        if obj.model_path.casefold().endswith(r"\gravel3.p3d")
        and 505.5 <= obj.x <= 506.5
        and abs(obj.z - 500.0) <= 0.1
    ]
    assert len(fillers) == 1
    assert abs(
        ((fillers[0].heading_degrees - 90.0 + 180.0) % 360.0) - 180.0
    ) <= 0.1

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
        and obj.model_path.casefold().endswith(
            (r"\gravel3.p3d", r"\gravel6.p3d")
        )
        for obj in report.objects
    )
