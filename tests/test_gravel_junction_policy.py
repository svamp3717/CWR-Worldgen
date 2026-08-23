from pathlib import Path
import math

import cwr_worldgen.gravel_junction_policy as gravel_policy
import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as road_quality
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.gravel_family_policy import (
    GRAVEL_JUNCTION_ARM_EXTENT_METRES,
    GRAVEL_JUNCTION_VARIANTS,
    gravel_junction_template_headings,
    gravel_junction_variant_for_directions,
)
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


def _point_from_heading(origin, heading_degrees, distance):
    angle = math.radians(heading_degrees)
    return (
        origin[0] + math.sin(angle) * distance,
        origin[1] + math.cos(angle) * distance,
    )


def _heading_direction(heading):
    angle = math.radians(heading)
    return (math.sin(angle), math.cos(angle))


def test_policy_is_layered_on_top_of_general_road_quality_policy() -> None:
    assert road_quality._junction_geometry is gravel_policy._junction_geometry
    assert road_quality._exit_distance is gravel_policy._exit_distance
    assert road_quality._quality_window is gravel_policy._quality_window
    # functools.wraps preserves the wrapped road-quality module metadata even
    # though the short disconnected-endpoint repair remains the outer wrapper.
    assert playability.fit_road_objects.__module__ == "cwr_worldgen.gravel_family_policy"


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
    # Exact normalized geometry from the user's Lundby source around player
    # position [3950, 35, 2292]. The branch is ~54 degrees from the main road,
    # so the fixed family should reuse the nearest 60-degree T model.
    bbox = (59.4012347, 16.8211928, 59.4587912, 16.9343614)
    projection = BboxProjection.create(bbox, 6400.0)
    centre = (3943.500087480133, 2295.750436527429)
    southwest = (3840.0, 2096.249771962806)
    northeast = (4098.0000459490875, 2595.7503322819402)
    west_branch = (3898.4999019163347, 2288.9997828219803)
    dataset = OsmDataset(
        source_generator="uploaded-lundby-regression",
        element_count=2,
        coastlines=(), water=(), forests=(), farmland=(), urban=(),
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
    assert math.isclose(junction.half_width, GENERATED_GRAVEL_HALF_WIDTH_METRES, abs_tol=1e-9)
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
    assert cap.model_path.casefold().endswith(
        rf"\gravel_j3_{variant}.p3d"
    )
    assert report.failed_connections == 0
    assert report.maximum_connection_gap <= spec.road_connection_tolerance
    # The old acute-corner workaround inserted a raised gravel3 patch. The new
    # family must cover the junction with one reusable hub instead.
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
    # The physical family model reaches about four metres along this bucketed
    # branch. Incoming ribbons then retain 0.70 m of centreline overlap plus the
    # existing 0.90 m lowered visual tip.
    assert exit_distance >= 3.9
    assert 0.70 + GENERATED_GRAVEL_VISUAL_OVERLAP_METRES >= 1.5


def test_skew_three_way_family_rotation_minimizes_all_arm_errors() -> None:
    # Synthetic copy of a problematic shape, shifted/rotated away from any real
    # map coordinates. The gaps are about 98, 151 and 111 degrees, so there is
    # no genuinely straight main-road pair to use as a rotation anchor.
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
        max(angular_error(model, actual) for model, actual in zip(model_headings, ordering))
        for ordering in __import__("itertools").permutations(headings)
    )
    # The reusable 90-degree T can cover this skewed geometry when it is rotated
    # to balance all three arms, instead of being pinned to one imperfectly
    # opposite road pair.
    assert best_maximum <= 21.0
