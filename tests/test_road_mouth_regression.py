from __future__ import annotations

import math
from pathlib import Path

import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import substring, unary_union

from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _world_polygon(obj, points):
    angle = math.radians(obj.heading_degrees)
    right = math.cos(angle), -math.sin(angle)
    forward = math.sin(angle), math.cos(angle)
    return Polygon([
        (
            obj.x + right[0] * x + forward[0] * z,
            obj.z + right[1] * x + forward[1] * z,
        )
        for x, z in points
    ])


def _road_footprint(obj, spec):
    signature = infrastructure.custom_road_model_signature(obj.model_path)
    junction = infrastructure.custom_road_junction_signature(obj.model_path)
    if signature is not None or junction is not None:
        if signature is not None:
            _surface, width, length, _curve = signature
        else:
            _surface, width, _headings = junction
            length = infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 2.0
        subtype = obj.model_path.replace("/", "\\").rsplit("\\", 1)[-1][:-4]
        key = infrastructure.InfrastructureModelKey(
            "road", subtype, round(width * 10), round(length * 10)
        )
        lod = infrastructure._road_lods(key, "unused.paa")[0]
        return unary_union([
            _world_polygon(
                obj,
                [(lod.points[corner[0]][0], lod.points[corner[0]][2])
                 for corner in face.vertices],
            )
            for face in lod.faces
        ])
    width = playability.road_model_width_metres(obj.model_path)
    if width is None:
        return Polygon()
    piece = playability.road_model_variants(
        obj.model_path, spec.road_segment_length, donor_only=True
    )[0]
    half_length = piece.length_metres * 0.5
    return _world_polygon(obj, [
        (-width * 0.5, -half_length), (width * 0.5, -half_length),
        (width * 0.5, half_length), (-width * 0.5, half_length),
    ])


def _fit(paved_model, roads, *, available_siblings=True):
    bbox = (59.40, 16.82, 59.41, 16.83)
    projection = BboxProjection.create(bbox, 1000.0)
    dataset = OsmDataset(
        source_generator="road-mouth-regression",
        element_count=len(roads),
        coastlines=(), water=(), forests=(), farmland=(), urban=(),
        roads=tuple(
            OsmLineFeature(
                f"way/{index}", tags,
                tuple(projection.to_latlon(point) for point in points),
            )
            for index, (tags, points) in enumerate(roads)
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="seamtest", heightmap_path=Path("unused.png"), bbox=bbox,
        cells=40, cell_size=25.0, max_road_objects=10000,
        strict_assets=False, paved_road_model=paved_model,
    )
    key = playability._road_model_key(paved_model)
    if key.startswith("o\\road\\"):
        dimensions = {}
        available = frozenset({key})
    elif available_siblings:
        sibling_paths = tuple(
            paved_model[:-len("25.p3d")] + suffix
            for suffix in ("25.p3d", "12.p3d", "6.p3d")
        )
        sibling_keys = tuple(
            playability._road_model_key(path)
            for path in sibling_paths
        )
        dimensions = {
            sibling_keys[0]: (7.0, 25.0),
            sibling_keys[1]: (7.0, 12.5),
            sibling_keys[2]: (7.0, 6.25),
        }
        available = frozenset(sibling_keys)
    else:
        dimensions = {key: (7.0, 25.0)}
        available = frozenset({key})
    tokens = [
        (playability._ROAD_MODEL_DIMENSIONS,
         playability._ROAD_MODEL_DIMENSIONS.set(dimensions)),
        (playability._ROAD_MODEL_VARIANTS_AVAILABLE,
         playability._ROAD_MODEL_VARIANTS_AVAILABLE.set({key: available})),
        (playability._ROAD_MODEL_EFFECTIVE_DONORS,
         playability._ROAD_MODEL_EFFECTIVE_DONORS.set({key: paved_model})),
    ]
    try:
        report = playability.fit_road_objects(
            dataset, projection, [0.0] * (spec.cells * spec.cells), spec
        )
        surfaces = {}
        for obj in report.objects:
            surface = playability.road_model_surface(spec, obj.model_path)
            junction = infrastructure.custom_road_junction_signature(obj.model_path)
            if junction is not None:
                surface = junction[0]
            if surface is not None:
                surfaces.setdefault(surface, []).append(_road_footprint(obj, spec))
        footprints = {surface: unary_union(values) for surface, values in surfaces.items()}
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
    return spec, report, footprints


@pytest.mark.parametrize(("paved_model", "available_siblings"), [
    (r"o\road\sil25.p3d", True),
    (r"bas_o\_road\bas_asf25.p3d", True),
    (r"bas_o\_road\bas_asf25.p3d", False),
])
def test_bent_paved_road_has_no_grass_wedges(paved_model, available_siblings):
    points = ((300.0, 300.0), (300.0, 380.0), (380.0, 380.0))
    spec, report, footprints = _fit(paved_model, [
        ({"highway": "residential", "surface": "asphalt"}, points),
    ], available_siblings=available_siblings)
    pieces = playability.road_model_variants(
        paved_model, spec.road_segment_length, donor_only=True
    )
    expected = LineString(playability._representable_road_run(points, pieces))
    width = 9.1 if paved_model.startswith("o\\") else 7.0
    # This check targets interior wedges; endpoint rounding has its own
    # connection tolerance and must not count as a bend seam failure.
    corridor = substring(expected, 0.5, expected.length - 0.5).buffer(
        width * 0.40, cap_style="flat"
    )
    uncovered = corridor.difference(footprints["paved"])
    assert uncovered.area <= 0.10


@pytest.mark.parametrize("paved_model", [
    r"o\road\sil25.p3d", r"bas_o\_road\bas_asf25.p3d",
])
@pytest.mark.parametrize("angle", [30.0, 60.0, 90.0])
def test_angled_paved_gravel_transition_covers_both_mouths(paved_model, angle):
    node = (500.0, 500.0)
    heading = math.radians(angle)
    direction = math.sin(heading), math.cos(heading)
    end = (node[0] + direction[0] * 100.0, node[1] + direction[1] * 100.0)
    _spec, report, footprints = _fit(paved_model, [
        ({"highway": "residential", "surface": "asphalt"},
         ((500.0, 400.0), node)),
        ({"highway": "track", "surface": "gravel"}, (node, end)),
    ])
    # Check the visible track width through the transition, including its outer
    # corners. Coincident centreline endpoints alone cannot prove a closed seam.
    expected = LineString([node, (node[0] + direction[0] * 10.0,
                                 node[1] + direction[1] * 10.0)])
    corridor = expected.buffer(1.9, cap_style="flat")
    uncovered = corridor.difference(unary_union(tuple(footprints.values())))
    assert uncovered.area <= 0.10

@pytest.mark.parametrize("paved_model", [
    r"o\road\sil25.p3d", r"bas_o\_road\bas_asf25.p3d",
])
@pytest.mark.parametrize("angle", [45.0, 90.0, 135.0])
@pytest.mark.parametrize("reverse", (False, True))
def test_gravel_arm_reaches_paved_three_way_hub(paved_model, angle, reverse):
    node = (500.0, 500.0)
    heading = math.radians(angle)
    direction = math.sin(heading), math.cos(heading)
    end = (node[0] + direction[0] * 100.0, node[1] + direction[1] * 100.0)
    _spec, report, footprints = _fit(paved_model, [
        ({"highway": "residential", "surface": "asphalt"},
         ((500.0, 400.0), node, (500.0, 600.0))),
        ({"highway": "residential", "surface": "asphalt"},
         (node, (400.0, 500.0))),
        ({"highway": "track", "surface": "gravel"},
         (end, node) if reverse else (node, end)),
    ])
    expected = LineString([node, (node[0] + direction[0] * 15.0,
                                 node[1] + direction[1] * 15.0)])
    corridor = expected.buffer(1.9, cap_style="flat")
    uncovered = corridor.difference(unary_union(tuple(footprints.values())))
    assert uncovered.area <= 0.10

@pytest.mark.parametrize(("paved_model", "available_siblings"), [
    (r"o\road\sil25.p3d", True),
    (r"bas_o\_road\bas_asf25.p3d", True),
    (r"bas_o\_road\bas_asf25.p3d", False),
])
@pytest.mark.parametrize("turn_angle", [30.0, 60.0, 90.0])
@pytest.mark.parametrize("reverse_gravel", (False, True))
def test_angled_paved_bend_with_gravel_branch_has_no_wedge(
    paved_model,
    available_siblings,
    turn_angle,
    reverse_gravel,
):
    node = (500.0, 500.0)
    paved_heading = math.radians(turn_angle)
    paved_direction = math.sin(paved_heading), math.cos(paved_heading)
    paved_end = (
        node[0] + paved_direction[0] * 100.0,
        node[1] + paved_direction[1] * 100.0,
    )
    gravel_end = (node[0] - 100.0, node[1])
    paved_points = ((500.0, 400.0), node, paved_end)
    gravel_points = (
        (gravel_end, node) if reverse_gravel else (node, gravel_end)
    )

    spec, report, footprints = _fit(
        paved_model,
        [
            (
                {"highway": "residential", "surface": "asphalt"},
                paved_points,
            ),
            (
                {"highway": "track", "surface": "gravel"},
                gravel_points,
            ),
        ],
        available_siblings=available_siblings,
    )

    pieces = playability.road_model_variants(
        paved_model, spec.road_segment_length, donor_only=True
    )
    paved_centreline = LineString(
        playability._representable_road_run(paved_points, pieces)
    )
    paved_width = 9.1 if paved_model.startswith("o\\") else 7.0
    paved_corridor = substring(
        paved_centreline,
        0.5,
        paved_centreline.length - 0.5,
    ).buffer(paved_width * 0.40, cap_style="flat")

    gravel_centreline = LineString(
        [node, (node[0] - 15.0, node[1])]
    )
    gravel_corridor = gravel_centreline.buffer(1.9, cap_style="flat")
    expected = unary_union((paved_corridor, gravel_corridor))
    actual = unary_union(tuple(footprints.values()))
    uncovered = expected.difference(actual)

    assert report.junction_cap_objects == 0
    assert uncovered.area <= 0.10

@pytest.mark.parametrize(("paved_model", "available_siblings"), [
    (r"o\road\sil25.p3d", True),
    (r"bas_o\_road\bas_asf25.p3d", True),
])
def test_unified_fit_never_emits_external_short_road_siblings(
    paved_model,
    available_siblings,
):
    _spec, report, _footprints = _fit(
        paved_model,
        [
            (
                {"highway": "residential", "surface": "asphalt"},
                ((300.0, 300.0), (300.0, 360.0), (345.0, 390.0)),
            ),
        ],
        available_siblings=available_siblings,
    )
    external = {
        obj.model_path.casefold()
        for obj in report.objects
        if not obj.model_path.casefold().startswith("seamtest\\i\\")
    }
    assert external <= {paved_model.casefold()}
    assert not any(
        path.endswith(("12.p3d", "6.p3d"))
        for path in external
    )

