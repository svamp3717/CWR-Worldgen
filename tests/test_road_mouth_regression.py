from __future__ import annotations

import math
from pathlib import Path

import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

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
    if signature is not None:
        _surface, width, length, _curve = signature
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


def _fit(paved_model, roads):
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
    dimensions = {} if key.startswith("o\\road\\") else {key: (7.0, 25.0)}
    tokens = [
        (playability._ROAD_MODEL_DIMENSIONS,
         playability._ROAD_MODEL_DIMENSIONS.set(dimensions)),
        (playability._ROAD_MODEL_VARIANTS_AVAILABLE,
         playability._ROAD_MODEL_VARIANTS_AVAILABLE.set({key: frozenset({key})})),
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
            if surface is not None:
                surfaces.setdefault(surface, []).append(_road_footprint(obj, spec))
        footprints = {surface: unary_union(values) for surface, values in surfaces.items()}
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
    return spec, report, footprints


@pytest.mark.parametrize("paved_model", [
    r"o\road\sil25.p3d", r"bas_o\_road\bas_asf25.p3d",
])
def test_bent_paved_road_has_no_grass_wedges(paved_model):
    points = ((300.0, 300.0), (300.0, 380.0), (380.0, 380.0))
    spec, report, footprints = _fit(paved_model, [
        ({"highway": "residential", "surface": "asphalt"}, points),
    ])
    pieces = playability.road_model_variants(
        paved_model, spec.road_segment_length, donor_only=True
    )
    expected = LineString(playability._representable_road_run(points, pieces))
    width = 9.1 if paved_model.startswith("o\\") else 7.0
    corridor = expected.buffer(width * 0.40, cap_style="flat")
    uncovered = corridor.difference(footprints["paved"])
    print("BEND", paved_model, "uncovered", uncovered.area,
          "objects", [(obj.model_path, obj.x, obj.z, obj.heading_degrees) for obj in report.objects])
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
    print("MIXED", paved_model, angle, "uncovered", uncovered.area,
          "objects", [(obj.model_path, obj.x, obj.z, obj.heading_degrees) for obj in report.objects
                      if math.dist((obj.x, obj.z), node) < 25.0])
    assert uncovered.area <= 0.10

@pytest.mark.parametrize("paved_model", [
    r"o\road\sil25.p3d", r"bas_o\_road\bas_asf25.p3d",
])
@pytest.mark.parametrize("angle", [45.0, 90.0, 135.0])
def test_gravel_arm_reaches_paved_three_way_hub(paved_model, angle):
    node = (500.0, 500.0)
    heading = math.radians(angle)
    direction = math.sin(heading), math.cos(heading)
    end = (node[0] + direction[0] * 100.0, node[1] + direction[1] * 100.0)
    _spec, report, footprints = _fit(paved_model, [
        ({"highway": "residential", "surface": "asphalt"},
         ((500.0, 400.0), node, (500.0, 600.0))),
        ({"highway": "residential", "surface": "asphalt"},
         (node, (400.0, 500.0))),
        ({"highway": "track", "surface": "gravel"}, (node, end)),
    ])
    expected = LineString([node, (node[0] + direction[0] * 15.0,
                                 node[1] + direction[1] * 15.0)])
    corridor = expected.buffer(1.9, cap_style="flat")
    uncovered = corridor.difference(unary_union(tuple(footprints.values())))
    print("HUB", paved_model, angle, "uncovered", uncovered.area,
          "objects", [(obj.model_path, obj.x, obj.z, obj.heading_degrees) for obj in report.objects
                      if math.dist((obj.x, obj.z), node) < 25.0])
    assert uncovered.area <= 0.10
