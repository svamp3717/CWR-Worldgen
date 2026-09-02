from pathlib import Path
from types import SimpleNamespace
import math
import re

import cwr_worldgen.generator as generator
import cwr_worldgen.paved_junction_policy as paved_junctions
import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as road_quality
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.road_quality_policy import (
    _CONTEXT,
    _Junction,
    _Context,
    _exit_distance,
    _quality_window,
)


def _flat_context(*, cells: int = 4, cell_size: float = 25.0):
    spec = SimpleNamespace(
        cells=cells,
        cell_size=cell_size,
        road_connection_tolerance=0.35,
    )
    return _Context([0.0] * (cells * cells), spec, {})


def _angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    a = math.degrees(math.atan2(first[0], first[1]))
    b = math.degrees(math.atan2(second[0], second[1]))
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _junction_dataset(
    projection: BboxProjection,
    branch_end: tuple[float, float],
) -> OsmDataset:
    centre = (500.0, 500.0)
    main = OsmLineFeature(
        "way/main",
        {"highway": "residential"},
        tuple(
            projection.to_latlon(point)
            for point in ((500.0, 300.0), centre, (500.0, 700.0))
        ),
    )
    branch = OsmLineFeature(
        "way/branch",
        {"highway": "residential"},
        tuple(projection.to_latlon(point) for point in (centre, branch_end)),
    )
    return OsmDataset(
        source_generator="road-quality-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch),
    )


def _surface_junction_dataset(
    projection: BboxProjection,
    *,
    surface: str,
) -> OsmDataset:
    centre = (500.0, 500.0)
    tags = {"highway": "track", "surface": surface}
    return OsmDataset(
        source_generator=f"{surface}-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            OsmLineFeature(
                "way/main",
                tags,
                tuple(
                    projection.to_latlon(point)
                    for point in ((500.0, 300.0), centre, (500.0, 700.0))
                ),
            ),
            OsmLineFeature(
                "way/branch",
                tags,
                tuple(
                    projection.to_latlon(point)
                    for point in (centre, (700.0, 500.0))
                ),
            ),
        ),
    )


def _junction_spec(
    bbox: tuple[float, float, float, float],
    *,
    procedural_gravel_roads: bool = False,
) -> _Milestone9PlayabilitySpec:
    return _Milestone9PlayabilitySpec(
        name="road_quality",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        procedural_gravel_roads=procedural_gravel_roads,
    )


def _object_endpoints(obj):
    path = obj.model_path.casefold()
    curve = re.search(r"\\(sil|asf|kos)10 (25|50|75|100)\.p3d$", path)
    if curve:
        begin, end = paved_junctions._curve_points(
            curve.group(1), float(curve.group(2))
        )
        points = []
        for local in (begin, end):
            dx, dz = paved_junctions._rotate(local, obj.heading_degrees)
            points.append((obj.x + dx, obj.z + dz))
        return tuple(points)

    straight = re.search(r"\\(?:sil|asf|kos)(25|12|6)\.p3d$", path)
    if straight:
        length = {25: 25.0, 12: 12.5, 6: 6.25}[int(straight.group(1))]
        return playability._model_axis(obj, length)
    return ()


def test_policy_is_installed_for_playability_and_generator() -> None:
    assert playability.fit_road_objects is generator.fit_road_objects
    assert playability.fit_road_objects.__module__ == "cwr_worldgen.gravel_family_policy"


def test_road_type_catalogue_lists_reference_families_turns_and_paved_junctions() -> None:
    data = paved_junctions._catalogue()
    families = {
        str(value["name"]).casefold(): str(value["kind"]).casefold()
        for value in data["families"]
    }
    assert families == {
        "asfaltka": "paved",
        "cesta": "dirt",
        "silnice": "paved",
        "asf": "paved",
        "ces": "dirt",
        "kos": "paved",
        "sil": "paved",
        "gravel": "gravel",
    }
    junctions = {
        value.casefold() for value in paved_junctions.stock_junction_model_paths()
    }
    turns = {
        value.casefold() for value in paved_junctions.stock_turn_model_paths()
    }
    assert r"o\road\kr_new_sil_sil_t.p3d" in junctions
    assert r"o\road\kr_new_silxsil.p3d" in junctions
    assert not any("ces" in value for value in junctions)
    assert r"o\road\sil10 25.p3d" in turns
    assert r"o\road\asf10 100.p3d" in turns
    assert r"o\road\kos10 50.p3d" in turns


def test_stock_junction_plans_reject_dirt_gravel_and_mixed_nodes() -> None:
    mixed_dirt = paved_junctions._plan(
        (0.0, 0.0),
        (
            ((0.0, 1.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((1.0, 0.0), "ces"),
        ),
    )
    mixed_gravel = paved_junctions._plan(
        (0.0, 0.0),
        (
            ((0.0, 1.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((1.0, 0.0), "gravel"),
        ),
    )
    all_paved = paved_junctions._plan(
        (0.0, 0.0),
        (
            ((0.0, 1.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((1.0, 0.0), "asf"),
        ),
    )
    assert mixed_dirt is None
    assert mixed_gravel is None
    assert all_paved is not None
    assert "ces" not in all_paved.model_path.casefold()


def test_diagonal_junction_trim_uses_oriented_hub_edge() -> None:
    diagonal = (math.sqrt(0.5), math.sqrt(0.5))
    junction = _Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=3.0,
        half_width=3.0,
        directions=(diagonal,),
    )
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (20.0, 20.0), (40.0, 40.0))
    )
    pieces = (
        playability._RoadPiece(r"o\road\sil25.p3d", 25.0, 25),
        playability._RoadPiece(r"o\road\sil12.p3d", 12.0, 12),
        playability._RoadPiece(r"o\road\sil6.p3d", 6.0, 6),
    )
    context = _Context(
        [0.0] * 64,
        SimpleNamespace(cells=8, cell_size=25.0, road_connection_tolerance=0.35),
        {playability._road_node_key(measure.points[0]): junction},
    )
    adjusted = _quality_window(
        measure,
        pieces,
        2.30,
        measure.total,
        measure.total,
        measure.total,
        context,
    )
    exit_distance = _exit_distance(junction, diagonal)
    assert exit_distance > 4.2
    assert adjusted[0] > 3.9
    assert math.isclose(exit_distance - adjusted[0], 0.22, abs_tol=1e-6)


def test_chain_lookahead_avoids_awkward_final_overshoot() -> None:
    measure = playability._PolylineMeasure.create(((0.0, 0.0), (0.0, 30.0)))
    pieces = (
        playability._RoadPiece(r"o\road\sil25.p3d", 25.0, 25),
        playability._RoadPiece(r"o\road\sil12.p3d", 12.0, 12),
        playability._RoadPiece(r"o\road\sil6.p3d", 6.0, 6),
    )
    token = _CONTEXT.set(_flat_context())
    try:
        fitted = playability._stock_piece_chain(
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=30.0,
            minimum_end_distance=30.0,
            maximum_end_distance=30.0,
        )
    finally:
        _CONTEXT.reset(token)
    assert [piece.nominal_length for piece, _start, _end in fitted] == [12, 12, 6]
    assert fitted[-1][2] == (0.0, 30.0)


def test_terrain_profile_prefers_shorter_rigid_pieces_over_midspan_clipping() -> None:
    measure = playability._PolylineMeasure.create(((0.0, 0.0), (0.0, 50.0)))
    pieces = (
        playability._RoadPiece(r"test\road25.p3d", 50.0, 25),
        playability._RoadPiece(r"test\road12.p3d", 25.0, 12),
    )
    elevations = []
    for row in range(4):
        height = 1.0 if row == 1 else 0.0
        elevations.extend([height] * 4)
    context = _Context(
        elevations,
        SimpleNamespace(cells=4, cell_size=25.0, road_connection_tolerance=0.35),
        {},
    )
    token = _CONTEXT.set(context)
    try:
        fitted = playability._stock_piece_chain(
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=50.0,
            minimum_end_distance=50.0,
            maximum_end_distance=50.0,
        )
    finally:
        _CONTEXT.reset(token)
    assert [piece.length_metres for piece, _start, _end in fitted] == [25.0, 25.0]
    assert fitted[-1][2] == (0.0, 50.0)


def test_diagonal_t_junction_uses_real_turn_pieces_and_connects_each_arm() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    dataset = _junction_dataset(projection, (650.0, 650.0))
    spec = _junction_spec(bbox)
    report = playability.fit_road_objects(
        dataset, projection, [0.0] * (40 * 40), spec
    )

    node = (500.0, 500.0)
    plan = paved_junctions._plans(
        dataset, projection, spec
    )[playability._road_node_key(node)]
    assert report.junction_cap_objects == 1
    assert report.objects[0].model_path.casefold() == plan.model_path.casefold()
    assert report.objects[0].model_path.casefold().endswith(
        r"\kr_new_sil_sil_t.p3d"
    )

    approaches = report.objects[report.junction_cap_objects :]
    assert any(
        re.search(r"\\(?:sil|asf|kos)10 (?:25|50|75|100)\.p3d$", obj.model_path.casefold())
        for obj in approaches
    )
    endpoints = tuple(
        endpoint
        for obj in approaches
        for endpoint in _object_endpoints(obj)
    )
    for connector in plan.connectors:
        assert min(math.dist(connector.point, endpoint) for endpoint in endpoints) <= 0.05


def test_skew_four_way_intersection_uses_stock_x_and_turn_approaches() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    angle = math.radians(60.0)
    dx, dz = math.sin(angle) * 220.0, math.cos(angle) * 220.0
    roads = (
        OsmLineFeature(
            "way/main",
            {"highway": "residential"},
            tuple(
                projection.to_latlon(point)
                for point in ((500.0, 280.0), centre, (500.0, 720.0))
            ),
        ),
        OsmLineFeature(
            "way/cross",
            {"highway": "residential"},
            tuple(
                projection.to_latlon(point)
                for point in (
                    (centre[0] - dx, centre[1] - dz),
                    centre,
                    (centre[0] + dx, centre[1] + dz),
                )
            ),
        ),
    )
    dataset = OsmDataset(
        source_generator="road-quality-x",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=roads,
    )
    spec = _junction_spec(bbox)
    report = playability.fit_road_objects(
        dataset, projection, [0.0] * (40 * 40), spec
    )

    assert report.junction_cap_objects == 1
    assert report.objects[0].model_path.casefold().endswith(
        r"\kr_new_silxsil.p3d"
    )
    assert any(
        "10 " in obj.model_path.casefold()
        for obj in report.objects[report.junction_cap_objects :]
    )


def test_dirt_and_gravel_t_nodes_do_not_create_junction_models() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    centre_key = playability._road_node_key((500.0, 500.0))

    for surface, procedural in (("dirt", False), ("gravel", True)):
        dataset = _surface_junction_dataset(projection, surface=surface)
        spec = _junction_spec(
            bbox, procedural_gravel_roads=procedural
        )
        geometry = road_quality._junction_geometry(dataset, projection, spec)
        assert centre_key not in geometry

        report = playability.fit_road_objects(
            dataset, projection, [0.0] * (40 * 40), spec
        )
        assert not any(
            "kr_new_" in obj.model_path.casefold()
            or "gravel_j" in obj.model_path.casefold()
            for obj in report.objects
        )


def test_paved_cap_is_matched_by_position_not_filtered_plan_index() -> None:
    plan = paved_junctions._plan(
        (500.0, 500.0),
        (
            ((0.0, 1.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((1.0, 0.0), "sil"),
        ),
    )
    assert plan is not None
    report = playability.RoadFitReport(
        objects=(
            playability.WorldObject(1, r"o\road\ces6.p3d", 100.0, 0.0, 100.0),
            playability.WorldObject(2, r"o\road\sil6.p3d", 500.0, 0.0, 500.0),
        ),
        chain_count=0,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=2,
    )
    assert paved_junctions._cap_index(report, plan, set()) == 1


def test_milestone9_trusts_paved_junction_turn_and_approach_assets() -> None:
    spec = _junction_spec((0.0, 0.0, 0.01, 0.01))
    trusted = set(generator._trusted_legacy_asset_paths(spec, 9))
    for path in (
        r"o\road\kr_new_sil_sil_t.p3d",
        r"o\road\kr_new_silxsil.p3d",
        r"o\road\sil10 25.p3d",
        r"o\road\asf10 100.p3d",
        r"o\road\kos6.p3d",
    ):
        assert generator.canonical_asset_path(path) in trusted
