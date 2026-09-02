from pathlib import Path
from types import SimpleNamespace
import math

import cwr_worldgen.generator as generator
import cwr_worldgen.playability as playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.road_quality_policy import (
    _CONTEXT,
    _Junction,
    _Context,
    _exit_distance,
    _quality_window,
    _tail_error,
)


def _flat_context(*, cells: int = 4, cell_size: float = 25.0):
    spec = SimpleNamespace(
        cells=cells,
        cell_size=cell_size,
        road_connection_tolerance=0.35,
    )
    return _Context([0.0] * (cells * cells), spec, {})


def test_policy_is_installed_for_playability_and_generator() -> None:
    # The final fitter is deliberately wrapped by several stock-road policies.
    # The stable integration contract is that both public call sites use the
    # same composed fitter, not which intermediate wrapper happens to be last.
    assert playability.fit_road_objects is generator.fit_road_objects


def test_stock_overlay_allows_diagonal_approach_to_continue_under_cap() -> None:
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
    # A stock straight cap is a surface underlay, not a hard trimming boundary.
    # Running the approach to the node above it avoids a visible triangular hole
    # at skew junctions while the source-aligned approach owns the visible edge.
    assert adjusted[0] == 0.0


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


def test_tail_lookahead_does_not_fit_hypothetical_geometry() -> None:
    class GeometryForbidden:
        def chord_endpoint(self, *_args, **_kwargs):
            raise AssertionError("tail lookahead must not run geometric chord fitting")

    pieces = (
        playability._RoadPiece(r"o\road\sil25.p3d", 25.0, 25),
        playability._RoadPiece(r"o\road\sil12.p3d", 12.0, 12),
        playability._RoadPiece(r"o\road\sil6.p3d", 6.0, 6),
    )
    measure = GeometryForbidden()

    # From 12 m, the depth-two tail can plan 12 + 6 and finish exactly at 30 m.
    assert _tail_error(measure, pieces, 12.0, 30.0, 30.0, 2) == 0.0
    # From 25 m, no stock sibling fits inside the five-metre remainder.
    assert _tail_error(measure, pieces, 25.0, 30.0, 30.0, 2) == 5.0


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


def test_diagonal_t_junction_does_not_add_overlap_repair_pieces() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    main = OsmLineFeature(
        "way/main",
        {"highway": "residential"},
        tuple(
            projection.to_latlon(point)
            for point in ((500.0, 300.0), (500.0, 500.0), (500.0, 700.0))
        ),
    )
    branch = OsmLineFeature(
        "way/branch",
        {"highway": "residential"},
        tuple(
            projection.to_latlon(point)
            for point in ((500.0, 500.0), (650.0, 650.0))
        ),
    )
    dataset = OsmDataset(
        source_generator="road-quality-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch),
    )
    spec = _Milestone9PlayabilitySpec(
        name="road_quality",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )
    report = playability.fit_road_objects(
        dataset, projection, [0.0] * (40 * 40), spec
    )
    assert report.connection_count == 1
    assert report.failed_connections == 0
    assert report.maximum_connection_gap <= spec.road_connection_tolerance

    node = (500.0, 500.0)
    cap = report.objects[0]
    # A 45-degree side arm is too skewed for the rigid 90-degree stock T mesh.
    # Keep the fitted approaches as the visible edges and use the stock sil6
    # main-axis cap at their shared centre. Do not manufacture a generated fill
    # or append incident-aligned overlap slabs.
    assert cap.model_path.casefold() == r"o\road\sil6.p3d"
    assert cap.y < playability._STOCK_ROAD_VERTICAL_OFFSET_METRES

    branch_obj = next(
        obj
        for obj in report.objects[report.junction_cap_objects :]
        if obj.y >= playability._STOCK_ROAD_VERTICAL_OFFSET_METRES - 1.0e-6
        and abs(((obj.heading_degrees - 45.0 + 180.0) % 360.0) - 180.0) < 1.0
    )
    nominal = int(branch_obj.model_path.casefold().rsplit("sil", 1)[1].split(".p3d", 1)[0])
    length = spec.road_segment_length * nominal / 25.0
    axis = playability._model_axis(branch_obj, length)
    inner_distance = min(math.dist(node, axis[0]), math.dist(node, axis[1]))
    assert inner_distance <= 0.05

    overlap_helpers = [
        obj
        for obj in report.objects[report.junction_cap_objects :]
        if obj.model_path.casefold() == r"o\road\sil6.p3d"
        and obj.y < playability._STOCK_ROAD_VERTICAL_OFFSET_METRES
        and math.dist((obj.x, obj.z), node) < 5.0
    ]
    assert overlap_helpers == []