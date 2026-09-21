from pathlib import Path
import math
from types import SimpleNamespace

import pytest

from cwr_worldgen import paved_intersection_overlap_policy as overlap
from cwr_worldgen import paved_junction_policy as paved
from cwr_worldgen import playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.model import WorldObject
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _report(objects, *, caps=1, short_pieces=1):
    return playability.RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=caps,
        short_piece_objects=short_pieces,
    )


def _unit_spec():
    return SimpleNamespace(
        cells=16,
        cell_size=25.0,
        road_segment_length=25.0,
        road_connection_tolerance=5.0,
        max_road_objects=100,
        advisory_object_limits=False,
    )


def _sil_t_plan(point=(100.0, 100.0)):
    plan = paved._plan(
        point,
        (
            ((0.0, 1.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((1.0, 0.0), "sil"),
        ),
    )
    assert plan is not None
    return plan


def _dataset(projection, branch_degrees: float) -> OsmDataset:
    centre = (500.0, 500.0)
    angle = math.radians(branch_degrees)
    branch_end = (
        centre[0] + math.sin(angle) * 220.0,
        centre[1] + math.cos(angle) * 220.0,
    )
    tags = {"highway": "residential"}
    return OsmDataset(
        source_generator="paved-intersection-overlap-regression",
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
                    for point in ((500.0, 280.0), centre, (500.0, 720.0))
                ),
            ),
            OsmLineFeature(
                "way/branch",
                tags,
                tuple(
                    projection.to_latlon(point)
                    for point in (centre, branch_end)
                ),
            ),
        ),
    )


def _integration_spec(bbox):
    return _Milestone9PlayabilitySpec(
        name="paved_overlap",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10_000,
        strict_assets=False,
    )


def test_failed_paved_t_cap_becomes_lower_fill_with_one_axis_tongue() -> None:
    plan = _sil_t_plan()
    cap = WorldObject(
        1,
        r"o\road\sil6.p3d",
        plan.point[0],
        playability._STOCK_ROAD_VERTICAL_OFFSET_METRES,
        plan.point[1],
        paved._heading(plan.axis),
        0.0,
    )

    result = overlap.finish_paved_intersection_overlaps(
        _report((cap,)),
        {(1000, 1000): plan},
        [0.0] * (16 * 16),
        _unit_spec(),
    )

    assert len(result.objects) == 2
    assert result.objects[0].object_id == cap.object_id
    assert result.objects[0].y == pytest.approx(
        playability._STOCK_ROAD_VERTICAL_OFFSET_METRES
        - overlap._CAP_UNDERLAY_DROP_METRES
    )
    tongue = result.objects[1]
    assert tongue.model_path == cap.model_path
    assert (tongue.x, tongue.z) == pytest.approx(plan.point)
    assert tongue.y == pytest.approx(
        playability._STOCK_ROAD_VERTICAL_OFFSET_METRES
        - overlap._TONGUE_UNDERLAY_DROP_METRES
    )
    branch_heading = paved._heading((1.0, 0.0))
    assert overlap._axis_heading_difference(
        tongue.heading_degrees,
        branch_heading,
    ) <= 1.0e-6
    assert result.short_piece_objects == 2


def test_native_paved_junction_is_not_rewritten_as_fallback_fill() -> None:
    plan = _sil_t_plan()
    native = WorldObject(
        1,
        plan.model_path,
        plan.point[0],
        paved._JUNCTION_VERTICAL_OFFSET_METRES,
        plan.point[1],
        paved._heading(plan.axis),
        0.0,
    )
    report = _report((native,))

    result = overlap.finish_paved_intersection_overlaps(
        report,
        {(1000, 1000): plan},
        [0.0] * (16 * 16),
        _unit_spec(),
    )

    assert result is report


def test_paved_merge_tolerance_cannot_inherit_large_global_gap_allowance() -> None:
    assert paved._merge_length_tolerance(
        SimpleNamespace(road_connection_tolerance=5.0)
    ) == pytest.approx(0.20)
    assert paved._merge_length_tolerance(
        SimpleNamespace(road_connection_tolerance=0.10)
    ) == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("branch_degrees", "expected_cap", "expected_height"),
    (
        (45.0, r"\sil6.p3d", 0.029),
        (90.0, r"\kr_new_sil_sil_t.p3d", 0.058),
    ),
)
def test_final_pipeline_orders_paved_junction_surfaces_without_z_fighting(
    branch_degrees,
    expected_cap,
    expected_height,
) -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    spec = _integration_spec(bbox)

    report = playability.fit_road_objects(
        _dataset(projection, branch_degrees),
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    cap = report.objects[0]
    assert cap.model_path.casefold().endswith(expected_cap)
    assert cap.y == pytest.approx(expected_height)
    if branch_degrees == 45.0:
        assert any(
            obj.object_id != cap.object_id
            and obj.model_path.casefold().endswith(r"\sil6.p3d")
            and obj.y == pytest.approx(0.030)
            for obj in report.objects
        )
    else:
        assert any(
            obj.y == pytest.approx(paved._APPROACH_VERTICAL_OFFSET_METRES)
            for obj in report.objects[report.junction_cap_objects :]
        )
