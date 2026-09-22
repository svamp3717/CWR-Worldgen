from pathlib import Path
import math
from types import SimpleNamespace

import pytest

from cwr_worldgen import paved_intersection_overlap_policy as overlap
from cwr_worldgen import paved_junction_policy as paved
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen.generator import _verify_single_world_pbo_layout
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.model import WorldObject
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.pbo import pack_directory


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
        name="paved_overlap_unit",
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


def _cross_dataset(projection, crossing_degrees: float) -> OsmDataset:
    centre = (500.0, 500.0)
    angle = math.radians(crossing_degrees)
    dx = math.sin(angle) * 220.0
    dz = math.cos(angle) * 220.0
    tags = {"highway": "residential"}
    roads = (
        OsmLineFeature(
            "way/main",
            tags,
            tuple(
                projection.to_latlon(point)
                for point in ((500.0, 280.0), centre, (500.0, 720.0))
            ),
        ),
        OsmLineFeature(
            "way/cross",
            tags,
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
    return OsmDataset(
        source_generator="paved-crossroad-overlap-regression",
        element_count=len(roads),
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=roads,
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


def test_failed_paved_t_cap_becomes_one_edge_free_generated_hub() -> None:
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

    assert len(result.objects) == 1
    assert result.objects[0].object_id == cap.object_id
    assert result.objects[0].model_path.casefold().endswith(
        r"\paved_j3_sil_t90.p3d"
    )
    assert result.objects[0].y == pytest.approx(
        overlap._PAVED_JUNCTION_PLACEMENT_OFFSET_METRES
    )
    assert (
        result.objects[0].y
        + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
        > playability._STOCK_ROAD_VERTICAL_OFFSET_METRES
    )
    assert result.short_piece_objects == 1


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
        (45.0, r"\paved_j3_sil_t45r.p3d", 0.012),
        (90.0, r"\kr_new_sil_sil_t.p3d", 0.062),
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
        assert not any(
            obj.object_id != cap.object_id
            and math.dist((obj.x, obj.z), (500.0, 500.0)) <= 0.75
            for obj in report.objects
        )
        approaches = report.objects[report.junction_cap_objects :]
        assert any(
            obj.y == pytest.approx(playability._STOCK_ROAD_VERTICAL_OFFSET_METRES)
            for obj in approaches
        )
        assert (
            cap.y + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
            > max(obj.y for obj in approaches)
        )
    else:
        approaches = report.objects[report.junction_cap_objects :]
        assert any(
            obj.y == pytest.approx(paved._APPROACH_VERTICAL_OFFSET_METRES)
            for obj in approaches
        )
        assert cap.y > max(obj.y for obj in approaches)


def test_skew_crossroad_emits_only_one_central_surface() -> None:
    """Match the photographed failure: no bright-edged slab crosses the hub."""

    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    spec = _integration_spec(bbox)

    report = playability.fit_road_objects(
        _cross_dataset(projection, 45.0),
        projection,
        [0.0] * (spec.cells * spec.cells),
        spec,
    )

    centred = tuple(
        obj
        for obj in report.objects
        if math.dist((obj.x, obj.z), (500.0, 500.0)) <= 0.75
    )
    assert len(centred) == 1
    hub = centred[0]
    assert hub.model_path.casefold().endswith(r"\paved_j4_sil_x45.p3d")
    assert hub.y == pytest.approx(
        overlap._PAVED_JUNCTION_PLACEMENT_OFFSET_METRES
    )
    approaches = report.objects[report.junction_cap_objects :]
    assert (
        hub.y + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
        > max(obj.y for obj in approaches)
    )


def test_generated_paved_hub_assets_have_visual_and_roadway_surfaces(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.cpp").write_text("class CfgPatches {};\n", encoding="ascii")
    (source / "hub_assets.wrp").write_bytes(b"4WVR")
    library = infrastructure.ProceduralInfrastructureLibrary("hub_assets")
    model_path = overlap.paved_junction_model_path(
        "hub_assets", 4, "sil", "x45"
    )
    library.register_model(model_path)

    result = library.write_assets(
        source,
        tmp_path / "infrastructure.json",
    )

    assert "i/paved_j4_sil_x45.p3d" in result.model_files
    assert "i/pjs.paa" in result.texture_files
    key = next(iter(library._usage))
    visual, _geometry, roadway, _land = overlap._paved_junction_lods(
        key, r"hub_assets\i\pjs.paa"
    )
    assert visual.faces
    assert roadway.faces
    assert {face.texture for face in visual.faces} == {r"hub_assets\i\pjs.paa"}

    pbo_path = tmp_path / "hub_assets.pbo"
    pack_directory(source, pbo_path)
    layout = _verify_single_world_pbo_layout(
        pbo_path,
        "hub_assets",
        result,
    )
    assert layout["generated_road_models"] == [r"i\paved_j4_sil_x45.p3d"]
    assert layout["generated_road_textures"] == [r"i\pjs.paa"]


@pytest.mark.parametrize(
    "kind",
    ("paved_junction_sil", "paved_junction_asf", "paved_junction_kos"),
)
def test_generated_paved_hub_texture_is_low_contrast_asphalt(kind: str) -> None:
    image = infrastructure._texture_image(kind, 128)

    assert image.mode == "RGB"
    assert image.size == (128, 128)
    assert max(high - low for low, high in image.getextrema()) <= 4

    pixels = image.load()
    maximum_adjacent_delta = max(
        abs(pixels[x, y][channel] - pixels[x - 1, y][channel])
        for y in range(128)
        for x in range(1, 128)
        for channel in range(3)
    )
    assert maximum_adjacent_delta <= 2


def test_generated_paved_hub_participates_in_final_building_clearance() -> None:
    hub = WorldObject(
        1,
        r"test_world\i\paved_j4_sil_x45.p3d",
        100.0,
        overlap._PAVED_JUNCTION_PLACEMENT_OFFSET_METRES,
        100.0,
        0.0,
        0.0,
    )

    primitives = clearance._road_object_primitives(hub, _unit_spec())

    assert len(primitives) == 4
    assert {primitive.half_width for primitive in primitives} == {4.55}
    assert all(
        math.dist(primitive.start, primitive.end)
        == pytest.approx(overlap._PAVED_JUNCTION_ARM_EXTENT_METRES)
        for primitive in primitives
    )
