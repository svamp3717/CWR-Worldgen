"""CWA must not paint gravel or dirt visual faces across asphalt.

WRP origins and pitch do not guarantee CWA's road artwork depth precedence.
These tests check the actual generated visual footprint, including 0.9m
overhang lips, rather than simply asserting Y ordering.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen import final_road_dedup_policy as road
from cwr_worldgen import procedural_infrastructure as pi
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.road_visible_surface_priority import (
    _visual_footprint, enforce_visible_road_priority,
)


SPEC = SimpleNamespace(
    name="mixed_layers",
    cells=16,
    cell_size=25.0,
    custom_road_shapes=True,
    road_segment_length=25.0,
)
HEIGHTS = [10.0] * (SPEC.cells * SPEC.cells)


def _report(objects):
    return RoadFitReport(
        objects=tuple(objects), chain_count=2, connection_count=1,
        failed_connections=0, maximum_connection_gap=0.0,
        maximum_chain_gap=0.0, truncated=False, junction_cap_objects=0,
    )


def _footprint(obj, index):
    axis = road._lower_surface_axis(obj, index, SPEC)
    assert axis is not None
    sig = pi.custom_road_model_signature(obj.model_path)
    lip = (
        pi.GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
        if sig is not None and sig[0] in {"gravel", "dirt"}
        and not obj.model_path.endswith("_edge.p3d")
        else 0.0
    )
    return _visual_footprint(axis, overlap=lip)


def _assert_lower_surfaces_do_not_print(report):
    blockers = road._paved_blockers(report, SPEC)
    assert blockers
    lower = 0
    for index, obj in enumerate(report.objects):
        axis = road._lower_surface_axis(obj, index, SPEC)
        if axis is None:
            continue
        lower += 1
        for blocker in blockers:
            if blocker.elevation - axis.elevation > 1.5:
                continue
            area = road._polygon_area(road._clip_convex_polygon(
                _footprint(obj, index), blocker.polygon,
            ))
            assert area < 1.0e-5, (obj.model_path, area)
    assert lower > 0


@pytest.mark.parametrize(
    "surface,source",
    (
        ("gravel", r"o\road\asf25.p3d"),
        ("dirt", r"o\road\ces25.p3d"),
        ("gravel", r"mixed_layers\i\road_gravel_w070_l0250.p3d"),
        ("dirt", r"mixed_layers\i\road_dirt_w035_l0250.p3d"),
    ),
)
def test_both_road_families_have_no_painted_faces_inside_asphalt(surface, source):
    width = 7.0 if surface == "gravel" else 3.5
    asphalt = WorldObject(
        2, r"o\road\sil25.p3d", 200, 10.095, 200, 0, 0,
    )
    lower = WorldObject(
        1, source, 200, 10.0, 200, 90, 0,
    )
    original = _report((lower, asphalt))
    # The former final fitter intentionally inserts overlapping, pitched
    # terminals, which CWA can render over the higher asphalt slab.
    old = road.deduplicate_final_road_objects(original, SPEC)
    result = enforce_visible_road_priority(old, SPEC, HEIGHTS)
    assert result.objects != old.objects
    assert asphalt in result.objects
    _assert_lower_surfaces_do_not_print(result)
    repaired = [
        obj for obj in result.objects
        if (sig := pi.custom_road_model_signature(obj.model_path)) is not None
        and sig[0] == surface
    ]
    assert repaired, (surface, source)
    # Ordinary exterior chain pieces can retain their old overlap lips. Only
    # affected junction pieces need flush visual ends.
    assert any(o.model_path.endswith("_edge.p3d") for o in repaired)


@pytest.mark.parametrize("surface", ("gravel", "dirt"))
def test_generated_overlap_lip_is_removed_even_when_nominal_axis_is_clear(surface):
    width = 7.0 if surface == "gravel" else 3.5
    gravel = WorldObject(
        1, pi.custom_road_model_path(SPEC.name, surface, width, 6.0),
        91.7, 10.0, 100, 90, 0,
    )
    asphalt = WorldObject(
        2, r"o\road\sil25.p3d", 100, 10.095, 100, 0, 0,
    )
    original = _report((gravel, asphalt))
    axis = road._lower_surface_axis(gravel, 0, SPEC)
    blocker = road._paved_blockers(original, SPEC)[0]
    assert road._polygon_area(road._clip_convex_polygon(
        road._surface_polygon(axis), blocker.polygon,
    )) == pytest.approx(0.0, abs=1e-6)
    assert road._polygon_area(road._clip_convex_polygon(
        _footprint(gravel, 0), blocker.polygon,
    )) > 0.01
    result = enforce_visible_road_priority(original, SPEC, HEIGHTS)
    assert result.objects[0].model_path.endswith("_edge.p3d")
    assert result.objects[0].object_id == gravel.object_id
    _assert_lower_surfaces_do_not_print(result)


@pytest.mark.parametrize("surface", ("gravel", "dirt"))
def test_flush_end_p3d_has_no_visual_overhang_but_keeps_roadway(surface):
    width = 7.0 if surface == "gravel" else 3.5
    path = pi.custom_road_model_path(
        SPEC.name, surface, width, 6.0, flush_ends=True,
    )
    signature = pi.custom_road_model_signature(path)
    assert signature == (surface, width, 6.0, 0)
    key = pi.InfrastructureModelKey(
        "road", path.rsplit("\\", 1)[-1][:-4], int(width * 10), 60,
    )
    visual, geometry, roadway, land = pi._road_lods(
        key, r"landtext\road.paa",
    )
    assert visual.faces
    assert max(p[2] for p in visual.points) <= 3.0 + 1e-6
    assert min(p[2] for p in visual.points) >= -3.0 - 1e-6
    assert roadway.faces
    assert geometry.resolution != roadway.resolution


@pytest.mark.parametrize(
    ("surface", "width", "paved_x"),
    (("gravel", 7.0, 104.8), ("dirt", 3.5, 103.2)),
)
def test_curved_lower_ribbon_bow_does_not_escape_paved_visual_cutout(
    surface, width, paved_x,
):
    curved = WorldObject(
        1, pi.custom_road_model_path(
            SPEC.name, surface, width, 20.0, 40.0,
        ),
        100, 10.0, 100, 0.0, 0.0,
    )
    paved = WorldObject(
        2, pi.custom_road_model_path(SPEC.name, "paved", 2.0, 25.0),
        paved_x, 10.095, 100, 0.0, 0.0,
    )
    scene = _report((curved, paved))
    axis = road._lower_surface_axis(curved, 0, SPEC)
    blocker = road._paved_blockers(scene, SPEC)[0]
    # Their nominal straight chords do not intersect, but the lower road's
    # outward-curving visible mesh does reach the paved road.
    assert road._polygon_area(road._clip_convex_polygon(
        road._surface_polygon(axis), blocker.polygon,
    )) == pytest.approx(0.0, abs=1e-6)
    clipped = enforce_visible_road_priority(scene, SPEC, HEIGHTS)
    assert clipped.objects == (paved,)


def test_custom_paved_curve_blocker_includes_its_bow_and_rotated_end_corners():
    path = pi.custom_road_model_path(
        SPEC.name, "paved", 9.1, 25.0, 45.0,
    )
    obj = WorldObject(9, path, 100, 10.095, 100, 0.0, 0.0)
    axis = road._road_axis(obj, 0, SPEC)
    assert axis is not None
    assert road._generated_curve_padding(path, axis) > 4.0
    polygon = road._paved_blockers(_report((obj,)), SPEC)[0].polygon
    assert max(x for x, z in polygon) - obj.x > axis.half_width + 4.0


def test_flush_end_road_variants_are_written_as_real_p3d_assets(tmp_path):
    models = tuple(
        pi.custom_road_model_path(
            SPEC.name, surface, width, 6.0, flush_ends=True,
        )
        for surface, width in (("gravel", 7.0), ("dirt", 3.5))
    )
    library = pi.ProceduralInfrastructureLibrary(
        SPEC.name, cache_enabled=False,
        paved_texture_path=r"o\road\sil_new.paa",
    )
    library.register_models(models)
    report = library.write_assets(tmp_path, tmp_path / "roads.json")
    assert report.generated_variants == 2
    for model in models:
        output = tmp_path / model.split("\\", 1)[1].replace("\\", "/")
        assert output.is_file()
        info = pi.inspect_mlod(output)
        assert any(abs(v - pi._ROADWAY_LOD) < 1e9 for v in info.resolutions)
