"""Regression tests for visible stock-road surface ordering at crossings.

CWA stock asf/gravel should never render above stock sil/paved at the same
flat ground plane; dirt/ces is beneath both. The same ordering must hold
for stock junction pieces and generated road fallback visual surfaces.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cwr_worldgen import playability as road
from cwr_worldgen import paved_junction_policy as turns
from cwr_worldgen import procedural_infrastructure as proc
from cwr_worldgen import road_finish_parallel_policy as fast


SPEC = SimpleNamespace(
    cells=16, cell_size=25.0,
    road_segment_length=25.0,
    custom_road_shapes=False,
    paved_road_model=r"o\road\sil25.p3d",
    gravel_road_model=r"o\road\asf25.p3d",
    dirt_road_model=r"o\road\ces25.p3d",
)
HEIGHTS = (10.0,) * (SPEC.cells * SPEC.cells)
START, END = (150.0, 170.0), (175.0, 170.0)


@pytest.mark.parametrize(
    ("name", "tags", "path"),
    (
        ("dirt", {"highway": "track", "surface": "dirt"}, r"o\road\ces25.p3d"),
        ("gravel", {"highway": "track", "surface": "gravel"}, r"o\road\asf25.p3d"),
        ("paved", {"highway": "residential", "surface": "asphalt"}, r"o\road\sil25.p3d"),
    ),
)
def test_stock_family_tag_selection_and_vertical_offset(name, tags, path):
    assert road.road_model_for_tags(SPEC, tags) == path
    assert road._road_surface_priority(tags) == {"dirt": 0, "gravel": 1, "paved": 2}[name]
    placed = road._road_object_on_slope(
        1, path, START, END, HEIGHTS, SPEC,
        vertical_offset=road._road_vertical_offset(tags),
    )
    assert placed.y == pytest.approx(
        10.0 + road._road_surface_vertical_offset(name),
        abs=1e-8,
    )
    assert placed.pitch_degrees == 0.0


def test_flat_crossing_covers_dirt_with_gravel_and_both_with_paved():
    y = {}
    for key, model in (
        ("dirt", SPEC.dirt_road_model),
        ("gravel", SPEC.gravel_road_model),
        ("paved", SPEC.paved_road_model),
    ):
        obj = road._road_object_on_slope(
            1, model, START, END, HEIGHTS, SPEC,
            vertical_offset=road._road_surface_vertical_offset(key),
        )
        y[key] = obj.y
    assert y["dirt"] < y["gravel"] < y["paved"]
    # Previous gaps were only 8 mm and 17 mm; preserve actual stock clearance.
    assert y["gravel"] - y["dirt"] >= 0.03
    assert y["paved"] - y["gravel"] >= 0.05
    assert y["paved"] - y["dirt"] >= 0.08
    assert (
        road._STOCK_DIRT_VERTICAL_OFFSET_METRES
        < road._STOCK_GRAVEL_VERTICAL_OFFSET_METRES
        < road._STOCK_GRAVEL_JUNCTION_VERTICAL_OFFSET_METRES
        < road._STOCK_ROAD_VERTICAL_OFFSET_METRES
        < road._STOCK_PAVED_JUNCTION_VERTICAL_OFFSET_METRES
    )


def test_stock_gravel_and_paved_junction_turns_use_correct_planes():
    # These helpers previously hardcoded 0.060 regardless of road family.
    # Such gravel caps could cover paved road pieces at an intersection.
    gravel = turns._straight_object(
        1, "asf", 25, START, END, HEIGHTS, SPEC,
    )
    asphalt = turns._straight_object(
        2, "sil", 25, START, END, HEIGHTS, SPEC,
    )
    assert gravel.y == pytest.approx(
        10.0 + road._STOCK_GRAVEL_JUNCTION_VERTICAL_OFFSET_METRES,
    )
    assert asphalt.y == pytest.approx(
        10.0 + road._STOCK_PAVED_JUNCTION_VERTICAL_OFFSET_METRES,
    )
    assert asphalt.y > gravel.y + 0.06


def test_stock_curve_height_matches_same_family_junction_plane():
    family_expected = (
        ("asf", road._STOCK_GRAVEL_JUNCTION_VERTICAL_OFFSET_METRES),
        ("sil", road._STOCK_PAVED_JUNCTION_VERTICAL_OFFSET_METRES),
    )
    for name, expected in family_expected:
        obj, end, heading = turns._curve_object(
            1, name, 25, (175.0, 160.0), 0.0, 1, HEIGHTS, SPEC,
        )
        assert obj.y == pytest.approx(10.0 + expected, abs=1e-7)
        assert math.isfinite(heading)
        assert all(math.isfinite(v) for v in end)


def test_stock_road_preset_and_explicit_gravel_are_not_sorted_as_paved():
    # Historically the draw priority was inferred indirectly via
    # road_is_dirt(), which was fragile for model/surface mismatches.
    tags = (
        {"highway": "footway", "surface": "asphalt"},
        {"highway": "service", "surface": "dirt"},
        {"highway": "service", "surface": "gravel"},
        {"highway": "residential", "surface": "paved"},
    )
    assert [road._road_surface_priority(t) for t in tags] == [0, 0, 1, 2]


def test_generated_and_stock_surfaces_keep_same_visual_priority():
    # Fast WRP road transforms author generated surfaces 25 mm above their
    # origin. Do not conflate the P3D origin with the actual visual height.
    for key, width in (("dirt", 3.2), ("gravel", 4.6), ("paved", 7.5)):
        model = proc.custom_road_model_path("surface_layers", key, width, 25.0)
        base_offset = road._road_surface_vertical_offset(key)
        obj = road._road_object_on_slope(
            1, model, START, END, HEIGHTS, SPEC,
            vertical_offset=base_offset,
        )
        actual_top = obj.y + proc.GENERATED_GRAVEL_VISUAL_TOP_METRES
        assert actual_top == pytest.approx(10.0 + base_offset, abs=1e-7)
        context = fast._Context(HEIGHTS, SPEC.cells, SPEC.cell_size)
        context.geometry[fast._geometry_key(START, END)] = (
            162.5, 10.0, 170.0, 90.0, 0.0,
        )
        token = fast._CONTEXT.set(context)
        try:
            cached = fast._cached_road_object(
                2, model, START, END, HEIGHTS, SPEC,
                vertical_offset=base_offset,
            )
        finally:
            fast._CONTEXT.reset(token)
        assert cached.y == pytest.approx(obj.y, abs=1e-7)
