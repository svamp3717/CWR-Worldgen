from __future__ import annotations

import base64
import json
import math

import pytest

from cwr_worldgen import procedural_buildings as pb


def _token(*, utility_width: float = 0.0, utility_height: float = 0.0) -> str:
    metadata = {
        "texture_renderer_revision": 3,
        "window": {
            "width_m": 1.20,
            "height_m": 1.35,
            "sill_height_m": 0.85,
            "edge_margin_m": 0.70,
            "target_bay_spacing_m": 3.60,
            "density_multiplier": 1.0,
            "type": "paired casement",
            "frame_material": "painted timber",
        },
        "door": {
            "width_m": 0.95,
            "height_m": 2.10,
            "utility_width_m": utility_width,
            "utility_height_m": utility_height,
            "corner_clearance_m": 0.70,
            "keep_clear_of_windows_m": 0.40,
            "type": "panel",
            "material": "timber",
        },
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"western_stucco|plaster~{encoded}|cream"


def _residential_key(**changes) -> pb.BuildingVariantKey:
    values = dict(
        family="residential",
        roof_style="flat",
        width_m=10.0,
        length_m=8.0,
        height_m=6.0,
        foundation_depth_m=0.0,
        regional_style="western_stucco",
        interiors=True,
        second_storey=True,
        facade_storeys=2,
        storey_height_m=3.0,
        window_width_m=1.20,
        window_height_m=1.35,
        window_sill_height_m=0.85,
        window_edge_margin_m=0.70,
        window_bay_spacing_m=3.60,
        window_density_multiplier=1.0,
        door_width_m=0.95,
        door_height_m=2.10,
        door_corner_clearance_m=0.70,
        door_window_clearance_m=0.40,
        texture_style_token=_token(),
    )
    values.update(changes)
    return pb.BuildingVariantKey(**values)


def test_pedestrian_door_opening_uses_exact_modeler_dimensions() -> None:
    key = _residential_key()
    half_width, height, _pivot = pb._door_dimensions(key)
    assert half_width * 2.0 == pytest.approx(0.95, abs=1.0e-6)
    assert height == pytest.approx(2.10, abs=1.0e-6)


def test_utility_door_uses_modeler_utility_dimensions_not_old_family_heuristic() -> None:
    key = pb.BuildingVariantKey(
        family="industrial",
        roof_style="flat",
        width_m=9.0,
        length_m=14.0,
        height_m=5.0,
        regional_style="regional_concrete",
        interiors=True,
        door_width_m=0.95,
        door_height_m=2.10,
        door_corner_clearance_m=0.70,
        texture_style_token=_token(utility_width=3.20, utility_height=3.40),
    )
    half_width, height, _pivot = pb._door_dimensions(key)
    assert half_width * 2.0 == pytest.approx(3.20, abs=1.0e-6)
    assert height == pytest.approx(3.40, abs=1.0e-6)


def test_enterable_windows_use_modeler_width_height_sill_and_edge_margin() -> None:
    key = _residential_key()
    openings = pb._interior_window_openings(key, -5.0, 5.0, 5.8)
    assert openings
    for x0, x1, y0, y1 in openings:
        assert x1 - x0 == pytest.approx(1.20, abs=1.0e-6)
        assert y1 - y0 == pytest.approx(1.35, abs=1.0e-6)
        assert x0 >= -4.30 - 1.0e-6
        assert x1 <= 4.30 + 1.0e-6
    ground = [opening for opening in openings if opening[2] < 2.0]
    assert ground
    assert all(opening[2] == pytest.approx(0.85, abs=1.0e-6) for opening in ground)


def test_window_density_can_increase_as_well_as_reduce_bay_count() -> None:
    sparse = _residential_key(window_density_multiplier=0.50)
    dense = _residential_key(window_density_multiplier=1.60)
    sparse_openings = pb._interior_window_openings(sparse, -5.0, 5.0, 2.8)
    dense_openings = pb._interior_window_openings(dense, -5.0, 5.0, 2.8)
    assert len(dense_openings) > len(sparse_openings)


def test_polygon_native_openings_keep_same_meter_dimensions() -> None:
    key = _residential_key(
        roof_style="gabled",
        height_m=3.2,
        second_storey=False,
        facade_storeys=1,
        footprint_vertices=((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0)),
        entrance_edge=0,
        entrance_fraction=0.5,
    )
    openings = pb._polygon_native_edge_openings(key, 0, 10.0, 3.0)
    door = next(opening for opening in openings if abs(opening[2]) < 1.0e-8)
    windows = [opening for opening in openings if opening[2] > 0.1]
    assert door[1] - door[0] == pytest.approx(0.95, abs=1.0e-6)
    assert door[3] - door[2] == pytest.approx(2.10, abs=1.0e-6)
    assert windows
    assert all(opening[1] - opening[0] == pytest.approx(1.20, abs=1.0e-6) for opening in windows)
    assert all(opening[3] - opening[2] == pytest.approx(1.35, abs=1.0e-6) for opening in windows)


def test_closed_eight_metre_facade_does_not_double_door_width() -> None:
    key = _residential_key(
        width_m=8.0,
        length_m=8.0,
        height_m=3.0,
        interiors=False,
        second_storey=False,
        facade_storeys=1,
    )
    lod = pb._visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        foundation_texture="foundation.paa",
        foundation_depth=0.0,
        plain_wall_texture="plain.paa",
    )
    door_faces = [face for face in lod.faces if face.texture == "front.paa"]
    assert door_faces
    for face in door_faces:
        xs = [lod.points[int(vertex[0])][0] for vertex in face.vertices]
        ys = [lod.points[int(vertex[0])][1] for vertex in face.vertices]
        assert max(xs) - min(xs) == pytest.approx(0.95, abs=1.0e-6)
        # The visual panel keeps a tiny 2 cm top/bottom reveal inside the exact
        # 2.10 m architectural opening, matching CWR's animated-door convention.
        assert max(ys) - min(ys) == pytest.approx(2.06, abs=1.0e-6)

    front_z = -4.0
    front_wall_faces = []
    for face in lod.faces:
        if face.texture != "wall.paa":
            continue
        zs = [lod.points[int(vertex[0])][2] for vertex in face.vertices]
        if zs and all(abs(value - front_z) <= 1.0e-6 for value in zs):
            front_wall_faces.append(face)
    assert front_wall_faces
    # Eight physical metres must cover two 4 m modeler facade bays, so window
    # artwork remains meter-scaled instead of stretching to twice its size.
    assert any(
        max(vertex[2] for vertex in face.vertices) - min(vertex[2] for vertex in face.vertices)
        == pytest.approx(2.0, abs=1.0e-6)
        for face in front_wall_faces
    )
