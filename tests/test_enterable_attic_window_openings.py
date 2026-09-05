# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace

from cwr_worldgen import osm_house_modeler_upgrade as upgrade
from cwr_worldgen import procedural_buildings as pb


def _roof_storey_house(*, interiors: bool) -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        14.0,
        6.0,
        regional_style="swedish_wood",
        interiors=interiors,
        country_style_identifier="se_sweden",
        wall_material="painted vertical timber cladding",
        colour_palette=("falun red",),
        window_width_m=1.0,
        window_height_m=1.1,
        window_frame_material="white-painted timber",
        roof_storey=True,
        roof_storey_windows_per_gable=1,
    )


def _solid_gables(key: pb.BuildingVariantKey, pitch: float):
    eave, rise, _slope = pb._gabled_profile(key, pitch)
    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    apex = eave + rise
    points = [
        (-half_width, eave, -half_length),
        (0.0, apex, -half_length),
        (half_width, eave, -half_length),
        (half_width, eave, half_length),
        (0.0, apex, half_length),
        (-half_width, eave, half_length),
    ]
    normals = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    front = pb._Face(
        r"attic_opening_regression\d\wall.paa",
        ((0, 0, 0.0, 0.0), (1, 0, 0.5, 1.0), (2, 0, 1.0, 0.0)),
    )
    back = pb._Face(
        r"attic_opening_regression\d\wall.paa",
        ((3, 1, 0.0, 0.0), (4, 1, 0.5, 1.0), (5, 1, 1.0, 0.0)),
    )
    faces = list(pb._double_sided_faces((front, back)))
    return points, normals, faces, eave, rise


def _point_in_triangle(px: float, py: float, triangle) -> bool:
    (ax, ay), (bx, by), (cx, cy) = triangle
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    negative = d1 < -1.0e-7 or d2 < -1.0e-7 or d3 < -1.0e-7
    positive = d1 > 1.0e-7 or d2 > 1.0e-7 or d3 > 1.0e-7
    return not (negative and positive)


def _wall_covers(points, faces, *, z: float, x: float, y: float) -> bool:
    for face in faces:
        if face.texture != r"attic_opening_regression\d\wall.paa":
            continue
        coords = [points[int(vertex[0])] for vertex in face.vertices]
        if len(coords) != 3 or not all(abs(point[2] - z) < 1.0e-4 for point in coords):
            continue
        triangle = tuple((point[0], point[1]) for point in coords)
        if _point_in_triangle(x, y, triangle):
            return True
    return False


def test_enterable_attic_window_cuts_real_gable_hole() -> None:
    pitch = 40.0
    key = _roof_storey_house(interiors=True)
    points, normals, faces, eave, _rise = _solid_gables(key, pitch)

    upgrade._append_roof_storey_windows(
        points,
        normals,
        faces,
        key,
        roof_pitch_degrees=pitch,
        reference_texture=r"attic_opening_regression\d\wall.paa",
    )

    window_y0 = eave + 0.42
    window_height = 1.1 * 0.78
    centre_y = window_y0 + window_height * 0.5
    assert not _wall_covers(
        points,
        faces,
        z=-key.length_m * 0.5,
        x=0.0,
        y=centre_y,
    )
    textures = {face.texture for face in faces}
    assert r"attic_opening_regression\d\qgl.paa" not in textures
    assert r"attic_opening_regression\d\t.paa" in textures


def test_closed_attic_window_keeps_wall_and_decorative_glass() -> None:
    pitch = 40.0
    key = _roof_storey_house(interiors=False)
    points, normals, faces, eave, _rise = _solid_gables(key, pitch)

    upgrade._append_roof_storey_windows(
        points,
        normals,
        faces,
        key,
        roof_pitch_degrees=pitch,
        reference_texture=r"attic_opening_regression\d\wall.paa",
    )

    centre_y = eave + 0.42 + (1.1 * 0.78) * 0.5
    assert _wall_covers(
        points,
        faces,
        z=-key.length_m * 0.5,
        x=0.0,
        y=centre_y,
    )
    assert r"attic_opening_regression\d\qgl.paa" in {
        face.texture for face in faces
    }
