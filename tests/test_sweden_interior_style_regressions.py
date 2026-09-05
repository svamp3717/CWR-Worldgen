# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace

from cwr_worldgen import osm_house_modeler_texture_bridge as texture_bridge
from cwr_worldgen import osm_house_modeler_upgrade as upgrade
from cwr_worldgen import procedural_buildings as pb


def _swedish_house(
    *,
    colour: str = "falun red",
    width_m: float = 10.0,
    length_m: float = 20.0,
) -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        "residential",
        "gabled",
        width_m,
        length_m,
        6.0,
        foundation_depth_m=0.5,
        regional_style="swedish_wood",
        interiors=True,
        country_style_identifier="se_sweden",
        wall_material="painted vertical timber cladding",
        colour_palette=(colour,),
        window_width_m=1.0,
        window_height_m=1.1,
        window_frame_material="painted timber",
        roof_storey=True,
        roof_storey_windows_per_gable=1,
    )


def test_interior_variant_reuse_preserves_swedish_facade_colour_when_fit_is_valid() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="sweden_reuse_regression",
        generate_interiors=True,
    )
    requested = _swedish_house()
    red_fit = replace(requested, length_m=18.0)
    grey_closer = replace(
        requested,
        length_m=19.5,
        colour_palette=("grey",),
    )

    pool = library._reuse_candidates(requested, (red_fit, grey_closer))

    assert pool == [red_fit]
    assert library._best_variant(requested, pool) == red_fit


def test_interior_variant_reuse_keeps_physical_fit_ahead_of_colour() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="sweden_reuse_fit_regression",
        generate_interiors=True,
    )
    requested = _swedish_house()
    red_wrong_size = replace(requested, width_m=5.0)
    grey_fit = replace(
        requested,
        length_m=19.5,
        colour_palette=("grey",),
    )

    pool = library._reuse_candidates(requested, (red_wrong_size, grey_fit))

    assert pool == [grey_fit]


def test_shared_window_trim_texture_is_light_and_neutral() -> None:
    image = texture_bridge.modeler_window_frame_texture_image(64).convert("RGB")
    pixels = tuple(image.getdata())
    mean = tuple(
        sum(pixel[channel] for pixel in pixels) / len(pixels)
        for channel in range(3)
    )

    assert min(mean) >= 180.0
    assert max(mean) - min(mean) <= 5.0


def test_roof_storey_windows_reuse_shared_light_trim_texture() -> None:
    key = _swedish_house(width_m=10.0, length_m=14.0)
    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[pb._Face] = []

    upgrade._append_roof_storey_windows(
        points,
        normals,
        faces,
        key,
        roof_pitch_degrees=40.0,
        reference_texture=r"sweden_trim_regression\d\roof.paa",
    )

    assert faces
    assert r"sweden_trim_regression\d\t.paa" in {
        face.texture for face in faces
    }
