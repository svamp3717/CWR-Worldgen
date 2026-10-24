# SPDX-License-Identifier: GPL-3.0-or-later
"""Add readable parking-lot structure to paved and unsealed terrain overlays."""
from __future__ import annotations

import numpy as np
from PIL import Image


_INSTALLED = False
DETAIL_CACHE_SCHEMA = 4


def _world_grid(*, cell_index: int, spec, size: int):
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    row, col = divmod(int(cell_index), cells)
    pixel_scale = cell_size / size
    xs = col * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    zs = row * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    world_x, world_z = np.meshgrid(xs, zs)
    return world_x, world_z, pixel_scale


def _blend(image: np.ndarray, mask: np.ndarray, colour, strength: float) -> None:
    if not bool(np.any(mask)):
        return
    target = np.asarray(colour, dtype=np.float32)
    image[mask] = image[mask] * (1.0 - strength) + target * strength


def _add_gravel_detail(
    parking,
    image: np.ndarray,
    mask: np.ndarray,
    geometry,
    world_x: np.ndarray,
    world_z: np.ndarray,
    pixel_scale: float,
) -> None:
    """Keep real road aggregate intact, with faint parking-specific tyre wear.

    The base parking texture now IS generated road gravel, not a flat brown
    substitute. Adding artificial beige speckles and dirt colours here would
    change the material back into something unrelated to the stock-style roads.
    Darken existing pixels instead of tinting or repainting the photograph.
    """
    dx = world_x - geometry.centre_x
    dz = world_z - geometry.centre_z
    along = dx * geometry.ux + dz * geometry.uz
    lateral = dx * geometry.px + dz * geometry.pz
    if geometry.half_length < 2.0 or geometry.half_width < 1.45:
        return

    drive_half = max(0.82, geometry.half_width * 0.28)
    drive_half = min(4.2, drive_half, max(0.72, geometry.half_width * 0.62))
    aisle = mask & (np.abs(lateral) <= drive_half)
    image[aisle] *= 0.96

    rut_offset = max(0.36, drive_half * 0.43)
    rut_half = max(0.20, pixel_scale * 0.72)
    ruts = aisle & (np.abs(np.abs(lateral) - rut_offset) <= rut_half)
    image[ruts] *= 0.83

    if geometry.half_length < 3.6 or geometry.half_width < 2.2:
        return
    stall = 2.8
    phase = np.mod(along + geometry.half_length, stall)
    edge_depth = min(5.0, max(1.7, geometry.half_width * 0.46))
    row_centre = max(
        drive_half + 0.38,
        geometry.half_width - edge_depth * 0.50,
    )
    row_centre = min(row_centre, geometry.half_width - max(0.35, pixel_scale))
    parked_depth = np.abs(np.abs(lateral) - row_centre) <= min(
        1.15, edge_depth * 0.31,
    )
    parked_width = np.abs(phase - stall * 0.5) <= 0.68
    image[mask & parked_depth & parked_width] *= 0.91

    end_band = max(0.75, min(1.8, geometry.half_length * 0.20))
    turning = aisle & (np.abs(along) >= geometry.half_length - end_band)
    image[turning] *= 0.92


def _add_paved_detail(
    image: np.ndarray,
    mask: np.ndarray,
    geometry,
    world_x: np.ndarray,
    world_z: np.ndarray,
    pixel_scale: float,
) -> None:
    if geometry.half_length < 7.5 or geometry.half_width < 4.5:
        return

    dx = world_x - geometry.centre_x
    dz = world_z - geometry.centre_z
    along = dx * geometry.ux + dz * geometry.uz
    lateral = dx * geometry.px + dz * geometry.pz

    drive_half = min(4.1, max(2.5, geometry.half_width * 0.22))
    aisle = mask & (np.abs(lateral) <= drive_half)
    _blend(image, aisle, (59.0, 60.0, 59.0), 0.16)

    rut_offset = drive_half * 0.40
    rut_half = max(0.16, pixel_scale * 0.65)
    ruts = aisle & (np.abs(np.abs(lateral) - rut_offset) <= rut_half)
    _blend(image, ruts, (43.0, 44.0, 43.0), 0.17)

    # The base renderer already paints stall dividers. Add the transverse stop
    # line that visually closes each row, making the pattern unmistakably read
    # as nose-in parking rather than a set of unrelated white stripes.
    edge_depth = min(5.2, geometry.half_width * 0.42)
    inner_edge = max(0.0, geometry.half_width - edge_depth)
    line_half = max(0.18, pixel_scale * 0.72)
    stop_lines = mask & (np.abs(np.abs(lateral) - inner_edge) <= line_half)
    _blend(image, stop_lines, (208.0, 205.0, 188.0), 0.82)

    # Small, repeated dark wear patches keep the bays from looking freshly
    # painted. They are deterministic and world aligned, like the rest of the
    # overlay, so rebuilding does not shuffle the pattern around.
    stall = 2.7
    phase = np.mod(along + geometry.half_length, stall)
    row_centre = geometry.half_width - edge_depth * 0.52
    stain_depth = np.abs(np.abs(lateral) - row_centre) <= min(0.75, edge_depth * 0.20)
    stain_width = np.abs(phase - stall * 0.5) <= 0.42
    stains = mask & stain_depth & stain_width
    _blend(image, stains, (43.0, 44.0, 43.0), 0.16)


def apply_parking_visual_detail(
    parking,
    image: Image.Image,
    *,
    cell_index: int,
    geometries,
    spec,
) -> Image.Image:
    """Overlay parking-specific wear/structure on an already rendered cell."""
    size = int(image.width)
    if image.height != size:
        raise ValueError("parking detail input must be square")

    result = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    world_x, world_z, pixel_scale = _world_grid(
        cell_index=cell_index, spec=spec, size=size
    )

    for geometry in geometries:
        mask = parking._polygon_mask_for_cell(
            geometry,
            cell_index=cell_index,
            cells=int(spec.cells),
            cell_size=float(spec.cell_size),
            size=size,
        )
        if not bool(np.any(mask)):
            continue
        if str(geometry.surface).casefold() == "gravel":
            _add_gravel_detail(
                parking,
                result,
                mask,
                geometry,
                world_x,
                world_z,
                pixel_scale,
            )
        else:
            _add_paved_detail(
                result,
                mask,
                geometry,
                world_x,
                world_z,
                pixel_scale,
            )

    return Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8), mode="RGB"
    )


def install_parking_visual_detail_policy() -> None:
    """Install richer paved and gravel parking rendering and invalidate old cache."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import parking_surface_policy as parking

    original_renderer = parking._render_parking_cell

    def render_detailed_parking_cell(*, cell_index, geometries, spec, base, **kwargs):
        image = original_renderer(
            cell_index=cell_index,
            geometries=geometries,
            spec=spec,
            base=base,
            **kwargs,
        )
        return apply_parking_visual_detail(
            parking,
            image,
            cell_index=cell_index,
            geometries=geometries,
            spec=spec,
        )

    parking._render_parking_cell = render_detailed_parking_cell
    # PARKING_TEXTURE_CACHE_SCHEMA participates in every per-cell cache key, so
    # bumping it prevents older plain/detail-v2 parking textures from masking the
    # stronger small-lot gravel treatment.
    parking.PARKING_TEXTURE_CACHE_SCHEMA = max(
        int(getattr(parking, "PARKING_TEXTURE_CACHE_SCHEMA", 1)),
        DETAIL_CACHE_SCHEMA,
    )
    _INSTALLED = True
