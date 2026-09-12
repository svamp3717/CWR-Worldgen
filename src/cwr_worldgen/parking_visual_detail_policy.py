# SPDX-License-Identifier: GPL-3.0-or-later
"""Add readable parking-lot structure to paved and unsealed terrain overlays."""
from __future__ import annotations

import numpy as np
from PIL import Image


_INSTALLED = False
DETAIL_CACHE_SCHEMA = 2


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
    # Give unsealed lots aggregate at multiple scales rather than one uniform
    # brown noise field. The coordinates remain world based so cell boundaries
    # do not reveal where the 50 m WRP grid happens to be.
    coarse = parking._world_noise(world_x * 0.22, world_z * 0.22, salt=0x7741)
    broad = parking._world_noise(world_x * 0.065, world_z * 0.065, salt=0xC0A5)
    variation = (coarse - 0.5) * 12.0 + (broad - 0.5) * 9.0
    image[mask] = np.clip(
        image[mask] + variation[:, :, None][mask], 0.0, 255.0
    )

    if geometry.half_length < 8.0 or geometry.half_width < 5.0:
        return

    dx = world_x - geometry.centre_x
    dz = world_z - geometry.centre_z
    along = dx * geometry.ux + dz * geometry.uz
    lateral = dx * geometry.px + dz * geometry.pz

    # A compacted circulation aisle is the main cue that this is a parking lot
    # rather than a random gravel polygon. Twin ruts keep it suitably rural.
    drive_half = min(4.2, max(2.6, geometry.half_width * 0.22))
    aisle = mask & (np.abs(lateral) <= drive_half)
    _blend(image, aisle, (112.0, 105.0, 89.0), 0.18)

    rut_offset = drive_half * 0.42
    rut_half = max(0.24, pixel_scale * 0.85)
    ruts = aisle & (np.abs(np.abs(lateral) - rut_offset) <= rut_half)
    _blend(image, ruts, (75.0, 71.0, 62.0), 0.24)

    # Repeated compacted patches along the two parking rows suggest where tyres
    # repeatedly stop without painting implausible white bay markings on dirt.
    stall = 3.0
    phase = np.mod(along + geometry.half_length, stall)
    edge_depth = min(5.4, geometry.half_width * 0.42)
    row_centre = geometry.half_width - edge_depth * 0.48
    parked_depth = np.abs(np.abs(lateral) - row_centre) <= min(1.35, edge_depth * 0.32)
    parked_width = np.abs(phase - stall * 0.5) <= 0.72
    worn_stalls = mask & parked_depth & parked_width
    _blend(image, worn_stalls, (88.0, 82.0, 69.0), 0.13)


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
    """Install richer paved and gravel parking rendering and invalidate v1 cache."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import parking_surface_policy as parking

    original_renderer = parking._render_parking_cell

    def render_detailed_parking_cell(*, cell_index, geometries, spec, base):
        image = original_renderer(
            cell_index=cell_index,
            geometries=geometries,
            spec=spec,
            base=base,
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
    # bumping it prevents older plain parking textures from masking this change.
    parking.PARKING_TEXTURE_CACHE_SCHEMA = max(
        int(getattr(parking, "PARKING_TEXTURE_CACHE_SCHEMA", 1)),
        DETAIL_CACHE_SCHEMA,
    )
    _INSTALLED = True
