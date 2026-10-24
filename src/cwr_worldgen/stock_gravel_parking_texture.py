# SPDX-License-Identifier: GPL-3.0-or-later
"""Soft procedural dirt-and-gravel parking material, without borrowed textures.

The stock gravel donor is o\\road\\asf25.p3d. Its reference colour is useful,
but its surface pattern is NOT used: this module does not read any road PAA.
Parking should resemble weathered compacted ground, not a gravel P3D or camo.
"""
from __future__ import annotations

import numpy as np

STOCK_GRAVEL_REFERENCE_MODEL = r"o\road\asf25.p3d"
STOCK_GRAVEL_REFERENCE_GREY_RGB = (74, 74, 74)
# Dark, natural compacted soil with a subtle warm tone, not road-texture pixels.
PARKING_GROUND_BASE_RGB = (78, 76, 70)
STOCK_GRAVEL_BRIGHTNESS_RANGE = (64, 94)
STOCK_GRAVEL_PROCEDURAL_RECIPE = "procedural-asf25-colour-compacted-dirt-gravel-v3"


def _hashed_grain(
    world_x: np.ndarray, world_z: np.ndarray, scale: float, salt: int,
) -> np.ndarray:
    """Deterministic world-space hash. No per-parking-cell texture restart."""
    x = np.floor(world_x * scale).astype(np.int64).view(np.uint64)
    z = np.floor(world_z * scale).astype(np.int64).view(np.uint64)
    value = (
        x * np.uint64(0x9E3779B97F4A7C15)
        ^ z * np.uint64(0xD1B54A32D192ED03)
        ^ np.uint64(salt)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return (value & np.uint64(0xFFFF)).astype(np.float32) / 65535.0


def _smooth_ground_noise(
    world_x: np.ndarray, world_z: np.ndarray, frequency: float, salt: int,
) -> np.ndarray:
    """Interpolated value noise: unlike raw grid hashes it has no square patches."""
    scaled_x = world_x * frequency
    scaled_z = world_z * frequency
    x0 = np.floor(scaled_x)
    z0 = np.floor(scaled_z)
    fx = (scaled_x - x0).astype(np.float32)
    fz = (scaled_z - z0).astype(np.float32)
    # Smoothstep makes both first derivatives zero at every grid boundary.
    fx = fx * fx * (3.0 - 2.0 * fx)
    fz = fz * fz * (3.0 - 2.0 * fz)
    n00 = _hashed_grain(x0, z0, 1.0, salt)
    n10 = _hashed_grain(x0 + 1.0, z0, 1.0, salt)
    n01 = _hashed_grain(x0, z0 + 1.0, 1.0, salt)
    n11 = _hashed_grain(x0 + 1.0, z0 + 1.0, 1.0, salt)
    top = n00 * (1.0 - fx) + n10 * fx
    bottom = n01 * (1.0 - fx) + n11 * fx
    return top * (1.0 - fz) + bottom * fz


def stock_gravel_colour_material(
    world_x: np.ndarray, world_z: np.ndarray,
) -> np.ndarray:
    """Low-contrast hardpack with a dusting of fine gravel.

    Use soft 10-20m soil variation and very restrained smaller aggregate.
    Avoid sharp metre-scale hashed blocks (which looked like camouflage in
    CWA) and avoid broad dark tire-like bands in the base material.
    """
    if world_x.shape != world_z.shape or world_x.ndim != 2:
        raise ValueError("world_x and world_z must be same-shaped 2D grids")

    broad = _smooth_ground_noise(world_x, world_z, 0.070, 0xA531)
    medium = _smooth_ground_noise(world_x, world_z, 0.270, 0x51A7)
    fine = _hashed_grain(world_x, world_z, 1.95, 0x3D55)
    stone = _hashed_grain(world_x, world_z, 3.10, 0xCA83)

    # Contrast is deliberately an order of magnitude less visually forceful
    # than the old stock-grey gravel grain. Muted taupe is closer to terrain.
    shade = (
        (broad - 0.5) * 8.0
        + (medium - 0.5) * 4.0
        + (fine - 0.5) * 2.2
        + np.where(stone > 0.988, 1.8, 0.0)
        - np.where(stone < 0.012, 1.8, 0.0)
    )
    ground = np.asarray(PARKING_GROUND_BASE_RGB, dtype=np.float32)
    result = ground[None, None, :] + shade[:, :, None]
    return np.clip(result, *STOCK_GRAVEL_BRIGHTNESS_RANGE).astype(np.float32)
