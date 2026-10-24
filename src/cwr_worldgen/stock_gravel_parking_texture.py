# SPDX-License-Identifier: GPL-3.0-or-later
"""Procedural parking gravel graded to the stock asf25 road's grey palette.

CWR-Worldgen's stock gravel donor is o\\road\\asf25.p3d. Its reference
asf_new.paa (legacy AI88) is a dark near-monochrome material: the opaque
central area averages about RGB (74, 74, 74), with common intensities
51, 68, 85, 102. THESE VALUES ONLY calibrate this independently generated
texture. Neither PAA files nor the original road photograph are loaded,
decoded, sampled, or copied when generating parking lots.
"""
from __future__ import annotations

import numpy as np

# Derived from the opaque portion of the supplied stock gravel reference.
STOCK_GRAVEL_REFERENCE_MODEL = r"o\road\asf25.p3d"
STOCK_GRAVEL_REFERENCE_GREY_RGB = (74, 74, 74)
STOCK_GRAVEL_BRIGHTNESS_RANGE = (42, 115)
STOCK_GRAVEL_PROCEDURAL_RECIPE = "procedural-stock-asf25-gravel-grey-v1"


def _hashed_grain(
    world_x: np.ndarray, world_z: np.ndarray, scale: float, salt: int,
) -> np.ndarray:
    """Deterministic 2-D grain; independent of terrain-cell position/size."""
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


def stock_gravel_colour_material(
    world_x: np.ndarray, world_z: np.ndarray,
) -> np.ndarray:
    """Make opaque grey gravel with no dependence on stock image files.

    Broadband, stone-sized and fine grain combine into a muted dark gravel
    fill. Keeping all three channels equal reproduces the stock gravel tone,
    without borrowing its actual texture or adding brown earth colouring.
    """
    if world_x.shape != world_z.shape or world_x.ndim != 2:
        raise ValueError("world_x and world_z must be same-shaped 2D grids")

    broad = _hashed_grain(world_x, world_z, 0.18, 0xA531)
    stones = _hashed_grain(world_x, world_z, 1.60, 0x51A7)
    fines = _hashed_grain(world_x, world_z, 4.20, 0x3D55)
    specks = _hashed_grain(world_x, world_z, 3.15, 0xCA83)

    # The road's modal tones are around 68 and 85; the broader distribution
    # includes the darker 51 and rarer bright 102 pixels.
    value = (
        76.0
        + (broad - 0.5) * 16.0
        + (stones - 0.5) * 25.0
        + (fines - 0.5) * 15.0
    )
    value = value + np.where(specks > 0.965, 9.0, 0.0)
    value = value - np.where(specks < 0.045, 11.0, 0.0)
    value = np.clip(value, *STOCK_GRAVEL_BRIGHTNESS_RANGE)
    return np.repeat(value[:, :, None].astype(np.float32), 3, axis=2)
