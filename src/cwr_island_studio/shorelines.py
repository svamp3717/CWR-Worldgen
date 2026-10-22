# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, viewport-bounded shoreline highlighting for Island Studio."""
from __future__ import annotations

import math

import numpy as np


def highlight_shoreline(
    pixels: np.ndarray, elevations: np.ndarray, *,
    water_level: float, height_scale: float,
) -> np.ndarray:
    """Highlight land-side pixels bordering water, using the elevation grid.

    Both arrays must be north-up viewport samples of the same shape. The
    shoreline exists only at a land/water transition; the world perimeter is
    drawn separately as an always-visible map border. The returned image is
    a copy, so neither the WRP nor the underlying preview is modified.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("shoreline image must be an RGB uint8 array")
    if elevations.shape != pixels.shape[:2]:
        raise ValueError("shoreline elevation grid and preview size differ")
    if not math.isfinite(water_level) or not math.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("water level and height scale must be finite (scale > 0)")

    water = elevations.astype(np.float32) * height_scale <= water_level
    shore = np.zeros(water.shape, dtype=np.bool_)
    # Highlight the dry side of each shared pixel edge. This avoids drawing
    # artificial coasts at the WRP perimeter or elsewhere on uniform land.
    horizontal = water[:, 1:] != water[:, :-1]
    shore[:, :-1] |= horizontal & ~water[:, :-1]
    shore[:, 1:] |= horizontal & ~water[:, 1:]
    vertical = water[1:, :] != water[:-1, :]
    shore[:-1, :] |= vertical & ~water[:-1, :]
    shore[1:, :] |= vertical & ~water[1:, :]
    if not np.any(shore):
        return pixels

    # A dark, one-pixel halo gives the light cyan shore contrast on both
    # tinted water and bright mountain terrain, even when zoomed out.
    halo = shore.copy()
    halo[1:, :] |= shore[:-1, :]
    halo[:-1, :] |= shore[1:, :]
    halo[:, 1:] |= shore[:, :-1]
    halo[:, :-1] |= shore[:, 1:]
    result = pixels.copy()
    result[halo] = (11, 31, 42)
    result[shore] = (186, 244, 248)
    return result
