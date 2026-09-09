# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM runways with the stock CWA runway terrain textures.

The normalized source pipeline already preserves ``aeroway=runway`` centreline
ways and their mapped widths.  Milestone 9 historically merged those cells into
the generic paved-aeroway mask, so the generated WRP used ordinary road ground
artwork and a crossing service road could overwrite the runway entirely.

Keep taxiways/aprons on the existing paved material, but give runways a dedicated
one-character terrain material.  Grass-family presets use ``runtr_d`` while the
Desert preset uses ``runpi_d``.  The runway overlay is applied after the existing
surface pass so ordinary road masks cannot repaint it.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


RUNWAY_MATERIAL_CODE = "n"
GRASS_RUNWAY_TEXTURE = r"data\runtr_d.pac"
DESERT_RUNWAY_TEXTURE = r"data\runpi_d.pac"
_SURFACE_CACHE_V11 = "surface-pipeline-v11-vectorized-material-pass"
_SURFACE_CACHE_V12 = "surface-pipeline-v12-stock-runway-texture"
_INSTALLED = False
_ORIGINAL_BUILD_SURFACE_PASS = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_WRITE_SURFACE_TEXTURES = None


def runway_texture_for_profile(profile: object) -> str:
    """Return the stock runway terrain texture for one ground preset."""
    return (
        DESERT_RUNWAY_TEXTURE
        if str(profile or "").strip().casefold() == "desert"
        else GRASS_RUNWAY_TEXTURE
    )


def _tagged_width_metres(feature, default: float = 36.0) -> float:
    raw = feature.tags.get("width", "")
    try:
        value = float(str(raw or 0).replace(",", "."))
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value <= 0.0:
        value = default
    return max(3.0, value)


def runway_mask(dataset, projection, cells: int) -> np.ndarray:
    """Rasterize only mapped runway areas/centrelines onto the WRP cell grid."""
    from . import surface_pass as surface

    resolution = int(cells) * 8
    image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(image)

    for feature in dataset.aeroway_areas:
        if feature.tags.get("aeroway", "").casefold() != "runway":
            continue
        surface._draw_polygon(draw, feature, projection, resolution, 255)

    for feature in dataset.aeroway_lines:
        if feature.tags.get("aeroway", "").casefold() != "runway":
            continue
        points = [projection.to_pixel(point, resolution) for point in feature.points]
        if len(points) < 2:
            continue
        width_metres = _tagged_width_metres(feature)
        width_pixels = max(
            1,
            int(round(width_metres / projection.world_size * resolution)),
        )
        draw.line(points, fill=255, width=width_pixels, joint="curve")

    return surface._image_to_wrp_mask(image, int(cells), threshold=20)


def apply_runway_material_indices(
    indices: Sequence[int],
    dataset,
    projection,
    raster,
    elevations: Sequence[float],
    spec,
) -> tuple[int, ...]:
    """Apply the dedicated runway material without overriding real surface water."""
    from . import surface_pass as surface

    expected = int(spec.cells) * int(spec.cells)
    if len(indices) != expected or len(elevations) != expected:
        raise ValueError("runway surface grid size mismatch")

    mask = runway_mask(dataset, projection, int(spec.cells))
    if not np.any(mask):
        return tuple(int(value) for value in indices)

    water = (
        np.asarray(raster.water, dtype=np.bool_)
        & (
            np.asarray(elevations, dtype=np.float64)
            <= float(spec.sea_level) + 1.0e-7
        )
    )
    buildings = np.asarray(raster.buildings, dtype=np.bool_)
    selected = mask & (~water) & (~buildings)
    if not np.any(selected):
        return tuple(int(value) for value in indices)

    result = np.asarray(indices, dtype=np.int16).copy()
    result[selected] = int(surface.MATERIAL_INDEX[RUNWAY_MATERIAL_CODE])
    return tuple(int(value) for value in result)


def _install_runway_material() -> None:
    from . import generator
    from . import stock_desert_surface_policy as desert
    from . import surface_pass as surface

    if RUNWAY_MATERIAL_CODE not in surface.MATERIAL_INDEX:
        material = surface.SurfaceMaterialDefinition(
            RUNWAY_MATERIAL_CODE,
            "runway",
            (72, 72, 68),
            GRASS_RUNWAY_TEXTURE,
        )
        surface.MILESTONE9_MATERIALS = (*surface.MILESTONE9_MATERIALS, material)
        surface.MATERIAL_INDEX = {
            item.code: index
            for index, item in enumerate(surface.MILESTONE9_MATERIALS)
        }
        # generator.py imports this tuple directly, so keep its module-level
        # reference synchronized with the extended surface table.
        generator.MILESTONE9_MATERIALS = surface.MILESTONE9_MATERIALS

    stock_profiles = {
        name: dict(paths)
        for name, paths in surface.STOCK_SURFACE_TEXTURES.items()
    }
    for profile in ("generated", "everon", "nogova", "malden", "desert"):
        paths = stock_profiles.setdefault(profile, {})
        paths[RUNWAY_MATERIAL_CODE] = runway_texture_for_profile(profile)
    surface.STOCK_SURFACE_TEXTURES = stock_profiles

    # Desert's runtime policy bypasses surface_texture_wire_paths and performs
    # its own exhaustive material-code lookup, so extend that stock table too.
    desert_paths = dict(desert.DESERT_STOCK_SURFACE_TEXTURES)
    desert_paths[RUNWAY_MATERIAL_CODE] = DESERT_RUNWAY_TEXTURE
    desert.DESERT_STOCK_SURFACE_TEXTURES = desert_paths


def install_runway_surface_policy() -> None:
    """Install dedicated stock runway material handling exactly once."""
    global _INSTALLED, _ORIGINAL_BUILD_SURFACE_PASS, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_WRITE_SURFACE_TEXTURES
    if _INSTALLED:
        return

    from . import generator
    from . import surface_pass as surface

    _install_runway_material()
    _ORIGINAL_BUILD_SURFACE_PASS = surface.build_surface_pass
    _ORIGINAL_CACHE_KEY = generator.cache_key
    _ORIGINAL_WRITE_SURFACE_TEXTURES = surface.write_surface_textures

    def build_surface_pass_with_runways(
        dataset,
        projection,
        raster,
        elevations,
        slopes,
        spec,
    ):
        report = _ORIGINAL_BUILD_SURFACE_PASS(
            dataset,
            projection,
            raster,
            elevations,
            slopes,
            spec,
        )
        indices = apply_runway_material_indices(
            report.indices,
            dataset,
            projection,
            raster,
            elevations,
            spec,
        )
        return report if indices == report.indices else replace(report, indices=indices)

    def write_surface_textures_with_runway_bookkeeping(
        source_dir,
        world_name,
        profile,
        seed,
        size,
    ):
        paths = list(
            _ORIGINAL_WRITE_SURFACE_TEXTURES(
                source_dir,
                world_name,
                profile,
                seed,
                size,
            )
        )
        # generator.py historically stages every material as a local file for
        # generated/Malden profiles before it asks which WRP paths are external.
        # The runway WRP entry is stock, but keep the expected unused n.paa in
        # those two build trees so cache/PBO bookkeeping does not reference a
        # file that the stock-texture skip deliberately omitted.
        if str(profile or "").strip().casefold() in {"generated", "malden"}:
            path = source_dir / "data" / f"{RUNWAY_MATERIAL_CODE}.paa"
            if not path.is_file():
                material = surface.MILESTONE9_MATERIALS[
                    surface.MATERIAL_INDEX[RUNWAY_MATERIAL_CODE]
                ]
                surface.write_rgb_dxt1_paa(
                    path,
                    surface.create_surface_texture(material, seed, size),
                )
            if path not in paths:
                paths.append(path)
        return tuple(paths)

    def runway_cache_key(namespace: str, payload):
        if namespace == _SURFACE_CACHE_V11:
            namespace = _SURFACE_CACHE_V12
        return _ORIGINAL_CACHE_KEY(namespace, payload)

    # Both modules hold direct references imported during package initialization.
    # Patch the public surface helpers and the generator's imported bindings.
    surface.build_surface_pass = build_surface_pass_with_runways
    generator.build_surface_pass = build_surface_pass_with_runways
    surface.write_surface_textures = write_surface_textures_with_runway_bookkeeping
    generator.write_surface_textures = write_surface_textures_with_runway_bookkeeping
    generator.cache_key = runway_cache_key
    _INSTALLED = True
