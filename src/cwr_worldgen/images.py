# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable, Sequence

from PIL import Image


@dataclass(frozen=True, slots=True)
class HeightmapLoadResult:
    source_width: int
    source_height: int
    source_mode: str
    source_minimum: float
    source_maximum: float
    mapping_minimum: float
    mapping_maximum: float
    clipped_low: int
    clipped_high: int
    source_grid: str
    runtime_grid: str
    legacy_centre_to_vertex_conversion: bool
    elevations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MaterialMaskLoadResult:
    source_width: int
    source_height: int
    source_mode: str
    indices: tuple[int, ...]


def _image_values(image: Image.Image) -> Iterable[object]:
    getter = getattr(image, "get_flattened_data", None)
    if getter is not None:
        return getter()
    return image.getdata()


def _check_image_path(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    if path.suffix.casefold() not in {".png", ".tif", ".tiff"}:
        raise ValueError("heightmaps and material masks must be PNG or TIFF files")


def _default_source_range(mode: str, minimum: float, maximum: float) -> tuple[float, float]:
    if mode in {"1", "L", "P"}:
        return 0.0, 255.0
    if mode.startswith("I;16"):
        return 0.0, 65535.0
    if minimum == maximum:
        raise ValueError("heightmap source range is zero; provide --input-min and --input-max")
    return minimum, maximum


def load_heightmap(
    path: Path,
    width: int,
    height: int,
    *,
    input_mode: str,
    elevation_minimum: float,
    elevation_maximum: float,
    input_minimum: float | None = None,
    input_maximum: float | None = None,
    flip_y: bool = False,
    source_grid: str = "game-cell-centres",
) -> HeightmapLoadResult:
    """Load and bilinearly resample a grayscale PNG/TIFF heightmap.

    ``normalized`` maps an input sample range to an elevation range. ``meters``
    treats source samples as metres and ignores the elevation mapping arguments.
    """

    _check_image_path(path)
    if width <= 0 or height <= 0:
        raise ValueError("target heightmap dimensions must be positive")
    if input_mode not in {"normalized", "meters"}:
        raise ValueError("input mode must be 'normalized' or 'meters'")
    if source_grid not in {"game-cell-centres", "game-terrain-vertices"}:
        raise ValueError("unsupported heightmap sample grid")
    if not math.isfinite(elevation_minimum) or not math.isfinite(elevation_maximum):
        raise ValueError("elevation range must be finite")
    if elevation_minimum >= elevation_maximum:
        raise ValueError("elevation minimum must be lower than elevation maximum")

    with Image.open(path) as source:
        source.load()
        source_width, source_height = source.size
        source_mode = source.mode
        if source_width <= 0 or source_height <= 0:
            raise ValueError("heightmap has invalid dimensions")

        # Preserve integer and floating-point precision. RGB heightmaps are
        # deliberately rejected: silently converting coloured relief art into
        # terrain is the sort of convenience that creates geological comedy.
        if source.mode not in {"1", "L", "P", "I", "F", "I;16", "I;16L", "I;16B"}:
            raise ValueError(
                f"heightmap must be single-channel grayscale, integer, or float; got mode {source.mode!r}"
            )
        working = source.convert("F")
        if flip_y:
            working = working.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        extrema = working.getextrema()
        if extrema is None:
            raise ValueError("heightmap contains no samples")
        source_minimum = float(extrema[0])
        source_maximum = float(extrema[1])
        resampled = working.resize((width, height), resample=Image.Resampling.BILINEAR)
        values = tuple(float(value) for value in _image_values(resampled))

    if not math.isfinite(source_minimum) or not math.isfinite(source_maximum):
        raise ValueError("heightmap contains non-finite source samples")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("heightmap contains non-finite samples")

    clipped_low = 0
    clipped_high = 0

    if input_mode == "meters":
        elevations = values
        mapping_minimum = source_minimum
        mapping_maximum = source_maximum
    else:
        default_minimum, default_maximum = _default_source_range(source_mode, source_minimum, source_maximum)
        mapping_minimum = default_minimum if input_minimum is None else input_minimum
        mapping_maximum = default_maximum if input_maximum is None else input_maximum
        if not math.isfinite(mapping_minimum) or not math.isfinite(mapping_maximum):
            raise ValueError("input sample range must be finite")
        if mapping_minimum >= mapping_maximum:
            raise ValueError("input minimum must be lower than input maximum")

        scale = (elevation_maximum - elevation_minimum) / (mapping_maximum - mapping_minimum)
        mapped: list[float] = []
        for value in values:
            if value < mapping_minimum:
                clipped_low += 1
                value = mapping_minimum
            elif value > mapping_maximum:
                clipped_high += 1
                value = mapping_maximum
            mapped.append(elevation_minimum + (value - mapping_minimum) * scale)
        elevations = tuple(mapped)

    converted = source_grid == "game-cell-centres"
    if converted:
        elevations = regrid_cell_centres_to_vertices(elevations, width, height)

    return HeightmapLoadResult(
        source_width=source_width,
        source_height=source_height,
        source_mode=source_mode,
        source_minimum=source_minimum,
        source_maximum=source_maximum,
        mapping_minimum=mapping_minimum,
        mapping_maximum=mapping_maximum,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        source_grid=source_grid,
        runtime_grid="game-terrain-vertices",
        legacy_centre_to_vertex_conversion=converted,
        elevations=tuple(elevations),
    )


def regrid_cell_centres_to_vertices(
    elevations: Sequence[float], width: int, height: int
) -> tuple[float, ...]:
    """Shift a legacy cell-centre grid half a sample onto 4WVR vertices.

    Interior vertices are the bilinear value halfway between their four
    surrounding centre samples.  The west and south borders clamp to the
    nearest source centre because the legacy raster contains no samples beyond
    those boundaries.
    """

    if width <= 0 or height <= 0 or len(elevations) != width * height:
        raise ValueError("cell-centre elevation grid has the wrong dimensions")
    result: list[float] = []
    for row in range(height):
        previous_row = max(0, row - 1)
        for column in range(width):
            west_column = max(0, column - 1)
            result.append(
                (
                    float(elevations[previous_row * width + west_column])
                    + float(elevations[previous_row * width + column])
                    + float(elevations[row * width + west_column])
                    + float(elevations[row * width + column])
                )
                * 0.25
            )
    return tuple(result)


def _nearest_material(red: int, green: int, blue: int, palette: Sequence[tuple[int, int, int]]) -> int:
    return min(
        range(len(palette)),
        key=lambda index: (
            (red - palette[index][0]) ** 2
            + (green - palette[index][1]) ** 2
            + (blue - palette[index][2]) ** 2
        ),
    )


def load_material_mask(
    path: Path,
    width: int,
    height: int,
    *,
    palette: Sequence[tuple[int, int, int]],
    flip_y: bool = False,
) -> MaterialMaskLoadResult:
    """Load a mask using nearest-neighbour resampling.

    Grayscale masks split 0..255 evenly across the material count. RGB masks
    select the nearest configured material colour.
    """

    _check_image_path(path)
    if len(palette) < 2 or len(palette) > 255:
        raise ValueError("material palette must contain between 2 and 255 entries")

    with Image.open(path) as source:
        source.load()
        source_width, source_height = source.size
        source_mode = source.mode
        if flip_y:
            source = source.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if source.mode in {"1", "L", "P", "I", "I;16", "I;16L", "I;16B", "F"}:
            working = source.convert("L").resize((width, height), resample=Image.Resampling.NEAREST)
            divisor = 256.0 / len(palette)
            indices = tuple(min(len(palette) - 1, int(value / divisor)) for value in _image_values(working))
        else:
            working = source.convert("RGB").resize((width, height), resample=Image.Resampling.NEAREST)
            indices = tuple(_nearest_material(*pixel, palette) for pixel in _image_values(working))

    return MaterialMaskLoadResult(source_width, source_height, source_mode, indices)
