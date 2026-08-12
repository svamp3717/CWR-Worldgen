# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    code: str
    name: str
    colour: tuple[int, int, int]


DEFAULT_MATERIALS: tuple[MaterialDefinition, ...] = (
    MaterialDefinition("w", "seabed", (54, 82, 104)),
    MaterialDefinition("s", "sand", (183, 169, 112)),
    MaterialDefinition("g", "grass", (86, 125, 70)),
    MaterialDefinition("r", "rock", (108, 105, 101)),
)

OSM_MATERIALS: tuple[MaterialDefinition, ...] = DEFAULT_MATERIALS + (
    MaterialDefinition("f", "forest", (48, 88, 47)),
    MaterialDefinition("a", "farmland", (148, 143, 78)),
    MaterialDefinition("u", "urban", (137, 133, 127)),
    MaterialDefinition("p", "road", (58, 57, 54)),
)

# Conservative Everon/Eden ground palette using verified paths from the original
# Eden and O data PBOs. RVW4 assigns one legacy texture per 50 m cell. The
# expanded Milestone 9 palette supplies the additional rock and field variants;
# this legacy table uses the first texture from each of those families.
EVERON_GROUND_TEXTURES: dict[str, str] = {
    "w": r"Eden\tn.paa",
    "s": r"Eden\bak\bah.pac",
    "g": r"Eden\zbh.paa",
    "r": r"o\l1.paa",
    "f": r"Eden\zbh.paa",
    "a": r"o\pole1.paa",
    "u": r"Eden\tn.paa",
    "p": r"Eden\tn.paa",
}

# Nogova uses this three-tile family across general terrain materials. Keep the
# table independent so the explicit Everon preset remains unchanged.
NOGOVA_GROUND_TEXTURE_CYCLE: tuple[str, ...] = (
    r"o\t1.paa",
    r"o\trava2.paa",
    r"o\trava3.paa",
)
NOGOVA_GROUND_TEXTURES: dict[str, str] = {
    material.code: NOGOVA_GROUND_TEXTURE_CYCLE[index % len(NOGOVA_GROUND_TEXTURE_CYCLE)]
    for index, material in enumerate(OSM_MATERIALS)
}
NOGOVA_GROUND_TEXTURES.update({
    "a": r"o\pole1.paa",
})

STOCK_GROUND_TEXTURES: dict[str, dict[str, str]] = {
    "everon": EVERON_GROUND_TEXTURES,
    "nogova": NOGOVA_GROUND_TEXTURES,
}

GROUND_TEXTURE_PROFILES = ("nogova", "malden", "everon", "generated", "desert")


MALDEN_MATERIAL_COLOURS: dict[str, tuple[int, int, int]] = {
    # A restrained CWC-era Mediterranean palette. Farmland deliberately shares
    # the basic grass colour because the Malden preset does not assume a
    # dedicated stock field texture is available.
    "w": (48, 73, 91),
    "s": (184, 162, 109),
    "g": (109, 118, 70),
    "r": (110, 105, 96),
    "f": (64, 86, 48),
    "a": (109, 118, 70),
    "u": (132, 126, 115),
    "p": (61, 59, 55),
}

DESERT_MATERIAL_COLOURS: dict[str, tuple[int, int, int]] = {
    "w": (49, 82, 98),
    "s": (204, 183, 121),
    "g": (157, 145, 86),
    "r": (126, 111, 90),
    "f": (102, 118, 67),
    "a": (176, 154, 91),
    "u": (143, 132, 115),
    "p": (67, 62, 55),
}


def material_colour_for_profile(material: MaterialDefinition, profile: str) -> tuple[int, int, int]:
    if profile == "malden":
        return MALDEN_MATERIAL_COLOURS.get(material.code, material.colour)
    if profile == "desert":
        return DESERT_MATERIAL_COLOURS.get(material.code, material.colour)
    return material.colour


def ground_texture_path(world_name: str, material_code: str, profile: str = "generated") -> str:
    if material_code not in {material.code for material in OSM_MATERIALS}:
        raise ValueError(f"unknown terrain material code: {material_code}")
    if profile in {"generated", "desert", "malden"}:
        return rf"{world_name}\data\{material_code}.paa"
    if profile in STOCK_GROUND_TEXTURES:
        return STOCK_GROUND_TEXTURES[profile][material_code]
    raise ValueError(f"unknown ground texture profile: {profile}")


@dataclass(frozen=True, slots=True)
class SpawnPoint:
    cell_x: int
    cell_z: int
    x: float
    y: float
    z: float
    slope_degrees: float


def calculate_slopes(
    elevations: Sequence[float], width: int, height: int, cell_size: float
) -> tuple[float, ...]:
    if len(elevations) != width * height:
        raise ValueError("elevation grid has the wrong size")
    if cell_size <= 0:
        raise ValueError("cell size must be positive")

    slopes: list[float] = []
    for z in range(height):
        north = max(0, z - 1)
        south = min(height - 1, z + 1)
        z_span = max(1, south - north) * cell_size
        for x in range(width):
            west = max(0, x - 1)
            east = min(width - 1, x + 1)
            x_span = max(1, east - west) * cell_size
            dx = (elevations[z * width + east] - elevations[z * width + west]) / x_span
            dz = (elevations[south * width + x] - elevations[north * width + x]) / z_span
            slopes.append(math.degrees(math.atan(math.hypot(dx, dz))))
    return tuple(slopes)


def classify_materials(
    elevations: Sequence[float],
    slopes: Sequence[float],
    *,
    sea_level: float,
    beach_height: float,
    rock_height: float,
    rock_slope_degrees: float,
) -> tuple[int, ...]:
    if len(elevations) != len(slopes):
        raise ValueError("elevation and slope grids must have identical sizes")
    if beach_height < 0:
        raise ValueError("beach height must not be negative")
    if not 0 <= rock_slope_degrees < 90:
        raise ValueError("rock slope must be within 0..90 degrees")

    result: list[int] = []
    for elevation, slope in zip(elevations, slopes):
        if elevation <= sea_level:
            result.append(0)
        elif elevation <= sea_level + beach_height:
            result.append(1)
        elif elevation >= rock_height or slope >= rock_slope_degrees:
            result.append(3)
        else:
            result.append(2)
    return tuple(result)


def choose_spawn(
    elevations: Sequence[float],
    slopes: Sequence[float],
    width: int,
    height: int,
    cell_size: float,
    *,
    sea_level: float,
    minimum_clearance: float,
    maximum_slope_degrees: float,
    excluded: Sequence[bool] | None = None,
) -> SpawnPoint:
    if len(elevations) != width * height or len(slopes) != width * height:
        raise ValueError("terrain grids have the wrong size")
    if excluded is not None and len(excluded) != width * height:
        raise ValueError("spawn exclusion grid has the wrong size")
    centre_x = (width - 1) / 2.0
    centre_z = (height - 1) / 2.0
    candidates: list[tuple[float, float, int, int, float, float]] = []
    for z in range(height):
        for x in range(width):
            index = z * width + x
            elevation = elevations[index]
            slope = slopes[index]
            if excluded is not None and excluded[index]:
                continue
            if elevation < sea_level + minimum_clearance or slope > maximum_slope_degrees:
                continue
            distance_sq = (x - centre_x) ** 2 + (z - centre_z) ** 2
            # Distance dominates, with flatter and higher cells used as stable tie-breakers.
            candidates.append((distance_sq, slope, -elevation, x, z, elevation))
    if not candidates:
        raise ValueError(
            "heightmap contains no playable spawn cell above the water clearance and below the slope limit"
        )
    _, slope, _, x, z, elevation = min(candidates)
    return SpawnPoint(
        cell_x=x,
        cell_z=z,
        x=(x + 0.5) * cell_size,
        y=elevation + 0.2,
        z=(z + 0.5) * cell_size,
        slope_degrees=slope,
    )


def material_counts(indices: Sequence[int], material_count: int) -> tuple[int, ...]:
    counts = [0] * material_count
    for index in indices:
        if not 0 <= index < material_count:
            raise ValueError(f"material index {index} is outside 0..{material_count - 1}")
        counts[index] += 1
    return tuple(counts)
