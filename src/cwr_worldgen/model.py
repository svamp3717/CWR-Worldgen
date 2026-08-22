# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

DEFAULT_MAX_ROAD_OBJECTS = 1_024_000
DEFAULT_MAX_BUILDINGS = 1_000_000
DEFAULT_MAX_FOREST_OBJECTS = 500_000

_WORLD_NAME = re.compile(r"^[a-z][a-z0-9_]{2,19}$")


def _ascii_wire(value: str, maximum: int, label: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain ASCII characters only") from exc
    if not encoded:
        raise ValueError(f"{label} must not be empty")
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds its {maximum}-byte wire limit")
    if b"\x00" in encoded:
        raise ValueError(f"{label} must not contain NUL bytes")
    return encoded


def validate_world_identity(*, name: str, display_name: str, profile: str) -> None:
    """Validate identifiers before any source or geometry work begins."""
    if not _WORLD_NAME.fullmatch(name):
        raise ValueError(
            "world name must be 3-20 lowercase ASCII letters, digits, or underscores, "
            "starting with a letter"
        )
    if profile not in {"cwa", "cwr-ce"}:
        raise ValueError("profile must be 'cwa' or 'cwr-ce'")
    try:
        display_name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("display name must be ASCII for original CWA configs") from exc
    if '"' in display_name or "\n" in display_name or "\r" in display_name:
        raise ValueError("display name contains characters unsafe for config.cpp")


def _validate_common_world(
    *, name: str, display_name: str, profile: str, cells: int, cell_size: float
) -> None:
    validate_world_identity(name=name, display_name=display_name, profile=profile)
    if cells < 16 or cells > 2048 or cells & (cells - 1):
        raise ValueError("cells must be a power of two between 16 and 2048")
    if cell_size <= 0 or not math.isfinite(cell_size):
        raise ValueError("cell size must be positive and finite")


@dataclass(frozen=True, slots=True)
class WorldObject:
    object_id: int
    model_path: str
    x: float
    y: float
    z: float
    heading_degrees: float = 0.0
    pitch_degrees: float = 0.0

    def matrix_4x3(self) -> tuple[float, ...]:
        """Return the RVW4 object transform.

        Heading rotates around world up. Pitch then rotates around the object's
        local right axis, allowing long road pieces to follow the graded terrain
        instead of balancing horizontally on their highest sampled corner.
        Existing objects keep the historical yaw-only transform because pitch
        defaults to zero.
        """

        heading = math.radians(self.heading_degrees)
        pitch = math.radians(self.pitch_degrees)
        cosine_heading = math.cos(heading)
        sine_heading = math.sin(heading)
        cosine_pitch = math.cos(pitch)
        sine_pitch = math.sin(pitch)
        return (
            cosine_heading,
            0.0,
            -sine_heading,
            -sine_heading * sine_pitch,
            cosine_pitch,
            -cosine_heading * sine_pitch,
            sine_heading * cosine_pitch,
            sine_pitch,
            cosine_heading * cosine_pitch,
            self.x,
            self.y,
            self.z,
        )

    def validate(self) -> None:
        if self.object_id < 0:
            raise ValueError("object IDs must be non-negative")
        _ascii_wire(self.model_path, 75, "model path")
        for label, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not math.isfinite(value):
                raise ValueError(f"object {label} coordinate must be finite")
        if not math.isfinite(self.heading_degrees):
            raise ValueError("object heading must be finite")
        if not math.isfinite(self.pitch_degrees) or not -89.0 < self.pitch_degrees < 89.0:
            raise ValueError("object pitch must be finite and within -89..89 degrees")


@dataclass(frozen=True, slots=True)
class WorldSpec:
    name: str = "cwr_milestone1"
    display_name: str = "CWR Milestone 1"
    profile: str = "cwa"
    cells: int = 256
    cell_size: float = 25.0
    sea_border_cells: int = 12
    shore_cells: int = 6
    sea_height: float = -5.0
    land_height: float = 5.0
    texture_colour: tuple[int, int, int] = (86, 125, 70)

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size

    @property
    def centre(self) -> float:
        return self.world_size / 2.0

    @property
    def height_scale(self) -> float:
        # 4WVR stores signed 16-bit height units at a fixed 0.05 metres per
        # unit.  The content profile changes assets/configuration, not this
        # on-disk terrain contract.
        return 0.05

    @property
    def terrain_texture_path(self) -> str:
        return rf"{self.name}\data\g.paa"

    @property
    def dummy_texture_path(self) -> str:
        return rf"{self.name}\data\d.paa"

    def validate(self) -> None:
        _validate_common_world(
            name=self.name,
            display_name=self.display_name,
            profile=self.profile,
            cells=self.cells,
            cell_size=self.cell_size,
        )
        if self.sea_border_cells < 1:
            raise ValueError("sea border must contain at least one cell")
        if self.shore_cells < 1:
            raise ValueError("shore transition must contain at least one cell")
        if (self.sea_border_cells + self.shore_cells) * 2 >= self.cells:
            raise ValueError("sea border and shore transition consume the entire terrain")
        if self.sea_height >= 0:
            raise ValueError("sea height must be below zero")
        if self.land_height <= 0:
            raise ValueError("land height must be above zero")
        _ascii_wire(self.terrain_texture_path, 31, "terrain texture path")
        _ascii_wire(self.dummy_texture_path, 31, "dummy terrain texture path")
        for channel in self.texture_colour:
            if not isinstance(channel, int) or not 0 <= channel <= 255:
                raise ValueError("texture colour channels must be integers within 0..255")


@dataclass(frozen=True, slots=True)
class HeightmapSpec:
    heightmap_path: Path
    name: str = "cwr_milestone2"
    display_name: str = "CWR Milestone 2"
    profile: str = "cwa"
    cells: int = 256
    cell_size: float = 25.0
    heightmap_grid: str = "game-cell-centres"
    input_mode: str = "normalized"
    elevation_minimum: float = -10.0
    elevation_maximum: float = 250.0
    input_minimum: float | None = None
    input_maximum: float | None = None
    material_mask_path: Path | None = None
    flip_y: bool = False
    sea_level: float = 0.0
    beach_height: float = 4.0
    rock_height: float = 140.0
    rock_slope_degrees: float = 28.0
    spawn_clearance: float = 1.0
    maximum_spawn_slope_degrees: float = 18.0

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size

    @property
    def centre(self) -> float:
        return self.world_size / 2.0

    @property
    def height_scale(self) -> float:
        return 0.05

    def terrain_texture_path(self, code: str) -> str:
        return rf"{self.name}\data\{code}.paa"

    @property
    def dummy_texture_path(self) -> str:
        return rf"{self.name}\data\d.paa"

    def validate(self) -> None:
        _validate_common_world(
            name=self.name,
            display_name=self.display_name,
            profile=self.profile,
            cells=self.cells,
            cell_size=self.cell_size,
        )
        if self.input_mode not in {"normalized", "meters"}:
            raise ValueError("input mode must be 'normalized' or 'meters'")
        if self.heightmap_grid not in {"game-cell-centres", "game-terrain-vertices"}:
            raise ValueError(
                "heightmap grid must be 'game-cell-centres' or 'game-terrain-vertices'"
            )
        if not self.heightmap_path.is_file():
            raise ValueError(f"heightmap does not exist: {self.heightmap_path}")
        if self.material_mask_path is not None and not self.material_mask_path.is_file():
            raise ValueError(f"material mask does not exist: {self.material_mask_path}")
        for label, value in (
            ("elevation minimum", self.elevation_minimum),
            ("elevation maximum", self.elevation_maximum),
            ("sea level", self.sea_level),
            ("beach height", self.beach_height),
            ("rock height", self.rock_height),
            ("rock slope", self.rock_slope_degrees),
            ("spawn clearance", self.spawn_clearance),
            ("maximum spawn slope", self.maximum_spawn_slope_degrees),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.elevation_minimum >= self.elevation_maximum:
            raise ValueError("elevation minimum must be lower than elevation maximum")
        if self.input_minimum is not None and not math.isfinite(self.input_minimum):
            raise ValueError("input minimum must be finite")
        if self.input_maximum is not None and not math.isfinite(self.input_maximum):
            raise ValueError("input maximum must be finite")
        if (
            self.input_minimum is not None
            and self.input_maximum is not None
            and self.input_minimum >= self.input_maximum
        ):
            raise ValueError("input minimum must be lower than input maximum")
        if self.beach_height < 0:
            raise ValueError("beach height must not be negative")
        if not 0 <= self.rock_slope_degrees < 90:
            raise ValueError("rock slope must be within 0..90 degrees")
        if self.spawn_clearance < 0:
            raise ValueError("spawn clearance must not be negative")
        if not 0 <= self.maximum_spawn_slope_degrees < 90:
            raise ValueError("maximum spawn slope must be within 0..90 degrees")
        for code in ("w", "s", "g", "r"):
            _ascii_wire(self.terrain_texture_path(code), 31, "terrain texture path")
        _ascii_wire(self.dummy_texture_path, 31, "dummy terrain texture path")


@dataclass(frozen=True, slots=True)
class OsmSpec(HeightmapSpec):
    """Milestone 3 heightmap plus OpenStreetMap geography import settings."""

    name: str = "cwr_milestone3"
    display_name: str = "CWR Milestone 3"
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    osm_json_path: Path | None = None
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout_seconds: int = 90
    water_depth: float = 5.0
    coastline_blend_cells: int = 2
    road_segment_length: float = 24.5
    max_road_objects: int = DEFAULT_MAX_ROAD_OBJECTS
    max_buildings: int = DEFAULT_MAX_BUILDINGS
    building_minimum_area: float = 20.0
    forest_tree_spacing: float = 50.0
    forest_road_clearance: float = 0.0
    building_ground_clearance: float = 0.10
    forest_ground_clearance: float = 0.15
    iterative_grounding_enabled: bool = True
    iterative_grounding_maximum_adjustment: float = 2.0
    iterative_grounding_strength: float = 0.70
    point_building_footprint: float = 12.0
    max_forest_objects: int = DEFAULT_MAX_FOREST_OBJECTS
    # CLI/GUI builds opt into advisory object-count thresholds. Direct Python
    # callers retain the historical hard/truncating limits unless they opt in.
    advisory_object_limits: bool = False
    include_minor_roads: bool = False
    procedural_gravel_roads: bool = False
    paved_road_model: str = r"o\road\sil25.p3d"
    dirt_road_model: str = r"o\road\ces25.p3d"
    generic_building_model: str = r"O\Hous\domek_sedy.p3d"
    urban_building_model: str = r"data3d\dum_mesto2.p3d"
    industrial_building_model: str = r"O\Hous\hangar_2.p3d"
    forest_tree_model: str = r"data3d\les_su_ctver_pruhozi.p3d"

    def validate(self) -> None:
        HeightmapSpec.validate(self)
        south, west, north, east = self.bbox
        for label, value in (("south", south), ("west", west), ("north", north), ("east", east)):
            if not math.isfinite(value):
                raise ValueError(f"bbox {label} must be finite")
        if not -90.0 <= south < north <= 90.0:
            raise ValueError("bbox latitude order must satisfy -90 <= south < north <= 90")
        if not -180.0 <= west < east <= 180.0:
            raise ValueError("bbox longitude order must satisfy -180 <= west < east <= 180")
        if self.osm_json_path is not None and not self.osm_json_path.is_file():
            raise ValueError(f"OSM JSON does not exist: {self.osm_json_path}")
        if not self.overpass_url.startswith(("http://", "https://")):
            raise ValueError("Overpass URL must use http or https")
        if self.overpass_timeout_seconds < 5 or self.overpass_timeout_seconds > 600:
            raise ValueError("Overpass timeout must be between 5 and 600 seconds")
        for label, value in (
            ("water depth", self.water_depth),
            ("road segment length", self.road_segment_length),
            ("building minimum area", self.building_minimum_area),
            ("forest tree spacing", self.forest_tree_spacing),
            ("point building footprint", self.point_building_footprint),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        if not math.isfinite(self.forest_road_clearance) or self.forest_road_clearance < 0:
            raise ValueError("forest road clearance must be finite and non-negative")
        for label, value in (
            ("building ground clearance", self.building_ground_clearance),
            ("forest ground clearance", self.forest_ground_clearance),
            ("iterative grounding maximum adjustment", self.iterative_grounding_maximum_adjustment),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not math.isfinite(self.iterative_grounding_strength) or not 0.0 <= self.iterative_grounding_strength <= 1.0:
            raise ValueError("iterative grounding strength must be within 0..1")
        if not 0 <= self.coastline_blend_cells <= 32:
            raise ValueError("coastline blend cells must be within 0..32")
        for label, value in (
            ("maximum road objects", self.max_road_objects),
            ("maximum buildings", self.max_buildings),
            ("maximum forest objects", self.max_forest_objects),
        ):
            if value < 0:
                raise ValueError(f"{label} must not be negative")
        for path in (
            self.paved_road_model,
            self.dirt_road_model,
            self.generic_building_model,
            self.urban_building_model,
            self.industrial_building_model,
            self.forest_tree_model,
        ):
            _ascii_wire(path, 75, "model path")
        for code in ("w", "s", "g", "r", "f", "a", "u", "p"):
            _ascii_wire(self.terrain_texture_path(code), 31, "terrain texture path")


def encode_wire_path(value: str, maximum: int, label: str) -> bytes:
    return _ascii_wire(value, maximum, label)


@dataclass(frozen=True, slots=True)
class PlayabilitySpec(OsmSpec):
    """Milestone 4 playability controls with optional regeneration verification."""

    name: str = "cwr_milestone4"
    display_name: str = "CWR Milestone 4"
    road_connection_tolerance: float = 5.0
    maximum_road_grade_percent: float = 12.0
    road_grade_radius: float = 100.0
    building_grade_radius: float = 25.0
    maximum_grade_adjustment: float = 12.0
    transition_cells: int = 2
    asset_roots: tuple[Path, ...] = ()
    strict_assets: bool = False
    osm_asset_mapping_path: Path | None = None
    cache_dir: Path | None = None
    cache_enabled: bool = True
    cache_refresh: bool = False
    town_name_limit: int = 64
    deterministic_seed: str = "cwr-worldgen-milestone4"
    verify_regeneration: bool = False

    def validate(self) -> None:
        OsmSpec.validate(self)
        for label, value in (
            ("road connection tolerance", self.road_connection_tolerance),
            ("maximum road grade", self.maximum_road_grade_percent),
            ("road grade radius", self.road_grade_radius),
            ("building grade radius", self.building_grade_radius),
            ("maximum grade adjustment", self.maximum_grade_adjustment),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.maximum_road_grade_percent > 100:
            raise ValueError("maximum road grade must not exceed 100 percent")
        if not 0 <= self.transition_cells <= 16:
            raise ValueError("transition cells must be within 0..16")
        if not 0 <= self.town_name_limit <= 512:
            raise ValueError("town name limit must be within 0..512")
        if not self.deterministic_seed:
            raise ValueError("deterministic seed must not be empty")
        try:
            self.deterministic_seed.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("deterministic seed must be valid UTF-8") from exc
        for root in self.asset_roots:
            if not Path(root).exists():
                raise ValueError(f"asset root does not exist: {root}")
        if self.osm_asset_mapping_path is not None and not Path(self.osm_asset_mapping_path).is_file():
            raise ValueError(f"OSM asset mapping does not exist: {self.osm_asset_mapping_path}")


@dataclass(frozen=True, slots=True)
class ConstraintPlayabilitySpec(PlayabilitySpec):
    """Milestone 7 unified constraint-based terrain solver controls."""

    name: str = "cwr_milestone7"
    display_name: str = "CWR Milestone 7"
    major_road_grade_percent: float = 8.0
    shoreline_transition_cells: int = 3
    lake_shore_smoothing_cells: int = 8
    lake_shore_maximum_slope_percent: float = 8.0
    building_pad_margin: float = 1.0
    stream_channel_depth: float = 0.35
    river_channel_depth: float = 1.0
    watercourse_minimum_gradient_percent: float = 0.02
    natural_smoothing_strength: float = 0.16
    solver_iterations: int = 20
    world_edge_blend_cells: int = 3
    out_of_bounds_dem_path: Path | None = None

    def validate(self) -> None:
        PlayabilitySpec.validate(self)
        for label, value in (
            ("major road grade", self.major_road_grade_percent),
            ("lake shore maximum slope", self.lake_shore_maximum_slope_percent),
            ("building pad margin", self.building_pad_margin),
            ("stream channel depth", self.stream_channel_depth),
            ("river channel depth", self.river_channel_depth),
            ("watercourse minimum gradient", self.watercourse_minimum_gradient_percent),
            ("natural smoothing strength", self.natural_smoothing_strength),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.major_road_grade_percent > self.maximum_road_grade_percent:
            raise ValueError("major road grade must not exceed the general maximum road grade")
        if self.lake_shore_maximum_slope_percent > 100:
            raise ValueError("lake shore maximum slope must not exceed 100 percent")
        if not 0.0 <= self.natural_smoothing_strength <= 1.0:
            raise ValueError("natural smoothing strength must be within 0..1")
        if not 1 <= self.solver_iterations <= 200:
            raise ValueError("solver iterations must be within 1..200")
        if not 0 <= self.shoreline_transition_cells <= 32:
            raise ValueError("shoreline transition cells must be within 0..32")
        if not 0 <= self.lake_shore_smoothing_cells <= 32:
            raise ValueError("lake shore smoothing cells must be within 0..32")
        if not 0 <= self.world_edge_blend_cells <= 32:
            raise ValueError("world edge blend cells must be within 0..32")
        if self.out_of_bounds_dem_path is not None and not self.out_of_bounds_dem_path.is_file():
            raise ValueError(f"out-of-bounds DEM does not exist: {self.out_of_bounds_dem_path}")
