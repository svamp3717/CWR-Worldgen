# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import blake2s, sha256
from pathlib import Path
import json
import math
import struct
from typing import Mapping, Sequence

from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .cache import cache_key, restore_or_create_file
from .model import WorldObject
from .osm import (
    BboxProjection,
    BuildingPlacementPlan,
    OsmDataset,
    OsmLineFeature,
    OsmRaster,
    ROADSIDE_NUDGE_DISTANCE_METRES,
    _advisory_object_limit,
    _mask_at,
    _sample_elevation,
    _oriented_rectangle,
    _polygon_elevation_extrema,
    _square_elevation_extrema,
    forest_block_intersects_road_corridors,
    nearest_road_heading,
    nudge_point_away_from_road,
    project_road_corridors,
)
from .paa import inspect_paa, write_rgb_dxt1_paa
from .parallel_assets import process_asset_tasks
from .procedural_buildings import footprint_from_polygon

_MLOD_HEADER = struct.Struct("<4sBBHI")
_SP3X_HEADER = struct.Struct("<4siiiiii")
_POINT = struct.Struct("<fffi")
_NORMAL = struct.Struct("<fff")
_FACE_VERTEX = struct.Struct("<iiff")
_FACE_TRAILER = struct.Struct("<i")
_TAG_HEADER = struct.Struct("<64si")
_FLOAT = struct.Struct("<f")

GRAVE_MODELS: tuple[str, ...] = (
    r"O\Hous\Nahrobek1.p3d",
    r"O\Hous\Nahrobek2.p3d",
    r"O\Hous\Nahrobek3.p3d",
    r"O\Hous\Nahrobek4.p3d",
    r"O\Hous\Nahrobek5.p3d",
)

@dataclass(frozen=True, slots=True)
class GraveGroundingProfile:
    """Conservative stock-model support metadata for final WRP grounding.

    ``origin_lift_metres`` is the distance from the model origin down to the
    visible base. The footprint dimensions describe the support area that must
    clear the final, quantized terrain after heading rotation.
    """

    origin_lift_metres: float
    width_metres: float
    length_metres: float


# The five stock Nahrobek models do not share one useful origin/support shape.
# Keep separate profiles so visual testing can tune one stone without moving all
# of them. These are deliberately conservative because the previous shared
# 0.55 m correction still left the models visibly buried in CWA.
GRAVE_MODEL_GROUNDING_PROFILES: Mapping[str, GraveGroundingProfile] = {
    r"o\hous\nahrobek1.p3d": GraveGroundingProfile(0.74, 0.75, 1.10),
    r"o\hous\nahrobek2.p3d": GraveGroundingProfile(0.80, 0.80, 1.20),
    r"o\hous\nahrobek3.p3d": GraveGroundingProfile(0.86, 0.90, 1.30),
    r"o\hous\nahrobek4.p3d": GraveGroundingProfile(0.72, 0.75, 1.05),
    r"o\hous\nahrobek5.p3d": GraveGroundingProfile(0.82, 0.85, 1.25),
}
DEFAULT_GRAVE_GROUNDING_PROFILE = GraveGroundingProfile(0.80, 0.85, 1.20)
# Retain the historical public constant as the fallback value for external code.
GRAVE_MODEL_ORIGIN_LIFT_METRES = DEFAULT_GRAVE_GROUNDING_PROFILE.origin_lift_metres


def grave_grounding_profile(model_path: str) -> GraveGroundingProfile:
    canonical = str(model_path).replace("/", "\\").strip().casefold()
    return GRAVE_MODEL_GROUNDING_PROFILES.get(canonical, DEFAULT_GRAVE_GROUNDING_PROFILE)


@dataclass(frozen=True, order=True, slots=True)
class SiteVariantKey:
    kind: str
    width_m: float
    length_m: float

    @property
    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SemanticGenerationResult:
    objects: tuple[WorldObject, ...]
    bus_stop_objects: int
    sports_pitch_objects: int
    parking_objects: int
    rejected_site_relief: int
    rejected_landmarks: int
    maximum_site_relief: float
    grave_objects: int = 0
    cemetery_sites: int = 0
    rejected_graves: int = 0


@dataclass(frozen=True, slots=True)
class SiteAssetResult:
    generated_variants: int
    placements: int
    catalogue_sha256: str
    texture_files: tuple[str, ...]
    cache_hits: int = 0
    cache_misses: int = 0


def _cstring(value: str, size: int) -> bytes:
    encoded = value.encode("ascii")
    if len(encoded) >= size:
        raise ValueError(f"wire string exceeds {size - 1} bytes")
    return encoded + bytes(size - len(encoded))


def _write_tag(stream, name: str, payload: bytes) -> None:
    stream.write(_TAG_HEADER.pack(_cstring(name, 64), len(payload)))
    stream.write(payload)


def _write_site_mlod(path: Path, key: SiteVariantKey, texture: str) -> None:
    half_width = key.width_m / 2.0
    half_length = key.length_m / 2.0
    points = (
        (-half_width, 0.0, -half_length),
        (half_width, 0.0, -half_length),
        (half_width, 0.0, half_length),
        (-half_width, 0.0, half_length),
    )
    normals = ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    faces = (
        ((0, 0, 0.0, 1.0), (3, 0, 0.0, 0.0), (2, 0, 1.0, 0.0), (1, 0, 1.0, 1.0)),
        ((1, 1, 1.0, 1.0), (2, 1, 1.0, 0.0), (3, 1, 0.0, 0.0), (0, 1, 0.0, 1.0)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
        stream.write(_SP3X_HEADER.pack(b"SP3X", 28, 1, len(points), len(normals), len(faces), 0))
        for point in points:
            stream.write(_POINT.pack(*point, 0))
        for normal in normals:
            stream.write(_NORMAL.pack(*normal))
        for face in faces:
            stream.write(_cstring(texture, 32))
            stream.write(struct.pack("<i", 4))
            for vertex in face:
                stream.write(_FACE_VERTEX.pack(*vertex))
            stream.write(_FACE_TRAILER.pack(0))
        stream.write(b"TAGG")
        _write_tag(stream, "#EndOfFile#", b"")
        stream.write(_FLOAT.pack(1.0))


def _site_texture(kind: str, size: int = 128) -> Image.Image:
    if kind == "sports_pitch":
        image = Image.new("RGB", (size, size), (75, 119, 62))
        draw = ImageDraw.Draw(image)
        for y in range(0, size, 8):
            draw.rectangle((0, y, size, min(size, y + 3)), fill=(70, 111, 58))
        white = (226, 226, 209)
        draw.rectangle((5, 5, size - 6, size - 6), outline=white, width=3)
        draw.line((size // 2, 5, size // 2, size - 6), fill=white, width=3)
        draw.ellipse((size // 2 - 13, size // 2 - 13, size // 2 + 13, size // 2 + 13), outline=white, width=3)
        draw.rectangle((5, size // 2 - 24, 26, size // 2 + 24), outline=white, width=3)
        draw.rectangle((size - 27, size // 2 - 24, size - 6, size // 2 + 24), outline=white, width=3)
        return image
    image = Image.new("RGB", (size, size), (91, 94, 91))
    draw = ImageDraw.Draw(image)
    for y in range(0, size, 16):
        draw.line((0, y, size, y), fill=(84, 87, 84), width=1)
    for x in range(10, size - 8, 20):
        draw.line((x, 10, x, size - 10), fill=(219, 216, 197), width=2)
        draw.line((x + 13, 10, x + 13, size - 10), fill=(219, 216, 197), width=2)
    draw.line((0, size // 2, size, size // 2), fill=(198, 194, 178), width=2)
    return image


@dataclass(frozen=True, slots=True)
class _SiteAssetTask:
    key: SiteVariantKey
    wire: str
    relative: str
    path: Path
    texture: str
    cache_path: Path | None
    cache_enabled: bool
    cache_refresh: bool
    usage_count: int


def _write_site_asset_task(task: _SiteAssetTask) -> tuple[dict[str, object], bool]:
    hit = restore_or_create_file(
        cache_path=task.cache_path,
        destination=task.path,
        producer=lambda target: _write_site_mlod(target, task.key, task.texture),
        enabled=task.cache_enabled,
        refresh=task.cache_refresh,
    )
    return ({
        "key": asdict(task.key),
        "model_path": task.wire,
        "relative_path": task.relative,
        "usage_count": task.usage_count,
        "sha256": sha256(task.path.read_bytes()).hexdigest(),
    }, hit)


class ProceduralSiteLibrary:
    def __init__(
        self,
        world_name: str,
        maximum_variants: int = 64,
        *,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_refresh: bool = False,
    ) -> None:
        self.world_name = world_name
        self.maximum_variants = maximum_variants
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_refresh = cache_refresh
        self.cache_hits = 0
        self.cache_misses = 0
        self._usage: Counter[SiteVariantKey] = Counter()
        self._mapping: dict[SiteVariantKey, SiteVariantKey] = {}

    @staticmethod
    def key_for(kind: str, width: float, length: float) -> SiteVariantKey:
        width, length = sorted((width, length))
        return SiteVariantKey(
            kind,
            round(max(6.0, min(100.0, round(width / 4.0) * 4.0)), 3),
            round(max(8.0, min(180.0, round(length / 4.0) * 4.0)), 3),
        )

    def prepare(self, dataset: OsmDataset, projection: BboxProjection) -> None:
        requested: Counter[SiteVariantKey] = Counter()
        for feature in dataset.sites:
            kind = feature.tags.get("site", "")
            if kind not in {"sports_pitch", "parking"}:
                continue
            for polygon in feature.polygons:
                points = [projection.to_world(point) for point in polygon.outer[:-1]]
                if len(points) < 3:
                    continue
                footprint = footprint_from_polygon(points)
                requested[self.key_for(kind, footprint.width_m, footprint.length_m)] += 1
        selected = sorted(requested, key=lambda key: (-requested[key], key))[: self.maximum_variants]
        for key in sorted(requested):
            candidates = [candidate for candidate in selected if candidate.kind == key.kind] or selected or [key]
            self._mapping[key] = min(candidates, key=lambda candidate: (
                abs(candidate.width_m - key.width_m) + abs(candidate.length_m - key.length_m), candidate
            ))

    def model_path(self, key: SiteVariantKey) -> str:
        return rf"{self.world_name}\s\s_{key.digest}.p3d"

    def is_generated_model(self, path: str) -> bool:
        return path.casefold().startswith((self.world_name + r"\s\s_").casefold()) and path.casefold().endswith(".p3d")

    def place(self, kind: str, width: float, length: float) -> tuple[str, SiteVariantKey]:
        requested = self.key_for(kind, width, length)
        selected = self._mapping.get(requested, requested)
        self._usage[selected] += 1
        return self.model_path(selected), selected

    def _texture_path(self, kind: str) -> str:
        return rf"{self.world_name}\s\{'f' if kind == 'sports_pitch' else 'p'}.paa"

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> SiteAssetResult:
        texture_files: list[str] = []
        for kind in sorted({key.kind for key in self._usage}):
            wire = self._texture_path(kind)
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            path = source_dir / relative
            asset_key = cache_key("procedural-site-texture-v1", {"kind": kind})
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.paa" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=path,
                producer=lambda target, kind=kind: write_rgb_dxt1_paa(target, _site_texture(kind)),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            inspect_paa(path)
            texture_files.append(relative)
        model_tasks: list[_SiteAssetTask] = []
        for key in sorted(self._usage):
            wire = self.model_path(key)
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            path = source_dir / relative
            asset_key = cache_key(
                "procedural-site-model-v1",
                {"world_name": self.world_name, "variant": asdict(key)},
            )
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.p3d" if self.cache_dir else None
            model_tasks.append(_SiteAssetTask(
                key=key, wire=wire, relative=relative, path=path,
                texture=self._texture_path(key.kind), cache_path=cached,
                cache_enabled=self.cache_enabled, cache_refresh=self.cache_refresh,
                usage_count=self._usage[key],
            ))
        model_results = process_asset_tasks(_write_site_asset_task, model_tasks)
        models: list[dict[str, object]] = []
        for model, hit in model_results:
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            models.append(model)

        document = {
            "schema": 1,
            "generator": "cwr-worldgen procedural semantic sites",
            "placements": sum(self._usage.values()),
            "generated_variants": len(models),
            "textures": sorted(texture_files),
            "models": models,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        embedded = source_dir / "s" / "sites.json"
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return SiteAssetResult(len(models), sum(self._usage.values()), digest, tuple(sorted(texture_files)), self.cache_hits, self.cache_misses)


def _nearest_road_heading(dataset: OsmDataset, projection: BboxProjection, x: float, z: float) -> float:
    best: tuple[float, float] | None = None
    for road in dataset.roads:
        points = [projection.to_world(point) for point in road.points]
        for start, end in zip(points, points[1:]):
            dx, dz = end[0] - start[0], end[1] - start[1]
            length2 = dx * dx + dz * dz
            if length2 <= 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - start[0]) * dx + (z - start[1]) * dz) / length2))
            px, pz = start[0] + t * dx, start[1] + t * dz
            distance2 = (x - px) ** 2 + (z - pz) ** 2
            heading = math.degrees(math.atan2(dx, dz)) % 360.0
            candidate = (distance2, heading)
            if best is None or candidate < best:
                best = candidate
    return best[1] if best is not None else 0.0


def _cemetery_grave_candidates(
    points: Sequence[tuple[float, float]],
    *,
    spacing: float,
    inset: float,
    seed: str,
) -> tuple[tuple[str, float, float, float], ...]:
    if len(points) < 3:
        return ()
    shape = Polygon(points)
    if shape.is_empty or shape.area <= 0.0:
        return ()
    usable = shape.buffer(-max(0.0, inset), join_style=2)
    if usable.is_empty or usable.area <= 1.0:
        usable = shape
    footprint = footprint_from_polygon(points)
    heading = footprint.heading_degrees
    angle = math.radians(heading)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))
    centre = shape.centroid

    local_points = []
    for x, z in points:
        dx, dz = x - centre.x, z - centre.y
        local_points.append((dx * width_axis[0] + dz * width_axis[1], dx * length_axis[0] + dz * length_axis[1]))
    min_width = min(item[0] for item in local_points)
    max_width = max(item[0] for item in local_points)
    min_length = min(item[1] for item in local_points)
    max_length = max(item[1] for item in local_points)
    step = max(2.0, float(spacing))
    root = blake2s(seed.encode("utf-8"), digest_size=16).digest()
    width_offset = (int.from_bytes(root[:4], "little") / 2**32) * step
    length_offset = (int.from_bytes(root[4:8], "little") / 2**32) * step
    candidates: list[tuple[str, float, float, float]] = []
    row = 0
    local_length = min_length + length_offset
    while local_length <= max_length + 1e-9:
        column = 0
        local_width = min_width + width_offset + (0.5 * step if row % 2 else 0.0)
        while local_width <= max_width + 1e-9:
            identity = f"{seed}:{row}:{column}"
            digest = blake2s(identity.encode("utf-8"), digest_size=8).digest()
            jitter_x = ((int.from_bytes(digest[:2], "little") / 65535.0) - 0.5) * step * 0.28
            jitter_z = ((int.from_bytes(digest[2:4], "little") / 65535.0) - 0.5) * step * 0.28
            width_value = local_width + jitter_x
            length_value = local_length + jitter_z
            x = centre.x + width_value * width_axis[0] + length_value * length_axis[0]
            z = centre.y + width_value * width_axis[1] + length_value * length_axis[1]
            point = Point(x, z)
            if usable.covers(point):
                turn = 180.0 if digest[4] & 1 else 0.0
                heading_jitter = ((digest[5] / 255.0) - 0.5) * 8.0
                candidates.append((identity, x, z, (heading + turn + heading_jitter) % 360.0))
            column += 1
            local_width += step
        row += 1
        local_length += step
    return tuple(candidates)


def generate_semantic_objects(
    dataset: OsmDataset,
    projection: BboxProjection,
    elevations: Sequence[float],
    spec,
    site_library: ProceduralSiteLibrary,
    *,
    starting_object_id: int,
    raster: OsmRaster | None = None,
    building_placement_plans: Sequence[BuildingPlacementPlan] = (),
) -> SemanticGenerationResult:
    objects: list[WorldObject] = []
    next_id = starting_object_id
    bus_count = pitch_count = parking_count = rejected_sites = rejected_landmarks = 0
    grave_count = cemetery_count = rejected_graves = 0
    maximum_relief = 0.0
    bus_model = str(getattr(spec, "bus_stop_model", r"o\misc\aut_z_st.p3d"))
    bus_footprint = max(0.5, float(getattr(spec, "bus_stop_footprint", 1.6)))
    bus_clearance = max(0.0, float(getattr(spec, "bus_stop_ground_clearance", 0.12)))
    landmark_warning_threshold = max(0, int(getattr(spec, "maximum_landmark_objects", 1000)))
    advisory_limits = bool(getattr(spec, "advisory_object_limits", False))
    max_landmarks = _advisory_object_limit(landmark_warning_threshold, enabled=advisory_limits)
    if bool(getattr(spec, "bus_stops_enabled", False)):
        for landmark in sorted(dataset.landmarks, key=lambda item: item.osm_key):
            if bus_count >= max_landmarks:
                break
            if landmark.tags.get("landmark") != "bus_stop":
                continue
            x, z = projection.to_world(landmark.point)
            if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
                rejected_landmarks += 1
                continue
            road_heading = nearest_road_heading(dataset, projection, x, z)
            heading = (road_heading + 90.0) % 360.0
            x, z = nudge_point_away_from_road(
                dataset,
                projection,
                x,
                z,
                distance=ROADSIDE_NUDGE_DISTANCE_METRES,
                fallback_heading=road_heading,
                world_size=spec.world_size,
            )
            # Use the highest terrain sample under the sign's support footprint.
            # A centre-only sample can bury the post on sloping or coarsely
            # interpolated terrain even when the nominal clearance is positive.
            _minimum, support_height = _square_elevation_extrema(
                elevations, spec.cells, spec.cell_size, x, z, bus_footprint
            )
            y = support_height + bus_clearance
            objects.append(WorldObject(next_id, bus_model, x, y, z, heading))
            next_id += 1
            bus_count += 1

    if bool(getattr(spec, "cemeteries_enabled", True)):
        grave_models = tuple(getattr(spec, "grave_models", GRAVE_MODELS)) or GRAVE_MODELS
        grave_warning_threshold = max(0, int(getattr(spec, "maximum_grave_objects", 12000)))
        grave_limit = _advisory_object_limit(grave_warning_threshold, enabled=advisory_limits)
        grave_spacing = max(2.0, float(getattr(spec, "grave_spacing", 3.5)))
        grave_inset = max(0.0, float(getattr(spec, "grave_inset", 2.0)))
        grave_footprint = max(0.5, float(getattr(spec, "grave_footprint", 1.2)))
        grave_clearance = float(getattr(spec, "grave_ground_clearance", 0.12))
        grave_road_clearance = max(0.0, float(getattr(spec, "grave_road_clearance", 1.0)))
        grave_building_clearance = max(0.0, float(getattr(spec, "grave_building_clearance", 1.5)))
        road_corridors = project_road_corridors(dataset, projection, spec)
        building_exclusions = []
        for plan in building_placement_plans:
            if len(plan.support_polygon) >= 3:
                shape = Polygon(plan.support_polygon)
                if not shape.is_empty:
                    building_exclusions.append(
                        shape.buffer(grave_building_clearance, join_style=2)
                    )
        building_exclusion = unary_union(building_exclusions) if building_exclusions else None
        seed = str(getattr(spec, "deterministic_seed", "cwr-world"))
        for feature in sorted(dataset.sites, key=lambda item: item.osm_key):
            if grave_count >= grave_limit:
                break
            if feature.tags.get("site") != "cemetery":
                continue
            site_placed = False
            for polygon_index, polygon in enumerate(feature.polygons):
                points = [projection.to_world(point) for point in polygon.outer[:-1]]
                candidates = _cemetery_grave_candidates(
                    points,
                    spacing=grave_spacing,
                    inset=grave_inset,
                    seed=f"{seed}:{feature.osm_key}:{polygon_index}",
                )
                for identity, x, z, heading in candidates:
                    if grave_count >= grave_limit:
                        break
                    if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
                        rejected_graves += 1
                        continue
                    if raster is not None and (
                        _mask_at(raster.water, spec.cells, spec.world_size, x, z)
                        or _mask_at(raster.buildings, spec.cells, spec.world_size, x, z)
                        or _mask_at(raster.roads, spec.cells, spec.world_size, x, z)
                    ):
                        rejected_graves += 1
                        continue
                    digest = blake2s(identity.encode("utf-8"), digest_size=4).digest()
                    model = grave_models[int.from_bytes(digest, "little") % len(grave_models)]
                    profile = grave_grounding_profile(model)
                    support_width = max(grave_footprint, profile.width_metres)
                    support_length = max(grave_footprint, profile.length_metres)
                    support_polygon = _oriented_rectangle(
                        x, z, support_width, support_length, heading, margin=0.08
                    )
                    grave_shape = Polygon(support_polygon)
                    if (
                        building_exclusion is not None
                        and not building_exclusion.is_empty
                        and building_exclusion.intersects(grave_shape)
                    ):
                        rejected_graves += 1
                        continue
                    if forest_block_intersects_road_corridors(
                        road_corridors,
                        x,
                        z,
                        block_size=max(support_width, support_length) + 2.0 * grave_road_clearance,
                    ):
                        rejected_graves += 1
                        continue
                    # Ground the selected stock model against its own rotated
                    # support footprint on the final quantized terrain. The five
                    # Nahrobek variants have different origins and base sizes; a
                    # shared centre-point correction was never going to age well.
                    _minimum, support_height = _polygon_elevation_extrema(
                        elevations, spec.cells, spec.cell_size, support_polygon
                    )
                    y = support_height + profile.origin_lift_metres + grave_clearance
                    visible_base = y - profile.origin_lift_metres
                    if visible_base + 1e-7 < support_height + grave_clearance:
                        raise RuntimeError(
                            f"grave grounding validation failed for {model}: "
                            f"base={visible_base:.4f}, terrain={support_height:.4f}"
                        )
                    objects.append(WorldObject(next_id, model, x, y, z, heading))
                    next_id += 1
                    grave_count += 1
                    site_placed = True
            cemetery_count += int(site_placed)

    # Sports pitches are terrain semantics, not generated slab objects. The
    # surface pass paints them with the selected terrain material (Nogova uses
    # the same standard green grass texture as ordinary grass), which avoids
    # z-fighting/colour mismatches and respects the terrain profile. Parking
    # polygons likewise remain semantic-only for now.

    return SemanticGenerationResult(
        objects=tuple(objects),
        bus_stop_objects=bus_count,
        sports_pitch_objects=pitch_count,
        parking_objects=parking_count,
        rejected_site_relief=rejected_sites,
        rejected_landmarks=rejected_landmarks,
        maximum_site_relief=maximum_relief,
        grave_objects=grave_count,
        cemetery_sites=cemetery_count,
        rejected_graves=rejected_graves,
    )
