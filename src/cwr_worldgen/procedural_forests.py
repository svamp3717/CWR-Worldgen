# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Iterable, Mapping

from .cache import cache_key, restore_or_create_file
from .procedural_buildings import (
    _Face,
    _Lod,
    _MLOD_HEADER,
    _NamedSelection,
    _geometry_lod,
    _write_lod,
    BuildingVariantKey,
    inspect_mlod,
)

_VISUAL_LOD = 1.0
_LAND_CONTACT_LOD = 2.0e15


@dataclass(frozen=True, order=True, slots=True)
class ForestClusterVariant:
    name: str
    width_m: float
    length_m: float
    maximum_relief_m: float
    slope_axis: str
    proxy_layout: tuple[tuple[str, float, float, float, float], ...]
    category: str = "interior"

    @property
    def area_m2(self) -> float:
        return self.width_m * self.length_m


@dataclass(frozen=True, order=True, slots=True)
class ForestClusterModelKey:
    variant: str
    grade: float

    @property
    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class ForestClusterAssetResult:
    placements: int
    generated_variants: int
    catalogue_sha256: str
    cache_hits: int
    cache_misses: int
    proxy_models: tuple[str, ...]
    model_files: tuple[str, ...]
    texture_files: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, object]:
        return {
            "placements": self.placements,
            "generated_variants": self.generated_variants,
            "catalogue_sha256": self.catalogue_sha256,
            "proxy_models": self.proxy_models,
            "model_files": self.model_files,
            "texture_files": self.texture_files,
        }


# Performance-oriented interior fallback clusters reuse scaled instances of the
# two stock Everon forest groups. One WRP object therefore represents a dense
# stand without expanding it into many individual tree proxies.
DEFAULT_PROXY_MODELS: tuple[str, ...] = (
    r"data3d\les ctverec pruchozi_T1.p3d",
    r"data3d\les trojuhelnik pruchozi.p3d",
)

_INTERIOR_PROXY_SCALE: tuple[float, ...] = (0.28, 0.40)

# Original Cold War Crisis Data3D vegetation used for the soft forest edge.
# These match the asset family used by the Everon square and triangle forests,
# avoiding a dependency on the Resistance O.pbo vegetation set. They remain
# grouped into generated proxy clusters so one thicket costs one WRP object.
DEFAULT_BORDER_PROXY_MODELS: tuple[str, ...] = (
    r"data3d\ker listnac.p3d",
    r"data3d\ker pichlavej.p3d",
    r"data3d\ker deravej.p3d",
    r"data3d\str smrcicicek.p3d",
)


# Interior undergrowth reuses the same original Data3D bush and small-tree set.
DEFAULT_UNDERGROWTH_PROXY_MODELS: tuple[str, ...] = DEFAULT_BORDER_PROXY_MODELS

# Resistance/Nogova equivalents used when the selected forest profile is the
# Nogova O.pbo family.  Cluster geometry and placement remain identical; only
# the external stock proxies change, so steep/fallback stands do not quietly
# reintroduce Everon/Data3D trees and bushes.
NOGOVA_PROXY_MODELS: tuple[str, ...] = (
    r"o\tree\les_nw_ctver_pruhozi_T1.p3d",
    r"o\tree\les_nw_trojuhelnik.p3d",
)
NOGOVA_BORDER_PROXY_MODELS: tuple[str, ...] = (
    r"o\tree\dd_bush01.p3d",
    r"o\tree\dd_bush02.p3d",
    r"o\tree\dd_bush03.p3d",
    r"o\tree\smrk_maly.p3d",
)
NOGOVA_UNDERGROWTH_PROXY_MODELS: tuple[str, ...] = NOGOVA_BORDER_PROXY_MODELS

# Tall grass and reeds used for deterministic ditch-edge strips. These are
# stock Data3D/Resistance assets and remain external to the generated PBO.
DEFAULT_DITCH_PROXY_MODELS: tuple[str, ...] = (
    r"data3d\ker trs travy.p3d",
    r"data3d\ker trs travy2.p3d",
    r"o\tree\dd_rakosi.p3d",
    r"o\tree\dd_rakosi02.p3d",
)


def _layout(*entries: tuple[int, float, float, float, float]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (
            DEFAULT_PROXY_MODELS[model_index % len(DEFAULT_PROXY_MODELS)],
            x,
            z,
            scale * _INTERIOR_PROXY_SCALE[model_index % len(_INTERIOR_PROXY_SCALE)],
            heading,
        )
        for model_index, x, z, scale, heading in entries
    )


def _border_layout(*entries: tuple[int, float, float, float, float]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (DEFAULT_BORDER_PROXY_MODELS[model_index % len(DEFAULT_BORDER_PROXY_MODELS)], x, z, scale, heading)
        for model_index, x, z, scale, heading in entries
    )




def _undergrowth_layout(*entries: tuple[int, float, float, float, float]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (DEFAULT_UNDERGROWTH_PROXY_MODELS[model_index % len(DEFAULT_UNDERGROWTH_PROXY_MODELS)], x, z, scale, heading)
        for model_index, x, z, scale, heading in entries
    )


def _ditch_layout(*entries: tuple[int, float, float, float, float]) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        (DEFAULT_DITCH_PROXY_MODELS[model_index % len(DEFAULT_DITCH_PROXY_MODELS)], x, z, scale, heading)
        for model_index, x, z, scale, heading in entries
    )


# Layout coordinates are model-local metres. The final Y offset is generated
# from the quantized slope grade, keeping all proxied trees vertical while their
# bases follow a reusable inclined support plane.
FOREST_CLUSTER_VARIANTS: tuple[ForestClusterVariant, ...] = (
    ForestClusterVariant(
        "pine",
        24.0,
        24.0,
        24.0,
        "length",
        _layout(
            (0, 0.0, 0.0, 1.00, 17.0),
            (1, -6.0, 5.0, 0.92, 137.0),
            (1, 6.0, -5.0, 0.88, 257.0),
        ),
    ),
    ForestClusterVariant(
        "strip",
        14.0,
        28.0,
        27.0,
        "width",
        _layout(
            (1, 0.0, -8.0, 0.82, 29.0),
            (1, 0.0, 0.0, 0.88, 149.0),
            (1, 0.0, 8.0, 0.80, 269.0),
        ),
    ),
    ForestClusterVariant(
        "triangle",
        20.0,
        20.0,
        30.0,
        "length",
        _layout(
            (1, 0.0, -3.0, 0.96, 11.0),
            (1, -4.0, 4.0, 0.76, 131.0),
            (1, 4.0, 4.0, 0.74, 251.0),
        ),
    ),
    ForestClusterVariant(
        "irregular",
        15.0,
        16.0,
        36.0,
        "length",
        _layout(
            (1, -3.0, -3.0, 0.76, 43.0),
            (0, 3.0, 3.0, 0.68, 223.0),
        ),
    ),
)


# Reusable Nogova-style border thickets. Local Z follows the forest edge and
# local X points across it. Most proxies are bushes; occasional small spruce
# proxies break up the silhouette without creating a hard tree wall.
FOREST_BORDER_VARIANTS: tuple[ForestClusterVariant, ...] = (
    ForestClusterVariant(
        "border_strip",
        10.0,
        26.0,
        22.0,
        "width",
        _border_layout(
            (0, -2.5, -11.0, 0.95, 13.0), (1, 1.5, -8.0, 1.02, 77.0),
            (2, -1.0, -4.0, 0.92, 142.0), (0, 2.5, 0.0, 1.05, 205.0),
            (1, -2.0, 4.0, 0.98, 269.0), (2, 1.0, 8.0, 0.94, 326.0),
            (3, -1.0, 11.0, 0.82, 41.0),
        ),
        "border",
    ),
    ForestClusterVariant(
        "border_thicket",
        16.0,
        18.0,
        24.0,
        "width",
        _border_layout(
            (0, -5.5, -7.0, 0.98, 8.0), (1, 0.0, -7.5, 1.04, 68.0),
            (2, 5.0, -5.0, 0.92, 126.0), (1, -4.0, -1.0, 0.96, 185.0),
            (0, 1.5, 0.0, 1.05, 243.0), (2, 5.5, 2.0, 0.90, 304.0),
            (3, -2.5, 5.5, 0.84, 347.0), (0, 3.0, 7.0, 0.94, 33.0),
        ),
        "border",
    ),
    ForestClusterVariant(
        "border_corner",
        17.0,
        17.0,
        26.0,
        "width",
        _border_layout(
            (0, -6.0, -6.0, 0.96, 21.0), (1, 0.0, -6.5, 1.00, 81.0),
            (2, 6.0, -5.0, 0.91, 139.0), (0, -4.0, 0.0, 1.03, 201.0),
            (1, 1.0, 0.5, 0.96, 257.0), (3, 5.5, 2.0, 0.83, 315.0),
            (2, -1.0, 6.0, 0.90, 359.0),
        ),
        "border",
    ),
    ForestClusterVariant(
        "border_sparse",
        9.0,
        22.0,
        30.0,
        "width",
        _border_layout(
            (0, -2.0, -9.0, 0.88, 35.0), (2, 1.5, -5.0, 0.92, 103.0),
            (1, -1.0, -1.0, 0.95, 171.0), (0, 2.0, 4.0, 0.90, 239.0),
            (3, -1.5, 8.0, 0.78, 307.0),
        ),
        "border",
    ),
)




# Reusable undergrowth islands sprinkled through forest interiors. The models
# are deliberately low and irregular so they soften the empty ground beneath
# stock forest blocks without forming another hard perimeter.
FOREST_UNDERGROWTH_VARIANTS: tuple[ForestClusterVariant, ...] = (
    ForestClusterVariant(
        "undergrowth_patch", 15.0, 15.0, 18.0, "length",
        _undergrowth_layout(
            (0, -5.0, -4.0, 0.92, 19.0), (1, 1.0, -5.5, 1.02, 83.0),
            (2, 5.0, -1.0, 0.88, 151.0), (0, -3.0, 2.0, 1.05, 217.0),
            (1, 2.5, 4.5, 0.96, 281.0), (3, 0.0, 0.0, 0.72, 337.0),
        ),
        "undergrowth",
    ),
    ForestClusterVariant(
        "undergrowth_strip", 9.0, 20.0, 20.0, "width",
        _undergrowth_layout(
            (0, -2.5, -8.0, 0.90, 31.0), (2, 2.0, -5.0, 0.94, 99.0),
            (1, -1.0, -1.0, 1.02, 167.0), (0, 2.5, 3.5, 0.93, 235.0),
            (2, -2.0, 7.5, 0.89, 303.0),
        ),
        "undergrowth",
    ),
    ForestClusterVariant(
        "undergrowth_sparse", 13.0, 17.0, 24.0, "length",
        _undergrowth_layout(
            (1, -4.0, -6.0, 0.86, 47.0), (0, 4.0, -3.0, 0.91, 119.0),
            (2, -2.0, 2.0, 0.90, 193.0), (3, 3.0, 6.0, 0.68, 269.0),
        ),
        "undergrowth",
    ),
)

# Reusable strips placed along mapped OSM ditches. The local Z axis follows
# the ditch and the proxies sit on both banks, leaving the channel centre open.
DITCH_GRASS_VARIANTS: tuple[ForestClusterVariant, ...] = (
    ForestClusterVariant(
        "ditch_grass",
        8.0,
        18.0,
        18.0,
        "length",
        _ditch_layout(
            (0, -2.8, -7.5, 0.95, 17.0), (1, 2.6, -6.0, 1.02, 91.0),
            (0, -2.4, -2.5, 1.05, 163.0), (1, 2.9, -1.0, 0.96, 237.0),
            (0, -2.7, 3.0, 1.00, 309.0), (1, 2.4, 5.0, 1.04, 41.0),
            (0, -2.2, 7.5, 0.92, 119.0),
        ),
        "ditch",
    ),
    ForestClusterVariant(
        "ditch_reeds",
        10.0,
        16.0,
        20.0,
        "length",
        _ditch_layout(
            (2, -3.5, -6.5, 0.92, 12.0), (3, 3.2, -5.0, 0.98, 78.0),
            (2, -3.0, -1.5, 1.03, 147.0), (3, 3.6, 0.0, 0.94, 218.0),
            (2, -3.4, 4.0, 0.97, 286.0), (3, 3.0, 6.5, 1.00, 351.0),
        ),
        "ditch",
    ),
)

RURAL_VEGETATION_VARIANTS: tuple[ForestClusterVariant, ...] = (
    ForestClusterVariant(
        "orchard_row", 10.0, 28.0, 14.0, "length",
        _border_layout((3, 0.0, -10.0, 0.95, 0.0), (3, 0.0, 0.0, 1.0, 17.0), (3, 0.0, 10.0, 0.92, 351.0)),
        "rural",
    ),
    ForestClusterVariant(
        "vineyard_row", 8.0, 28.0, 12.0, "length",
        _border_layout((0, -1.8, -10.0, 0.72, 0.0), (1, 1.6, -5.0, 0.72, 8.0), (2, -1.4, 0.0, 0.72, 352.0), (0, 1.8, 5.0, 0.72, 4.0), (1, -1.6, 10.0, 0.72, 356.0)),
        "rural",
    ),
    ForestClusterVariant(
        "tree_row", 8.0, 30.0, 18.0, "length",
        _border_layout((3, 0.0, -11.0, 1.05, 3.0), (3, 0.0, 0.0, 1.0, 181.0), (3, 0.0, 11.0, 1.05, 357.0)),
        "rural",
    ),
    ForestClusterVariant(
        "scrub_patch", 18.0, 18.0, 22.0, "width",
        _border_layout((0, -5.0, -5.0, 1.0, 12.0), (1, 2.0, -6.0, 0.94, 91.0), (2, 6.0, 1.0, 0.98, 169.0), (0, -3.0, 5.0, 0.96, 248.0), (1, 3.0, 4.0, 0.90, 319.0)),
        "rural",
    ),
)


ALL_FOREST_CLUSTER_VARIANTS: tuple[ForestClusterVariant, ...] = (
    *FOREST_CLUSTER_VARIANTS,
    *FOREST_BORDER_VARIANTS,
    *FOREST_UNDERGROWTH_VARIANTS,
    *DITCH_GRASS_VARIANTS,
    *RURAL_VEGETATION_VARIANTS,
)

# Reusable incline classes. Grade is rise/run, not degrees.
FOREST_CLUSTER_GRADES: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50, 0.70)


def cluster_variant(name: str) -> ForestClusterVariant:
    for variant in ALL_FOREST_CLUSTER_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(name)


def _profiled_cluster_variant(variant: ForestClusterVariant, proxy_profile: str) -> ForestClusterVariant:
    profile = str(proxy_profile or "everon").strip().casefold()
    if profile == "everon":
        return variant
    if profile != "nogova":
        raise ValueError(f"unsupported forest proxy profile: {proxy_profile!r}")

    replacements = {
        **dict(zip(DEFAULT_PROXY_MODELS, NOGOVA_PROXY_MODELS)),
        **dict(zip(DEFAULT_BORDER_PROXY_MODELS, NOGOVA_BORDER_PROXY_MODELS)),
        **dict(zip(DEFAULT_UNDERGROWTH_PROXY_MODELS, NOGOVA_UNDERGROWTH_PROXY_MODELS)),
    }
    remapped = tuple(
        (replacements.get(model_path, model_path), x, z, scale, heading)
        for model_path, x, z, scale, heading in variant.proxy_layout
    )
    return replace(variant, proxy_layout=remapped)


def quantize_cluster_grade(grade: float) -> float:
    value = max(0.0, float(grade))
    return min(FOREST_CLUSTER_GRADES, key=lambda candidate: (abs(candidate - value), candidate))


def cluster_model_path(world_name: str, variant_name: str, grade: float) -> str:
    key = ForestClusterModelKey(variant_name, quantize_cluster_grade(grade))
    grade_label = int(round(key.grade * 100.0))
    category = cluster_variant(variant_name).category
    prefix = {"interior": "c", "border": "b", "undergrowth": "u", "ditch": "g", "rural": "r"}[category]
    return rf"{world_name}\f\{prefix}_{variant_name}_{grade_label:02d}.p3d"


def cluster_proxy_models(
    *, include_border: bool = True, include_undergrowth: bool = True, include_ditch: bool = True
) -> tuple[str, ...]:
    models = DEFAULT_PROXY_MODELS
    if include_border:
        models += DEFAULT_BORDER_PROXY_MODELS
    if include_undergrowth:
        models += DEFAULT_UNDERGROWTH_PROXY_MODELS
    if include_ditch:
        models += DEFAULT_DITCH_PROXY_MODELS
    return tuple(dict.fromkeys(models))


def is_generated_cluster_model(world_name: str, model_path: str) -> bool:
    folded = model_path.casefold()
    prefixes = (
        (world_name + r"\f\c_").casefold(),
        (world_name + r"\f\b_").casefold(),
        (world_name + r"\f\u_").casefold(),
        (world_name + r"\f\g_").casefold(),
        (world_name + r"\f\r_").casefold(),
    )
    return folded.endswith(".p3d") and any(folded.startswith(prefix) for prefix in prefixes)


def _proxy_selection_name(model_path: str, index: int) -> str:
    path = model_path.replace("/", "\\")
    if path.casefold().endswith(".p3d"):
        path = path[:-4]
    if not path.startswith("\\"):
        path = "\\" + path
    return f"proxy:{path}.{index:02d}"


def _proxy_visual_lod(variant: ForestClusterVariant, grade: float) -> _Lod:
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    selections: list[_NamedSelection] = []
    normal = ((0.0, 0.0, 1.0),)
    grade = quantize_cluster_grade(grade)

    for proxy_index, (model_path, x, z, scale, heading) in enumerate(variant.proxy_layout, start=1):
        # Keep the proxied stock tree vertical. Only its base height varies along
        # the reusable support plane. For strip clusters the short X axis follows
        # the hill; for all other shapes the local Z axis follows it.
        y = grade * (x if variant.slope_axis == "width" else z)
        angle = math.radians(heading)
        aside = (math.cos(angle) * scale, 0.0, -math.sin(angle) * scale)
        up = (0.0, scale, 0.0)
        point_start = len(points)
        face_index = len(faces)
        points.extend((
            (x, y, z),
            (x + aside[0], y + aside[1], z + aside[2]),
            (x + up[0], y + up[1], z + up[2]),
        ))
        faces.append(_Face("", (
            (point_start, 0, 0.0, 0.0),
            (point_start + 1, 0, 1.0, 0.0),
            (point_start + 2, 0, 0.0, 1.0),
        ), flags=0x10))
        point_weights = bytearray(len(points))
        # Existing selections need extending after later points/faces are added;
        # defer construction until the final array sizes are known.
        selections.append(_NamedSelection(
            _proxy_selection_name(model_path, proxy_index),
            bytes((point_start, point_start + 1, point_start + 2)),
            bytes((face_index,)),
        ))

    final_selections: list[_NamedSelection] = []
    for selection in selections:
        point_indices = tuple(selection.point_weights)
        face_indices = tuple(selection.face_flags)
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in point_indices:
            point_weights[point_index] = 1
        for face_index in face_indices:
            face_flags[face_index] = 1
        final_selections.append(_NamedSelection(selection.name, bytes(point_weights), bytes(face_flags)))

    return _Lod(
        tuple(points),
        normal,
        tuple(faces),
        _VISUAL_LOD,
        selections=tuple(final_selections),
        properties=(("autocenter", "0"),),
    )


def _cluster_geometry_lod(variant: ForestClusterVariant) -> _Lod:
    # A very small closed component keeps a valid Geometry LOD and fixes the
    # cluster origin without imposing a giant invisible collision box over the
    # stand. The proxied stock vegetation remains the visible and physical detail.
    key = BuildingVariantKey("residential", "flat", 0.25, 0.25, 0.25)
    lod = _geometry_lod(key)
    model_class = "forest" if variant.category == "interior" else "bushsoft"
    return _Lod(
        lod.points,
        lod.normals,
        lod.faces,
        lod.resolution,
        lod.mass_per_point,
        lod.selections,
        (("autocenter", "0"), ("class", model_class)),
    )


def _land_contact_lod(variant: ForestClusterVariant, grade: float) -> _Lod:
    grade = quantize_cluster_grade(grade)
    points = tuple(
        (
            x,
            grade * (x if variant.slope_axis == "width" else z),
            z,
        )
        for _model, x, z, _scale, _heading in variant.proxy_layout
    )
    return _Lod(points, (), (), _LAND_CONTACT_LOD)


def write_forest_cluster_mlod(path: Path, variant: ForestClusterVariant, grade: float) -> None:
    lods = (
        _proxy_visual_lod(variant, grade),
        _cluster_geometry_lod(variant),
        _land_contact_lod(variant, grade),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
        for lod in lods:
            _write_lod(stream, lod)


class ProceduralForestClusterLibrary:
    def __init__(
        self,
        world_name: str,
        *,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_refresh: bool = False,
        proxy_profile: str = "everon",
    ) -> None:
        self.world_name = world_name
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_refresh = cache_refresh
        self.proxy_profile = str(proxy_profile or "everon").strip().casefold()
        if self.proxy_profile not in {"everon", "nogova"}:
            raise ValueError(f"unsupported forest proxy profile: {proxy_profile!r}")
        self.cache_hits = 0
        self.cache_misses = 0
        self._usage: Counter[ForestClusterModelKey] = Counter()

    def register_model(self, model_path: str) -> None:
        if not self.is_generated_model(model_path):
            return
        stem = model_path.rsplit("\\", 1)[-1][2:-4]
        variant_name, grade_label = stem.rsplit("_", 1)
        self._usage[ForestClusterModelKey(variant_name, int(grade_label) / 100.0)] += 1

    def register_models(self, model_paths: Iterable[str]) -> None:
        for model_path in model_paths:
            self.register_model(model_path)

    def is_generated_model(self, model_path: str) -> bool:
        return is_generated_cluster_model(self.world_name, model_path)

    def required_proxy_models(self) -> tuple[str, ...]:
        models: set[str] = set()
        for key in self._usage:
            variant = _profiled_cluster_variant(cluster_variant(key.variant), self.proxy_profile)
            models.update(entry[0] for entry in variant.proxy_layout)
        return tuple(sorted(models, key=str.casefold))

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> ForestClusterAssetResult:
        models: list[dict[str, object]] = []
        for key in sorted(self._usage):
            variant = _profiled_cluster_variant(cluster_variant(key.variant), self.proxy_profile)
            wire = cluster_model_path(self.world_name, key.variant, key.grade)
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            destination = source_dir / relative
            asset_key = cache_key(
                "procedural-forest-cluster-model-v6-profiled-stock-vegetation",
                {
                    "world_name": self.world_name,
                    "proxy_profile": self.proxy_profile,
                    "key": asdict(key),
                    "variant": asdict(variant),
                    "proxy_models": tuple(entry[0] for entry in variant.proxy_layout),
                },
            )
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.p3d" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=destination,
                producer=lambda target, variant=variant, grade=key.grade: write_forest_cluster_mlod(target, variant, grade),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            summary = inspect_mlod(destination)
            proxy_names = tuple(
                name
                for lod_names in summary.selection_names
                for name in lod_names
                if name.casefold().startswith("proxy:")
            )
            if len(proxy_names) != len(variant.proxy_layout):
                raise ValueError(f"forest cluster {key.variant!r} lost proxy selections")
            models.append({
                "key": asdict(key),
                "model_path": wire,
                "relative_path": relative,
                "usage_count": self._usage[key],
                "proxy_count": len(proxy_names),
                "proxy_models": sorted({entry[0] for entry in variant.proxy_layout}),
                "sha256": sha256(destination.read_bytes()).hexdigest(),
            })

        document: dict[str, object] = {
            "schema": 1,
            "generator": "cwr-worldgen procedural forest clusters",
            "proxy_profile": self.proxy_profile,
            "placements": sum(self._usage.values()),
            "generated_variants": len(models),
            "proxy_models": list(self.required_proxy_models()),
            "models": models,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        embedded = source_dir / "f" / "clusters.json"
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return ForestClusterAssetResult(
            placements=sum(self._usage.values()),
            generated_variants=len(models),
            catalogue_sha256=digest,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            proxy_models=self.required_proxy_models(),
            model_files=tuple(str(item["relative_path"]) for item in models),
        )
