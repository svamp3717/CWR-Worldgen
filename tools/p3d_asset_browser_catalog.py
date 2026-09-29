"""Asset catalogue helpers for the interactive P3D model/texture browser."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cwr_worldgen import assets
from cwr_worldgen.pbo import is_pbo_path
import p3d_preview_geometry as preview
from p3d_texture_io import decode_paa


@dataclass(frozen=True, slots=True)
class BrowserAsset:
    kind: str
    path: str
    source: str
    size: int
    dependencies: tuple[str, ...] = ()


@dataclass(slots=True)
class BrowserCatalogue:
    roots: tuple[Path, ...]
    assets: tuple[BrowserAsset, ...]
    records: dict[tuple[str, str], assets.AssetRecord]

    @property
    def models(self) -> tuple[BrowserAsset, ...]:
        return tuple(item for item in self.assets if item.kind == "model")

    @property
    def textures(self) -> tuple[BrowserAsset, ...]:
        return tuple(item for item in self.assets if item.kind == "texture")

    def record_for(self, item: BrowserAsset) -> assets.AssetRecord:
        return self.records[(item.source, item.path)]


def _kind(path: str) -> str | None:
    folded = path.casefold()
    if folded.endswith(".p3d"):
        return "model"
    if folded.endswith((".paa", ".pac")):
        return "texture"
    return None


def scan_catalogue(
    roots: Sequence[Path],
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> BrowserCatalogue:
    normalized = tuple(Path(root).expanduser() for root in roots)
    result = assets.scan_assets(
        normalized,
        (),
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh=refresh,
    )
    entries: list[BrowserAsset] = []
    records: dict[tuple[str, str], assets.AssetRecord] = {}
    for record in result.records:
        kind = _kind(record.path)
        if kind is None:
            continue
        entries.append(
            BrowserAsset(
                kind=kind,
                path=record.path,
                source=record.source,
                size=record.size,
                dependencies=record.dependencies,
            )
        )
        records[(record.source, record.path)] = record
    entries.sort(key=lambda item: (item.kind != "model", item.path, item.source.casefold()))
    return BrowserCatalogue(normalized, tuple(entries), records)


def model_source(item: BrowserAsset) -> str:
    """Return the source form expected by TextureResolver."""
    outer = Path(item.source.split("!", 1)[0])
    if is_pbo_path(outer):
        return item.source + "!" + item.path
    return item.source


def load_model(item: BrowserAsset, catalogue: BrowserCatalogue) -> preview.PreviewModel:
    if item.kind != "model":
        raise ValueError(f"not a model asset: {item.path}")
    record = catalogue.record_for(item)
    data = assets.read_asset_record_bytes(record)
    source = model_source(item)
    measurement, points, faces, textures = preview.extract_first_lod_geometry(
        data,
        model_path=item.path,
        source=source,
    )
    edges = preview.collect_edges(faces)
    return preview.PreviewModel(
        item.path,
        source,
        measurement,
        points.astype(np.float32, copy=False),
        faces,
        edges,
        len(points),
        textures,
        "textured mesh" if faces else "vertex fallback",
    )


def load_texture(item: BrowserAsset, catalogue: BrowserCatalogue) -> np.ndarray:
    if item.kind != "texture":
        raise ValueError(f"not a texture asset: {item.path}")
    record = catalogue.record_for(item)
    data = assets.read_asset_record_bytes(record)
    return decode_paa(data, is_paa=item.path.casefold().endswith(".paa"))


def model_texture_paths(
    item: BrowserAsset,
    catalogue: BrowserCatalogue,
) -> tuple[str, ...]:
    if item.kind != "model":
        return ()
    if item.dependencies:
        return tuple(item.dependencies)
    model = load_model(item, catalogue)
    return tuple(
        assets.canonical_asset_path(value)
        for value in model.texture_paths
        if value
    )


def texture_users(
    texture_path: str,
    catalogue: BrowserCatalogue,
) -> tuple[BrowserAsset, ...]:
    target = assets.canonical_asset_path(texture_path)
    basename = target.rsplit("\\", 1)[-1]
    users: list[BrowserAsset] = []
    for item in catalogue.models:
        deps = {
            assets.canonical_asset_path(value)
            for value in item.dependencies
        }
        if target in deps or (
            any("\\" not in value for value in deps)
            and basename in {value.rsplit("\\", 1)[-1] for value in deps}
        ):
            users.append(item)
    return tuple(users)


def filter_assets(
    catalogue: BrowserCatalogue,
    query: str,
    kinds: Iterable[str] = ("model", "texture"),
) -> tuple[BrowserAsset, ...]:
    wanted = set(kinds)
    needle = query.strip().casefold()
    return tuple(
        item
        for item in catalogue.assets
        if item.kind in wanted
        and (
            not needle
            or needle in item.path.casefold()
            or needle in item.source.casefold()
        )
    )
