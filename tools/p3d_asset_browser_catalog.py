"""Asset catalogue helpers for the interactive P3D model/texture browser."""
from __future__ import annotations

from collections import OrderedDict
import os
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cwr_worldgen import assets
from cwr_worldgen.pbo import is_pbo_path
import p3d_preview_geometry as preview
from p3d_texture_io import PaaFormatInfo, decode_paa, inspect_paa_format


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


def _browser_cache_dir() -> Path:
    """Keep catalogue fingerprints between GUI launches.

    The asset scanner otherwise reparses every PBO and hashes/decompresses
    every P3D just to display the model list. Respect the Windows cache
    location and XDG_CACHE_HOME, without writing beside users' game assets.
    """
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "CWR-Worldgen" / "p3d-browser"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg_cache) if xdg_cache else Path.home() / ".cache") / "cwr-worldgen" / "p3d-browser"


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
        cache_dir=cache_dir if cache_dir is not None or not use_cache else _browser_cache_dir(),
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


def load_model(
    item: BrowserAsset, catalogue: BrowserCatalogue, *, lod_quality: str = "full",
    source_data: bytes | None = None,
) -> preview.PreviewModel:
    if item.kind != "model":
        raise ValueError(f"not a model asset: {item.path}")
    read_start = perf_counter()
    data = (
        source_data if source_data is not None
        else assets.read_asset_record_bytes(catalogue.record_for(item))
    )
    read_seconds = 0.0 if source_data is not None else perf_counter() - read_start
    parse_start = perf_counter()
    source = model_source(item)
    model = preview.build_preview_model(
        data, model_path=item.path, source=source,
        lod_quality=lod_quality, max_points=25_000,
    )
    model.read_seconds = read_seconds
    model.parse_seconds = perf_counter() - parse_start
    model.archive_cached = source_data is not None
    # The browser's cache moves these bytes into its bounded raw-byte LRU.
    model.source_bytes = data
    if model.render_mode == "bounds only":
        # Fallback bounds cannot reveal textures; use the scan's reference list.
        model.texture_paths = tuple(
            path for path in item.dependencies
            if path.casefold().endswith((".paa", ".pac"))
        )
    return model


class ModelPreviewCache:
    """Bounded in-memory LRU of already-parsed preview geometry.

    Loading an ODOL can decompress substantial vertex buffers and read a PBO.
    Keep a handful of commonly revisited models until the source selection
    changes; skip caching models too large to fit the configured budget.
    """

    def __init__(
        self, max_models: int = 8, max_bytes: int = 96 * 1024 * 1024,
        raw_limit: int = 64 * 1024 * 1024, max_raw_models: int = 4,
    ):
        self.max_models = max(0, int(max_models))
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[
            tuple[str, str, str], tuple[preview.PreviewModel, int]
        ] = OrderedDict()
        self.bytes_used = 0
        self.raw_limit = max(0, int(raw_limit))
        self.max_raw_models = max(0, int(max_raw_models))
        self._raw_items: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self.raw_bytes_used = 0

    def _remember_raw(self, key: tuple[str, str], data: bytes) -> None:
        previous = self._raw_items.pop(key, None)
        if previous is not None:
            self.raw_bytes_used -= len(previous)
        if not self.max_raw_models or len(data) > self.raw_limit:
            return
        while self._raw_items and (
            len(self._raw_items) >= self.max_raw_models
            or self.raw_bytes_used + len(data) > self.raw_limit
        ):
            _, evicted = self._raw_items.popitem(last=False)
            self.raw_bytes_used -= len(evicted)
        self._raw_items[key] = data
        self.raw_bytes_used += len(data)

    def clear(self) -> None:
        self._items.clear()
        self.bytes_used = 0
        self._raw_items.clear()
        self.raw_bytes_used = 0

    def get(
        self, item: BrowserAsset, catalogue: BrowserCatalogue,
        *, lod_quality: str = "full",
    ) -> preview.PreviewModel:
        key = (item.source, item.path, lod_quality)
        if key in self._items:
            self._items.move_to_end(key)
            model = self._items[key][0]
            # Returning an already-decoded LOD performs neither PBO I/O nor
            # mesh parsing. Keep those zero in the GUI's current load timings.
            if hasattr(model, "preview_cache_hit"):
                model.preview_cache_hit = True
                model.archive_cached = True
                model.read_seconds = 0.0
                model.parse_seconds = 0.0
            return model
        raw_key = (item.source, item.path)
        raw = self._raw_items.get(raw_key)
        if raw is not None:
            self._raw_items.move_to_end(raw_key)
        if raw is None:
            model = (
                load_model(item, catalogue) if lod_quality == "full"
                else load_model(item, catalogue, lod_quality=lod_quality)
            )
        else:
            model = load_model(
                item, catalogue, lod_quality=lod_quality, source_data=raw,
            )
        data = getattr(model, "source_bytes", None)
        if data is not None:
            self._remember_raw(raw_key, data)
            model.source_bytes = None  # avoid retaining P3D bytes per parsed LOD
        if hasattr(model, "preview_cache_hit"):
            model.preview_cache_hit = False
        # Include decoded arrays, approximate Python face/UV objects and PBO
        # input size so highly detailed models cannot silently grow unbounded.
        weight = (
            model.points.nbytes + model.edges.nbytes
            + 160 * len(model.faces) + item.size
        )
        if self.max_models and weight <= self.max_bytes:
            while self._items and (
                len(self._items) >= self.max_models
                or self.bytes_used + weight > self.max_bytes
            ):
                _, (_, evicted) = self._items.popitem(last=False)
                self.bytes_used -= evicted
            self._items[key] = (model, weight)
            self.bytes_used += weight
        return model


def load_texture(
    item: BrowserAsset, catalogue: BrowserCatalogue, *, mipmap: str = "auto"
) -> np.ndarray:
    if item.kind != "texture":
        raise ValueError(f"not a texture asset: {item.path}")
    record = catalogue.record_for(item)
    data = assets.read_asset_record_bytes(record)
    return decode_paa(
        data, is_paa=item.path.casefold().endswith(".paa"), mipmap=mipmap,
    )


def load_texture_with_format(
    item: BrowserAsset, catalogue: BrowserCatalogue, *, mipmap: str = "auto",
) -> tuple[np.ndarray, PaaFormatInfo | None]:
    """Load pixels and stored-format information from one archive read."""
    if item.kind != "texture":
        raise ValueError(f"not a texture asset: {item.path}")
    data = assets.read_asset_record_bytes(catalogue.record_for(item))
    image = decode_paa(
        data, is_paa=item.path.casefold().endswith(".paa"), mipmap=mipmap,
    )
    try:
        info = inspect_paa_format(data, is_paa=item.path.casefold().endswith(".paa"))
    except ValueError:
        # A damaged trailing mip chain must not discard a readable preview.
        info = None
    return image, info


class TexturePreviewCache:
    """Bounded decoded-RGBA LRU for repeatedly visited standalone textures."""

    def __init__(self, max_images: int = 12, max_bytes: int = 64 * 1024 * 1024):
        self.max_images = max(0, int(max_images))
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[
            tuple[str, str, str], np.ndarray
        ] = OrderedDict()
        self.bytes_used = 0
        self._formats: dict[tuple[str, str, str], PaaFormatInfo | None] = {}

    def clear(self) -> None:
        self._items.clear()
        self._formats.clear()
        self.bytes_used = 0

    def get(
        self, item: BrowserAsset, catalogue: BrowserCatalogue,
        *, mipmap: str = "auto", with_format: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, PaaFormatInfo | None]:
        key = (item.source, item.path, mipmap)
        if key in self._items:
            self._items.move_to_end(key)
            image = self._items[key]
            if not with_format:
                return image
            if key not in self._formats:
                try:
                    data = assets.read_asset_record_bytes(catalogue.record_for(item))
                    self._formats[key] = inspect_paa_format(
                        data, is_paa=item.path.casefold().endswith(".paa"),
                    )
                except (KeyError, OSError, ValueError):
                    # Cached preview pixels can be valid even when metadata
                    # or a catalogue record is no longer available.
                    self._formats[key] = None
            return image, self._formats[key]
        if with_format:
            image, format_info = load_texture_with_format(
                item, catalogue, mipmap=mipmap,
            )
        else:
            image = (
                load_texture(item, catalogue) if mipmap == "auto"
                else load_texture(item, catalogue, mipmap=mipmap)
            )
            format_info = None
        if self.max_images and image.nbytes <= self.max_bytes:
            while self._items and (
                len(self._items) >= self.max_images
                or self.bytes_used + image.nbytes > self.max_bytes
            ):
                evicted, old = self._items.popitem(last=False)
                self._formats.pop(evicted, None)
                self.bytes_used -= old.nbytes
            self._items[key] = image
            if with_format:
                self._formats[key] = format_info
            self.bytes_used += image.nbytes
        return (image, format_info) if with_format else image


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
