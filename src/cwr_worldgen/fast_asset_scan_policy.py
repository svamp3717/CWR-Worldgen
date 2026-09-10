# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast selected-asset validation for normal CWA/game-folder builds.

The historical scanner builds a complete recursive catalogue before it checks its
cache. Pointing it at a game installation therefore walks every directory and
stats every P3D/PAA/PAC/PBO, then a cold scan walks the tree again and reads every
relevant asset. Milestone builds only need the explicitly selected models/textures
and the texture dependencies referenced by those models.

This policy resolves those assets directly. It opens only the likely package for
an asset prefix, caches each PBO header index by path/size/mtime, and reads bytes
only for selected entries. The second validation pass in the same build reuses the
in-memory indexes; later builds reuse the persistent index cache. If a nonstandard
layout cannot be resolved safely, the original exhaustive scanner remains the
correctness fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import io
import json
import os
from typing import Iterable, Sequence

from . import assets as _assets
from .cache import CACHE_SCHEMA_VERSION, atomic_write_json, cache_key


FAST_ASSET_INDEX_DIRNAME = ".cwr-worldgen-asset-index-cache"
_PBO_INDEX_SCHEMA = 1
_INSTALLED = False
_FULL_SCAN = _assets.scan_assets
_PBO_INDEX_MEMORY: dict[tuple[str, int, int], "_PboIndex"] = {}


@dataclass(frozen=True, slots=True)
class _PboEntry:
    canonical_path: str
    packing: int
    original_size: int
    data_size: int
    data_offset: int


@dataclass(frozen=True, slots=True)
class _PboIndex:
    path: str
    prefix: str
    size: int
    mtime_ns: int
    entries: tuple[_PboEntry, ...]

    def by_path(self) -> dict[str, _PboEntry]:
        return {entry.canonical_path: entry for entry in self.entries}


@dataclass(slots=True)
class _LookupStats:
    index_hits: int = 0
    index_misses: int = 0


def _read_cstring(stream) -> str:
    value = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("truncated PBO string")
        if byte == b"\0":
            return value.decode("latin-1")
        value.extend(byte)


def _resolved_roots(roots: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in roots:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"asset root does not exist: {root}")
        key = os.path.normcase(str(root))
        if key not in seen:
            seen.add(key)
            result.append(root)
    return tuple(result)


def _persistent_index_root(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    resolved = Path(cache_dir).expanduser().resolve()
    # Milestone 9 routes core caches to BUILD/.cwr-worldgen-build-cache/<rev>.
    # GUI cleanup deletes that tree, so keep the expensive game/PBO index beside
    # it where normal post-build cleanup deliberately does not reach.
    if resolved.parent.name == ".cwr-worldgen-build-cache":
        return resolved.parent.parent / FAST_ASSET_INDEX_DIRNAME
    return resolved / "asset-indexes"


def _parse_pbo_index(path: Path) -> _PboIndex:
    stat = path.stat()
    metadata: list[tuple[str, int, int, int]] = []
    properties: dict[str, str] = {}
    with path.open("rb") as stream:
        while True:
            name = _read_cstring(stream)
            fields = stream.read(_assets._ENTRY_FIELDS.size)
            if len(fields) != _assets._ENTRY_FIELDS.size:
                raise ValueError("truncated PBO header")
            packing, original_size, reserved, _timestamp, data_size = _assets._ENTRY_FIELDS.unpack(fields)
            if not name:
                if packing == _assets._PBO_PROPERTIES:
                    while True:
                        key = _read_cstring(stream)
                        if not key:
                            break
                        properties[key.casefold()] = _read_cstring(stream)
                    continue
                if any((packing, original_size, reserved, data_size)):
                    # Resistance archives can contain extension markers. Empty
                    # filename remains the reliable end-of-header fence.
                    pass
                break
            metadata.append((name, packing, original_size, data_size))
        data_cursor = stream.tell()

    prefix = properties.get("prefix", "").replace("/", "\\").strip("\\")
    if not prefix:
        prefix = path.stem
    entries: list[_PboEntry] = []
    for name, packing, original_size, data_size in metadata:
        combined = name.replace("/", "\\").lstrip("\\")
        prefix_key = _assets.canonical_asset_path(prefix)
        if prefix and not _assets.canonical_asset_path(combined).startswith(prefix_key + "\\"):
            combined = prefix + "\\" + combined
        entries.append(_PboEntry(
            canonical_path=_assets.canonical_asset_path(combined),
            packing=int(packing),
            original_size=int(original_size),
            data_size=int(data_size),
            data_offset=int(data_cursor),
        ))
        data_cursor += int(data_size)
    return _PboIndex(str(path), prefix, int(stat.st_size), int(stat.st_mtime_ns), tuple(entries))


def _index_cache_path(index_root: Path | None, path: Path, stat) -> Path | None:
    if index_root is None:
        return None
    key = cache_key("cwa-pbo-header-index-v1", {
        "schema": _PBO_INDEX_SCHEMA,
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    })
    return index_root / "pbo" / f"{key}.json"


def _load_pbo_index(
    path: Path,
    *,
    index_root: Path | None,
    use_cache: bool,
    refresh: bool,
    stats: _LookupStats,
) -> _PboIndex:
    resolved = path.resolve()
    stat = resolved.stat()
    memory_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    if use_cache and not refresh:
        cached = _PBO_INDEX_MEMORY.get(memory_key)
        if cached is not None:
            stats.index_hits += 1
            return cached

    cache_path = _index_cache_path(index_root, resolved, stat)
    if use_cache and not refresh and cache_path is not None and cache_path.is_file():
        try:
            document = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                document.get("cache_schema") == CACHE_SCHEMA_VERSION
                and document.get("index_schema") == _PBO_INDEX_SCHEMA
                and document.get("path") == str(resolved)
                and int(document.get("size", -1)) == int(stat.st_size)
                and int(document.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            ):
                index = _PboIndex(
                    path=str(resolved),
                    prefix=str(document.get("prefix", resolved.stem)),
                    size=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                    entries=tuple(
                        _PboEntry(
                            canonical_path=str(item[0]),
                            packing=int(item[1]),
                            original_size=int(item[2]),
                            data_size=int(item[3]),
                            data_offset=int(item[4]),
                        )
                        for item in document.get("entries", ())
                        if isinstance(item, list) and len(item) == 5
                    ),
                )
                _PBO_INDEX_MEMORY[memory_key] = index
                stats.index_hits += 1
                return index
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    index = _parse_pbo_index(resolved)
    _PBO_INDEX_MEMORY[memory_key] = index
    stats.index_misses += 1
    if use_cache and cache_path is not None:
        atomic_write_json(cache_path, {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "index_schema": _PBO_INDEX_SCHEMA,
            "path": index.path,
            "prefix": index.prefix,
            "size": index.size,
            "mtime_ns": index.mtime_ns,
            "entries": [
                [entry.canonical_path, entry.packing, entry.original_size, entry.data_size, entry.data_offset]
                for entry in index.entries
            ],
        })
    return index


def _casefold_child(parent: Path, name: str) -> Path | None:
    direct = parent / name
    if direct.exists():
        return direct
    if not parent.is_dir():
        return None
    folded = name.casefold()
    try:
        for child in parent.iterdir():
            if child.name.casefold() == folded:
                return child
    except OSError:
        return None
    return None


def _casefold_relative(root: Path, relative: str) -> Path | None:
    current = root
    for part in relative.replace("/", "\\").split("\\"):
        if not part:
            continue
        current = _casefold_child(current, part)
        if current is None:
            return None
    return current if current.is_file() else None


def _relative_candidate(root: Path, parts: Sequence[str]) -> Path | None:
    current = root
    for part in parts:
        current = _casefold_child(current, part)
        if current is None:
            return None
    return current if current.is_file() else None


def _likely_pbos(root: Path, prefix: str) -> tuple[Path, ...]:
    if root.is_file():
        return (root,) if root.suffix.casefold() == ".pbo" else ()
    filename = f"{prefix}.pbo"
    layouts = (
        (filename,),
        ("Dta", filename),
        ("Res", "Dta", filename),
        ("AddOns", filename),
        ("Res", "AddOns", filename),
    )
    result: list[Path] = []
    seen: set[str] = set()
    for parts in layouts:
        candidate = _relative_candidate(root, parts)
        if candidate is None:
            continue
        key = os.path.normcase(str(candidate.resolve()))
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _fallback_named_pbos(root: Path, prefix: str) -> tuple[Path, ...]:
    """Rare compatibility path for unusual mod layouts; never used for stock paths."""
    if not root.is_dir():
        return ()
    wanted = f"{prefix}.pbo".casefold()
    matches: list[Path] = []
    try:
        for directory, dirnames, filenames in os.walk(root):
            # Do not recurse into generated/cache trees that cannot be game assets.
            dirnames[:] = [
                name for name in dirnames
                if not name.casefold().startswith(".cwr-worldgen-")
                and name.casefold() not in {"source-data", "builds", "output", "outputs"}
            ]
            for name in filenames:
                if name.casefold() == wanted:
                    matches.append(Path(directory) / name)
    except OSError:
        return ()
    return tuple(matches)


def _read_indexed_entry(path: Path, entry: _PboEntry) -> bytes | None:
    if entry.packing != 0:
        return None
    with path.open("rb") as stream:
        stream.seek(entry.data_offset)
        data = stream.read(entry.data_size)
    if len(data) != entry.data_size:
        raise ValueError(f"truncated PBO entry {entry.canonical_path}")
    return data


def _record_from_loose(path: Path, canonical_path: str) -> _assets.AssetRecord:
    data = path.read_bytes()
    suffix = Path(canonical_path).suffix.casefold()
    return _assets.AssetRecord(
        path=canonical_path,
        source=str(path),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        dependencies=_assets._p3d_dependencies(data) if suffix == ".p3d" else (),
        readable=True,
    )


def _record_from_pbo(path: Path, entry: _PboEntry) -> _assets.AssetRecord:
    data = _read_indexed_entry(path, entry)
    readable = data is not None
    return _assets.AssetRecord(
        path=entry.canonical_path,
        source=str(path),
        size=entry.data_size,
        sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
        dependencies=(
            _assets._p3d_dependencies(data)
            if data is not None and entry.canonical_path.endswith(".p3d") else ()
        ),
        readable=readable,
    )


def _locate(
    roots: Sequence[Path],
    canonical_path: str,
    *,
    index_root: Path | None,
    use_cache: bool,
    refresh: bool,
    stats: _LookupStats,
) -> _assets.AssetRecord | None:
    prefix = canonical_path.split("\\", 1)[0]
    for root in roots:
        if root.is_dir():
            loose = _casefold_relative(root, canonical_path)
            if loose is None and root.name.casefold() == prefix.casefold() and "\\" in canonical_path:
                loose = _casefold_relative(root, canonical_path.split("\\", 1)[1])
            if loose is not None:
                try:
                    return _record_from_loose(loose, canonical_path)
                except OSError:
                    pass

        direct_pbos = _likely_pbos(root, prefix)
        checked: set[str] = set()
        for pbo in direct_pbos:
            checked.add(os.path.normcase(str(pbo.resolve())))
            try:
                index = _load_pbo_index(
                    pbo, index_root=index_root, use_cache=use_cache, refresh=refresh, stats=stats
                )
                entry = index.by_path().get(canonical_path)
                if entry is not None:
                    return _record_from_pbo(Path(index.path), entry)
            except (OSError, ValueError, UnicodeDecodeError):
                continue

        # Non-standard mods occasionally bury packages one level deeper. Search
        # only after all zero-recursion stock locations failed.
        for pbo in _fallback_named_pbos(root, prefix):
            key = os.path.normcase(str(pbo.resolve()))
            if key in checked:
                continue
            try:
                index = _load_pbo_index(
                    pbo, index_root=index_root, use_cache=use_cache, refresh=refresh, stats=stats
                )
                entry = index.by_path().get(canonical_path)
                if entry is not None:
                    return _record_from_pbo(Path(index.path), entry)
            except (OSError, ValueError, UnicodeDecodeError):
                continue
    return None


def _dependency_candidates(model: str, dependency: str) -> tuple[str, ...]:
    suffix = Path(dependency).suffix.casefold()
    aliases = [dependency]
    if suffix == ".paa":
        aliases.append(dependency[:-4] + ".pac")
    elif suffix == ".pac":
        aliases.append(dependency[:-4] + ".paa")

    candidates: list[str] = []
    if "\\" in dependency:
        candidates.extend(aliases)
    else:
        model_parts = model.split("\\")
        for alias in aliases:
            if len(model_parts) > 1:
                candidates.append("\\".join((*model_parts[:-1], alias)))
            if model_parts and model_parts[0] == "data3d":
                candidates.append("data\\" + alias)
            candidates.append(alias)
    return tuple(dict.fromkeys(_assets.canonical_asset_path(value) for value in candidates))


def _targeted_scan(
    roots: Sequence[Path],
    selected_assets: Sequence[str],
    *,
    cache_dir: Path | None,
    use_cache: bool,
    refresh: bool,
) -> tuple[_assets.AssetScanResult, _LookupStats]:
    root_paths = _resolved_roots(roots)
    root_names = tuple(str(path) for path in root_paths)
    selected = tuple(sorted({_assets.canonical_asset_path(value) for value in selected_assets}))
    index_root = _persistent_index_root(cache_dir)
    stats = _LookupStats()
    records: dict[str, _assets.AssetRecord] = {}
    missing_selected: list[str] = []

    for asset_path in selected:
        record = _locate(
            root_paths, asset_path, index_root=index_root,
            use_cache=use_cache, refresh=refresh, stats=stats,
        )
        if record is None:
            missing_selected.append(asset_path)
        else:
            records[record.path] = record

    missing_dependencies: set[str] = set()
    for model in selected:
        record = records.get(model)
        if record is None or not model.endswith(".p3d"):
            continue
        for dependency in record.dependencies:
            resolved = None
            for candidate in _dependency_candidates(model, dependency):
                existing = records.get(candidate)
                if existing is not None:
                    resolved = existing
                    break
                candidate_record = _locate(
                    root_paths, candidate, index_root=index_root,
                    use_cache=use_cache, refresh=refresh, stats=stats,
                )
                if candidate_record is not None:
                    records[candidate_record.path] = candidate_record
                    resolved = candidate_record
                    break
            if resolved is None:
                missing_dependencies.add(dependency)

    ordered = tuple(records[key] for key in sorted(records))
    canonical_doc = {
        "mode": "targeted",
        "roots": list(root_names),
        "records": [asdict(record) for record in ordered],
        "selected_models": selected,
        "missing_models": sorted(missing_selected),
        "missing_dependencies": sorted(missing_dependencies),
        "unreadable_pbos": [],
    }
    digest = hashlib.sha256(
        (json.dumps(canonical_doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    result = _assets.AssetScanResult(
        roots=root_names,
        records=ordered,
        selected_models=selected,
        missing_models=tuple(sorted(missing_selected)) if root_names else (),
        missing_dependencies=tuple(sorted(missing_dependencies)),
        unreadable_pbos=(),
        catalogue_sha256=digest,
        cache_hit=bool(stats.index_hits) and stats.index_misses == 0,
        cache_path=str(index_root) if index_root is not None else None,
    )
    return result, stats


def scan_assets_fast(
    roots: Sequence[Path],
    selected_models: Iterable[str],
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> _assets.AssetScanResult:
    """Validate selected assets without recursively cataloguing the game folder."""
    selected = tuple(str(value) for value in selected_models)
    if not roots:
        return _FULL_SCAN(
            roots, selected, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh
        )
    result, _stats = _targeted_scan(
        roots, selected, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh
    )
    if not result.missing_models and not result.missing_dependencies:
        return result
    # Preserve the old scanner's basename ambiguity handling and arbitrary custom
    # layouts for the rare cases the direct package resolver cannot prove.
    return _FULL_SCAN(
        roots, selected, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh
    )


def install_fast_asset_scan_policy() -> None:
    """Route generator validation through the selected-asset scanner."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import generator

    generator.scan_assets = scan_assets_fast
    _INSTALLED = True
