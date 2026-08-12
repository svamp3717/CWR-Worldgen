# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import io
import json
import re
import struct
from typing import Iterable, Mapping, Sequence

from .cache import CACHE_SCHEMA_VERSION, atomic_write_json, cache_key

_ENTRY_FIELDS = struct.Struct("<IIIII")
_PBO_PROPERTIES = 0x56657273  # 'Vers' in the legacy little-endian PBO header
_ASSET_SUFFIXES = {".p3d", ".paa", ".pac"}
_TEXTURE_REFERENCE = re.compile(rb"(?i)([a-z0-9_.$@/\\-]{2,240}\.(?:paa|pac))")


def canonical_asset_path(value: str) -> str:
    path = value.replace("/", "\\").strip().lstrip("\\")
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path.casefold()


@dataclass(frozen=True, slots=True)
class AssetRecord:
    path: str
    source: str
    size: int
    sha256: str | None
    dependencies: tuple[str, ...] = ()
    readable: bool = True


@dataclass(frozen=True, slots=True)
class AssetScanResult:
    roots: tuple[str, ...]
    records: tuple[AssetRecord, ...]
    selected_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    unreadable_pbos: tuple[str, ...]
    catalogue_sha256: str
    cache_hit: bool = False
    cache_path: str | None = None

    @property
    def verified(self) -> bool:
        return bool(self.roots) and not self.missing_models and not self.missing_dependencies

    def to_manifest(self) -> dict[str, object]:
        return {
            "roots": self.roots,
            "asset_count": len(self.records),
            "selected_models": self.selected_models,
            "missing_models": self.missing_models,
            "missing_dependencies": self.missing_dependencies,
            "unreadable_pbos": self.unreadable_pbos,
            "verified": self.verified,
            "catalogue_sha256": self.catalogue_sha256,
        }

    def cache_info(self) -> dict[str, object]:
        return {
            "hit": self.cache_hit,
            "path": self.cache_path,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_cstring(stream: io.BytesIO) -> str:
    value = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("truncated PBO string")
        if byte == b"\0":
            return value.decode("latin-1")
        value.extend(byte)


def _p3d_dependencies(data: bytes) -> tuple[str, ...]:
    found: set[str] = set()
    for match in _TEXTURE_REFERENCE.finditer(data):
        try:
            decoded = match.group(1).decode("ascii")
        except UnicodeDecodeError:
            continue
        found.add(canonical_asset_path(decoded))
    return tuple(sorted(found))


def _pbo_records(path: Path) -> tuple[list[AssetRecord], str | None]:
    """List uncompressed and compressed PBO assets without requiring extraction.

    Compressed entries remain useful for existence checks. Their bytes and embedded
    P3D dependencies cannot be inspected, so they are marked unreadable rather than
    being treated as absent, a distinction humans occasionally appreciate.
    """
    try:
        raw = path.read_bytes()
        stream = io.BytesIO(raw)
        metadata: list[tuple[str, int, int]] = []  # name, method, stored size
        properties: dict[str, str] = {}
        while True:
            name = _read_cstring(stream)
            fields = stream.read(_ENTRY_FIELDS.size)
            if len(fields) != _ENTRY_FIELDS.size:
                raise ValueError("truncated PBO header")
            packing, original_size, reserved, timestamp, data_size = _ENTRY_FIELDS.unpack(fields)
            if not name:
                if packing == _PBO_PROPERTIES:
                    while True:
                        key = _read_cstring(stream)
                        if not key:
                            break
                        properties[key.casefold()] = _read_cstring(stream)
                    continue
                if any((packing, original_size, reserved, timestamp, data_size)):
                    raise ValueError("unsupported PBO extension record")
                break
            metadata.append((name, packing, data_size))

        prefix = properties.get("prefix", "").replace("/", "\\").strip("\\")
        if not prefix:
            prefix = path.stem
        records: list[AssetRecord] = []
        data_cursor = stream.tell()
        for name, packing, data_size in metadata:
            data = raw[data_cursor : data_cursor + data_size]
            if len(data) != data_size:
                raise ValueError(f"truncated PBO entry {name}")
            data_cursor += data_size
            combined = name.replace("/", "\\").lstrip("\\")
            if prefix and not canonical_asset_path(combined).startswith(canonical_asset_path(prefix) + "\\"):
                combined = prefix + "\\" + combined
            canonical = canonical_asset_path(combined)
            if Path(canonical).suffix.casefold() not in _ASSET_SUFFIXES:
                continue
            readable = packing == 0
            dependencies = _p3d_dependencies(data) if readable and canonical.endswith(".p3d") else ()
            records.append(
                AssetRecord(
                    path=canonical,
                    source=str(path),
                    size=data_size,
                    sha256=_sha256_bytes(data) if readable else None,
                    dependencies=dependencies,
                    readable=readable,
                )
            )
        return records, None
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return [], f"{path}: {exc}"


def _loose_records(root: Path) -> tuple[list[AssetRecord], list[str]]:
    records: list[AssetRecord] = []
    errors: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold()):
        suffix = path.suffix.casefold()
        if suffix == ".pbo":
            scanned, error = _pbo_records(path)
            records.extend(scanned)
            if error:
                errors.append(error)
            continue
        if suffix not in _ASSET_SUFFIXES:
            continue
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        records.append(
            AssetRecord(
                path=canonical_asset_path(relative),
                source=str(path),
                size=len(data),
                sha256=_sha256_bytes(data),
                dependencies=_p3d_dependencies(data) if suffix == ".p3d" else (),
            )
        )
    return records, errors



_ASSET_CACHE_SCHEMA = 1
_CATALOGUE_MEMORY: dict[str, tuple[tuple[str, ...], tuple[AssetRecord, ...], tuple[str, ...]]] = {}


def _root_snapshot(roots: Sequence[Path]) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    root_names: list[str] = []
    entries: list[dict[str, object]] = []
    for raw_root in sorted((Path(item).resolve() for item in roots), key=lambda item: str(item).casefold()):
        if not raw_root.exists():
            raise ValueError(f"asset root does not exist: {raw_root}")
        root_names.append(str(raw_root))
        if raw_root.is_file():
            if raw_root.suffix.casefold() != ".pbo":
                raise ValueError(f"asset root file must be a PBO: {raw_root}")
            stat = raw_root.stat()
            entries.append({
                "root": str(raw_root),
                "path": raw_root.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
            continue
        for path in sorted((item for item in raw_root.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold()):
            if path.suffix.casefold() not in _ASSET_SUFFIXES | {".pbo"}:
                continue
            stat = path.stat()
            entries.append({
                "root": str(raw_root),
                "path": path.relative_to(raw_root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    return tuple(root_names), entries


def _scan_catalogue(roots: Sequence[Path]) -> tuple[tuple[str, ...], tuple[AssetRecord, ...], tuple[str, ...]]:
    records: list[AssetRecord] = []
    errors: list[str] = []
    root_names: list[str] = []
    for root in sorted((Path(item).resolve() for item in roots), key=lambda item: str(item).casefold()):
        root_names.append(str(root))
        if root.is_file():
            scanned, error = _pbo_records(root)
            records.extend(scanned)
            if error:
                errors.append(error)
        else:
            scanned, scan_errors = _loose_records(root)
            records.extend(scanned)
            errors.extend(scan_errors)

    by_path: dict[str, AssetRecord] = {}
    for record in records:
        previous = by_path.get(record.path)
        if previous is None or (record.readable and not previous.readable):
            by_path[record.path] = record
    ordered = tuple(by_path[key] for key in sorted(by_path))
    return tuple(root_names), ordered, tuple(sorted(errors))


def _load_or_scan_catalogue(
    roots: Sequence[Path],
    *,
    cache_dir: Path | None,
    use_cache: bool,
    refresh: bool,
) -> tuple[tuple[str, ...], tuple[AssetRecord, ...], tuple[str, ...], bool, str | None]:
    root_names, snapshot = _root_snapshot(roots)
    key = cache_key("asset-catalogue-v1", {"schema": _ASSET_CACHE_SCHEMA, "entries": snapshot})
    cache_path = cache_dir / "assets" / f"{key}.json" if cache_dir is not None else None

    if use_cache and not refresh and key in _CATALOGUE_MEMORY:
        names, records, errors = _CATALOGUE_MEMORY[key]
        return names, records, errors, True, str(cache_path) if cache_path else None

    if use_cache and not refresh and cache_path is not None and cache_path.is_file():
        try:
            document = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                document.get("schema") == CACHE_SCHEMA_VERSION
                and document.get("asset_schema") == _ASSET_CACHE_SCHEMA
                and document.get("root_names") == list(root_names)
                and document.get("snapshot_key") == key
            ):
                records = tuple(
                    AssetRecord(
                        path=str(item["path"]),
                        source=str(item["source"]),
                        size=int(item["size"]),
                        sha256=item.get("sha256"),
                        dependencies=tuple(str(value) for value in item.get("dependencies", ())),
                        readable=bool(item.get("readable", True)),
                    )
                    for item in document.get("records", ())
                    if isinstance(item, dict)
                )
                errors = tuple(str(value) for value in document.get("errors", ()))
                _CATALOGUE_MEMORY[key] = (root_names, records, errors)
                return root_names, records, errors, True, str(cache_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    names, records, errors = _scan_catalogue(roots)
    _CATALOGUE_MEMORY[key] = (names, records, errors)
    if use_cache and cache_path is not None:
        atomic_write_json(cache_path, {
            "schema": CACHE_SCHEMA_VERSION,
            "asset_schema": _ASSET_CACHE_SCHEMA,
            "snapshot_key": key,
            "root_names": list(names),
            "records": [asdict(record) for record in records],
            "errors": list(errors),
        })
    return names, records, errors, False, str(cache_path) if cache_path else None


def scan_assets(
    roots: Sequence[Path],
    selected_models: Iterable[str],
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> AssetScanResult:
    root_names, ordered, errors, cache_hit, cache_path = _load_or_scan_catalogue(
        roots, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh
    )

    by_path = {record.path: record for record in ordered}
    selected = tuple(sorted({canonical_asset_path(model) for model in selected_models}))
    available = set(by_path)
    missing_models = tuple(model for model in selected if model not in available) if root_names else ()

    by_basename: dict[str, set[str]] = {}
    for asset_path in available:
        basename = asset_path.rsplit("\\", 1)[-1]
        by_basename.setdefault(basename, set()).add(asset_path)

    def dependency_aliases(dependency: str) -> tuple[str, ...]:
        aliases = [dependency]
        suffix = Path(dependency).suffix.casefold()
        if suffix == ".paa":
            aliases.append(dependency[:-4] + ".pac")
        elif suffix == ".pac":
            aliases.append(dependency[:-4] + ".paa")
        return tuple(aliases)

    def dependency_is_available(model: str, dependency: str) -> bool:
        aliases = dependency_aliases(dependency)
        if any(alias in available for alias in aliases):
            return True
        if "\\" not in dependency:
            model_parts = model.split("\\")
            for alias in aliases:
                if len(model_parts) > 1:
                    same_directory = "\\".join((*model_parts[:-1], alias))
                    if same_directory in available:
                        return True
                    if model_parts[0] == "data3d" and "data\\" + alias in available:
                        return True
                matches = by_basename.get(alias, set())
                if len(matches) == 1:
                    return True
        return False

    missing_dependencies = tuple(
        sorted(
            dependency
            for model in selected
            if model in by_path
            for dependency in by_path[model].dependencies
            if not dependency_is_available(model, dependency)
        )
    )

    canonical_doc = {
        "roots": list(root_names),
        "records": [asdict(record) for record in ordered],
        "selected_models": selected,
        "missing_models": missing_models,
        "missing_dependencies": missing_dependencies,
        "unreadable_pbos": list(errors),
    }
    digest = hashlib.sha256(
        (json.dumps(canonical_doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return AssetScanResult(
        roots=root_names,
        records=ordered,
        selected_models=selected,
        missing_models=missing_models,
        missing_dependencies=missing_dependencies,
        unreadable_pbos=errors,
        catalogue_sha256=digest,
        cache_hit=cache_hit,
        cache_path=cache_path,
    )


def write_asset_catalogue(
    path: Path,
    scan: AssetScanResult,
    *,
    osm_asset_mapping: Mapping[str, object] | None = None,
) -> None:
    document = {
        "schema": 2,
        **scan.to_manifest(),
        "osm_asset_mapping": dict(osm_asset_mapping or {}),
        "assets": [asdict(record) for record in scan.records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
