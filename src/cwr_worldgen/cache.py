# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from hashlib import sha256
from pathlib import Path, PurePath
import json
import os
import pickle
import shutil
import struct
from collections.abc import Iterable, Mapping
from typing import Any, Callable, TypeVar

CACHE_SCHEMA_VERSION = 3
DEFAULT_CACHE_DIRNAME = ".cwr-worldgen-source-cache"
LEGACY_CACHE_DIRNAME = ".cwr-cache"
T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def cache_key(namespace: str, payload: Any) -> str:
    digest = sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()



def _hash_length(digest: Any, length: int) -> None:
    digest.update(struct.pack("<Q", max(0, int(length))))


def _hash_bytes(digest: Any, marker: bytes, data: bytes) -> None:
    digest.update(marker)
    _hash_length(digest, len(data))
    digest.update(data)


def _update_stable_hash(digest: Any, value: Any) -> None:
    """Feed one Python value into a deterministic hash without materializing it.

    This deliberately encodes type markers and lengths so heterogeneous values
    cannot collide through concatenation.  Dataclasses are walked field-by-field
    instead of calling ``asdict()``, and generic iterables are consumed lazily.
    The format is private and versioned by each caller's namespace.
    """

    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B\1" if value else b"B\0")
        return
    if isinstance(value, int):
        _hash_bytes(digest, b"I", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack("<d", value))
        return
    if isinstance(value, str):
        _hash_bytes(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        _hash_bytes(digest, b"Y", data)
        return
    if isinstance(value, PurePath):
        _hash_bytes(digest, b"P", str(value).encode("utf-8"))
        return
    if is_dataclass(value) and not isinstance(value, type):
        _hash_bytes(
            digest, b"D",
            f"{value.__class__.__module__}.{value.__class__.__qualname__}".encode("utf-8"),
        )
        dataclass_fields = fields(value)
        _hash_length(digest, len(dataclass_fields))
        for field in dataclass_fields:
            _hash_bytes(digest, b"K", field.name.encode("utf-8"))
            _update_stable_hash(digest, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        _hash_length(digest, len(value))
        # Cache payload mappings are normally tiny. Sorting only their keys keeps
        # the potentially huge *values* streaming and deterministic.
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__module__, type(item[0]).__qualname__, repr(item[0])),
        )
        for key, item in items:
            _update_stable_hash(digest, key)
            _update_stable_hash(digest, item)
        return
    if isinstance(value, tuple):
        digest.update(b"T")
        _hash_length(digest, len(value))
        for item in value:
            _update_stable_hash(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"L")
        _hash_length(digest, len(value))
        for item in value:
            _update_stable_hash(digest, item)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(b"E")
        _hash_length(digest, len(value))
        for item in sorted(value, key=lambda item: (type(item).__module__, type(item).__qualname__, repr(item))):
            _update_stable_hash(digest, item)
        return
    if isinstance(value, Iterable):
        # Generators are the important case for million-record fingerprints.
        # An explicit terminator makes the stream unambiguous without requiring
        # a preliminary pass just to count elements.
        digest.update(b"G")
        for item in value:
            _update_stable_hash(digest, item)
        digest.update(b"Z")
        return
    _hash_bytes(digest, b"R", repr(value).encode("utf-8"))


def streaming_hash(namespace: str, *values: Any) -> str:
    """Return a deterministic SHA-256 while keeping large iterables streaming."""

    digest = sha256()
    _hash_bytes(digest, b"V", namespace.encode("utf-8"))
    for value in values:
        _update_stable_hash(digest, value)
    return digest.hexdigest()


def resolve_cache_dir(source_dir: Path, requested: Path | None) -> Path:
    """Resolve the persistent cache directory with legacy-name compatibility.

    Explicit cache paths are never rewritten. For the implicit source-local
    cache, new/fresh builds use ``.cwr-worldgen-source-cache``. Existing
    ``.cwr-cache`` directories remain usable when the new directory has not
    been created yet, avoiding an expensive forced cache migration. If both
    exist, the new name wins.
    """
    if requested is not None:
        return requested.resolve()

    source_dir = source_dir.resolve()
    current = source_dir / DEFAULT_CACHE_DIRNAME
    legacy = source_dir / LEGACY_CACHE_DIRNAME
    if current.exists() or not legacy.exists():
        return current
    return legacy


def file_snapshot(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    stat = resolved.stat()
    digest = sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def float_sequence_sha256(values: Any) -> str:
    digest = sha256()
    pack = struct.Struct("<d").pack
    for value in values:
        digest.update(pack(float(value)))
    return digest.hexdigest()


def int_sequence_sha256(values: Any) -> str:
    digest = sha256()
    pack = struct.Struct("<i").pack
    for value in values:
        digest.update(pack(int(value)))
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def load_or_create_pickle(
    *,
    cache_path: Path | None,
    producer: Callable[[], T],
    enabled: bool,
    refresh: bool,
    stage_schema: int,
    validator: Callable[[T], bool] | None = None,
) -> tuple[T, bool]:
    """Load/store a pickle cache without duplicating its serialized bytes in RAM."""

    if enabled and cache_path is not None and cache_path.is_file() and not refresh:
        try:
            with cache_path.open("rb") as stream:
                envelope = pickle.load(stream)
            if (
                isinstance(envelope, dict)
                and envelope.get("cache_schema") == CACHE_SCHEMA_VERSION
                and envelope.get("stage_schema") == stage_schema
                and "value" in envelope
            ):
                value = envelope["value"]
                if validator is None or validator(value):
                    return value, True
        except (OSError, EOFError, ValueError, TypeError, AttributeError, pickle.PickleError):
            pass
    value = producer()
    if enabled and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                pickle.dump(
                    {
                        "cache_schema": CACHE_SCHEMA_VERSION,
                        "stage_schema": stage_schema,
                        "value": value,
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return value, False


def load_or_create_pickle_streaming(
    *,
    cache_path: Path | None,
    producer: Callable[[], T],
    enabled: bool,
    refresh: bool,
    stage_schema: int,
    validator: Callable[[T], bool] | None = None,
) -> tuple[T, bool]:
    """Compatibility alias; pickle caches are streaming by default now."""

    return load_or_create_pickle(
        cache_path=cache_path, producer=producer, enabled=enabled, refresh=refresh,
        stage_schema=stage_schema, validator=validator,
    )

def restore_or_create_file(
    *,
    cache_path: Path | None,
    destination: Path,
    producer: Callable[[Path], None],
    enabled: bool,
    refresh: bool,
) -> bool:
    """Materialize a deterministic file and return whether the cache was used."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if enabled and cache_path is not None and cache_path.is_file() and not refresh:
        shutil.copyfile(cache_path, destination)
        return True
    producer(destination)
    if enabled and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        shutil.copyfile(destination, temporary)
        os.replace(temporary, cache_path)
    return False


def restore_bundle(cache_dir: Path | None, destination_root: Path, names: tuple[str, ...], *, enabled: bool, refresh: bool) -> bool:
    if not enabled or cache_dir is None or refresh:
        return False
    if not all((cache_dir / name).is_file() for name in names):
        return False
    for name in names:
        destination = destination_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_dir / name, destination)
    return True


def store_bundle(cache_dir: Path | None, source_root: Path, names: tuple[str, ...], *, enabled: bool) -> None:
    if not enabled or cache_dir is None:
        return
    for name in names:
        source = source_root / name
        destination = cache_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
