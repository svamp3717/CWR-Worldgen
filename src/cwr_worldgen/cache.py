# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import pickle
import shutil
import struct
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
    if enabled and cache_path is not None and cache_path.is_file() and not refresh:
        try:
            envelope = pickle.loads(cache_path.read_bytes())
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
        atomic_write_bytes(
            cache_path,
            pickle.dumps(
                {
                    "cache_schema": CACHE_SCHEMA_VERSION,
                    "stage_schema": stage_schema,
                    "value": value,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )
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
    """Large-object variant of :func:`load_or_create_pickle`.

    The ordinary helper intentionally uses ``read_bytes``/``pickle.dumps`` for
    small cache entries. Large world datasets can be hundreds of megabytes, so
    this variant streams pickle input/output directly through files and avoids
    a second full-size bytes object during cache load/store.
    """

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
