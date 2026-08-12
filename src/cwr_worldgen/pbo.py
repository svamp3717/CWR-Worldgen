# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from hashlib import sha256
import io
import json
import os
import shutil
import struct
import subprocess
from typing import Iterable

from .cache import atomic_write_json, cache_key

_ENTRY_FIELDS = struct.Struct("<IIIII")
_FIXED_POSEIDON_TIMESTAMP = 946684800  # 2000-01-01 UTC, safely representable by legacy tools.
_VALID_BACKENDS = {"auto", "python", "poseidon"}


@dataclass(frozen=True, slots=True)
class PboEntry:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class PboPackResult:
    entries: tuple[PboEntry, ...]
    archive_hit: bool
    total_entries: int
    reused_blob_entries: int
    new_blob_entries: int
    archive_key: str
    requested_backend: str = "python"
    backend: str = "python"
    poseidon_tools_path: str | None = None
    fallback_reason: str | None = None


def _normalise_name(name: str) -> str:
    value = str(PurePosixPath(name.replace("\\", "/")))
    if value in {"", "."} or value.startswith("/") or value.startswith("../") or "/../" in value:
        raise ValueError(f"unsafe PBO entry path: {name!r}")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("PBO entry paths must be ASCII") from exc
    return value.replace("/", "\\")


def _scan_source(source: Path) -> tuple[list[PboEntry], list[dict[str, object]]]:
    if not source.is_dir():
        raise ValueError(f"PBO source directory does not exist: {source}")
    entries: list[PboEntry] = []
    manifest: list[dict[str, object]] = []
    for file_path in sorted((path for path in source.rglob("*") if path.is_file()), key=lambda p: p.as_posix().lower()):
        relative = _normalise_name(file_path.relative_to(source).as_posix())
        data = file_path.read_bytes()
        digest = sha256(data).hexdigest()
        entries.append(PboEntry(relative, data))
        manifest.append({"name": relative, "size": len(data), "sha256": digest})
    return entries, manifest


def _validate_backend(backend: str) -> str:
    value = backend.strip().lower()
    if value not in _VALID_BACKENDS:
        raise ValueError("PBO backend must be auto, python, or poseidon")
    return value


def _resolve_poseidon_tools(executable: Path | str | None) -> Path | None:
    candidates: list[str] = []
    if executable is not None and str(executable).strip():
        candidates.append(str(executable).strip())
    env_path = os.environ.get("CWR_POSEIDON_TOOLS", "").strip()
    if env_path:
        candidates.append(env_path)
    for name in ("PoseidonTools", "PoseidonTools.exe", "poseidontools", "poseidontools.exe"):
        located = shutil.which(name)
        if located:
            candidates.append(located)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _normalise_source_timestamps(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            os.utime(path, (_FIXED_POSEIDON_TIMESTAMP, _FIXED_POSEIDON_TIMESTAMP))


def _validate_packed_entries(output: Path, expected: Iterable[PboEntry]) -> None:
    actual_entries = read_pbo(output)
    expected_map = {_normalise_name(entry.name).casefold(): entry.data for entry in expected}
    actual_map = {_normalise_name(entry.name).casefold(): entry.data for entry in actual_entries}
    if actual_map.keys() != expected_map.keys():
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        raise RuntimeError(f"PoseidonTools PBO entry mismatch: missing={missing}, extra={extra}")
    mismatched = sorted(name for name in expected_map if expected_map[name] != actual_map[name])
    if mismatched:
        raise RuntimeError(f"PoseidonTools PBO data mismatch: {mismatched}")


def _pack_poseidon(source: Path, output: Path, executable: Path, expected: Iterable[PboEntry]) -> None:
    _normalise_source_timestamps(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [str(executable), "pbo", "pack", str(source), str(output)]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"PoseidonTools failed to pack PBO: {detail}")
    if not output.is_file():
        raise RuntimeError("PoseidonTools reported success but did not create the PBO")
    _validate_packed_entries(output, expected)


def _choose_backend(backend: str, poseidon_tools_path: Path | str | None) -> tuple[str, Path | None, str | None]:
    requested = _validate_backend(backend)
    executable = _resolve_poseidon_tools(poseidon_tools_path)
    if requested == "python":
        return "python", executable, None
    if executable is not None:
        return "poseidon", executable, None
    if requested == "poseidon":
        supplied = f": {poseidon_tools_path}" if poseidon_tools_path else ""
        raise ValueError(f"PoseidonTools executable was not found{supplied}")
    return "python", None, "PoseidonTools executable was not found"


def pack_directory(
    source: Path,
    output: Path,
    *,
    backend: str = "python",
    poseidon_tools_path: Path | str | None = None,
) -> tuple[PboEntry, ...]:
    entries, _manifest = _scan_source(source)
    requested = _validate_backend(backend)
    selected, executable, _reason = _choose_backend(requested, poseidon_tools_path)
    if selected == "poseidon":
        assert executable is not None
        try:
            _pack_poseidon(source, output, executable, entries)
        except (OSError, RuntimeError, ValueError) as exc:
            if requested != "auto":
                raise
            write_pbo(output, entries)
    else:
        write_pbo(output, entries)
    return tuple(entries)


def write_pbo(output: Path, entries: Iterable[PboEntry]) -> None:
    checked = list(entries)
    names: set[str] = set()
    for entry in checked:
        name = _normalise_name(entry.name)
        folded = name.casefold()
        if folded in names:
            raise ValueError(f"duplicate PBO entry: {name}")
        names.add(folded)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for entry in checked:
            name = _normalise_name(entry.name).encode("ascii")
            stream.write(name + b"\0")
            stream.write(_ENTRY_FIELDS.pack(0, 0, 0, 0, len(entry.data)))
        stream.write(b"\0")
        stream.write(_ENTRY_FIELDS.pack(0, 0, 0, 0, 0))
        for entry in checked:
            stream.write(entry.data)


def pack_directory_cached(
    source: Path,
    output: Path,
    *,
    cache_dir: Path | None,
    enabled: bool = True,
    refresh: bool = False,
    backend: str = "python",
    poseidon_tools_path: Path | str | None = None,
) -> PboPackResult:
    """Pack a source tree with content-addressed entry and archive reuse."""
    requested = _validate_backend(backend)
    entries, entry_manifest = _scan_source(source)
    reused = 0
    created = 0
    for entry, metadata in zip(entries, entry_manifest):
        digest = str(metadata["sha256"])
        blob = cache_dir / "pbo" / "blobs" / f"{digest}.bin" if cache_dir is not None else None
        if enabled and blob is not None and blob.is_file() and not refresh:
            reused += 1
        else:
            created += 1
            if enabled and blob is not None:
                blob.parent.mkdir(parents=True, exist_ok=True)
                temporary = blob.with_name(blob.name + ".tmp")
                temporary.write_bytes(entry.data)
                os.replace(temporary, blob)

    selected, executable, fallback_reason = _choose_backend(requested, poseidon_tools_path)

    def archive_for(selected_backend: str) -> tuple[str, Path | None]:
        tool_signature: dict[str, int] | None = None
        if selected_backend == "poseidon" and executable is not None:
            stat = executable.stat()
            tool_signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        key = cache_key(
            "pbo-archive-v3",
            {
                "backend": selected_backend,
                "fixed_timestamp": _FIXED_POSEIDON_TIMESTAMP,
                "tool_signature": tool_signature,
                "entries": entry_manifest,
            },
        )
        path = cache_dir / "pbo" / "archives" / f"{key}.pbo" if cache_dir is not None else None
        return key, path

    archive_key, archive = archive_for(selected)
    archive_hit = bool(enabled and archive is not None and archive.is_file() and not refresh)
    output.parent.mkdir(parents=True, exist_ok=True)
    if archive_hit:
        shutil.copyfile(archive, output)
    else:
        try:
            if selected == "poseidon":
                assert executable is not None
                _pack_poseidon(source, output, executable, entries)
            else:
                write_pbo(output, entries)
        except (OSError, RuntimeError, ValueError) as exc:
            if requested != "auto" or selected != "poseidon":
                raise
            fallback_reason = str(exc)
            selected = "python"
            executable = None
            archive_key, archive = archive_for(selected)
            archive_hit = bool(enabled and archive is not None and archive.is_file() and not refresh)
            if archive_hit:
                shutil.copyfile(archive, output)
            else:
                write_pbo(output, entries)
        if not archive_hit and enabled and archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_name(archive.name + ".tmp")
            shutil.copyfile(output, temporary)
            os.replace(temporary, archive)

    if enabled and cache_dir is not None:
        atomic_write_json(
            cache_dir / "pbo" / "manifests" / f"{archive_key}.json",
            {
                "schema": 3,
                "archive_key": archive_key,
                "requested_backend": requested,
                "backend": selected,
                "poseidon_tools_path": str(executable) if executable else None,
                "fallback_reason": fallback_reason,
                "entries": entry_manifest,
            },
        )
    return PboPackResult(
        tuple(entries),
        archive_hit,
        len(entries),
        reused,
        created,
        archive_key,
        requested,
        selected,
        str(executable) if executable else None,
        fallback_reason,
    )


def read_pbo(path: Path) -> tuple[PboEntry, ...]:
    stream = io.BytesIO(path.read_bytes())
    metadata: list[tuple[str, int]] = []
    while True:
        name_bytes = bytearray()
        while True:
            value = stream.read(1)
            if not value:
                raise ValueError("truncated PBO header")
            if value == b"\0":
                break
            name_bytes.extend(value)
        fields = stream.read(_ENTRY_FIELDS.size)
        if len(fields) != _ENTRY_FIELDS.size:
            raise ValueError("truncated PBO entry fields")
        packing, original_size, reserved, timestamp, data_size = _ENTRY_FIELDS.unpack(fields)
        if not name_bytes:
            if any((packing, original_size, reserved, timestamp, data_size)):
                raise ValueError("unsupported PBO properties entry")
            break
        if packing != 0:
            raise ValueError("compressed PBO entries are not supported by this reader")
        metadata.append((name_bytes.decode("ascii"), data_size))

    entries: list[PboEntry] = []
    for name, size in metadata:
        data = stream.read(size)
        if len(data) != size:
            raise ValueError(f"truncated PBO data for {name}")
        entries.append(PboEntry(name, data))
    return tuple(entries)
