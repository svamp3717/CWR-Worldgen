# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable

from ._version import GENERATOR_VERSION

OWNERSHIP_FILENAME = ".cwr-worldgen-owned.json"
_OWNERSHIP_SCHEMA = 1

# These are stable root-level build artifacts. If one already exists but was not
# recorded as belonging to cwr-worldgen, the builder must not silently replace
# it. Failing is less exciting than deleting somebody's unrelated preview.png.
_RESERVED_ROOT_FILES = frozenset({
    OWNERSHIP_FILENAME,
    "manifest.json",
    "validation-report.txt",
    "cache-report.json",
    "preview.png",
    "height-preview.png",
    "material-preview.png",
    "osm-geography-preview.png",
    "meadow-grass-placement.png",
    "building-source-reference.png",
    "overview-map.png",
    "osm-source.json",
    "overpass-query.txt",
    "OSM-ATTRIBUTION.txt",
    "asset-catalogue.json",
    "road-fit-report.json",
    "terrain-grading-report.json",
    "terrain-solved-meters.tif",
    "reproducibility-report.json",
    "surface-pass-report.json",
    "building-asset-catalogue.json",
    "semantic-site-catalogue.json",
    "forest-cluster-catalogue.json",
    "infrastructure-asset-catalogue.json",
})


def _relative_under(root: Path, path: Path) -> str | None:
    # Keep the path lexical so a symlink *inside* the build root is still
    # identified as that build-root entry rather than as its external target.
    try:
        relative = path.absolute().relative_to(root.absolute())
    except (OSError, ValueError):
        return None
    if not relative.parts:
        return None
    return relative.as_posix()


def _safe_relative(value: str) -> PurePosixPath | None:
    normalized = str(value).replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or not candidate.parts:
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate


def _manifest_output_path(root: Path, world_name: str, key: str) -> Path | None:
    relative = _safe_relative(key)
    if relative is None:
        return None
    parts = relative.parts
    # Build manifests intentionally describe files inside the packed island
    # source tree as "source/<path>" while the on-disk build tree is
    # "source/<world>/<path>". Preserve that long-standing manifest format and
    # translate it only for ownership cleanup.
    if parts[0].casefold() == "source" and len(parts) > 1:
        return root / "source" / world_name / Path(*parts[1:])
    return root / Path(*parts)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _legacy_owned_files(root: Path, world_name: str) -> set[str]:
    """Best-effort migration from builds made before ownership manifests.

    Only exact file paths already declared by a cwr-worldgen manifest are
    considered owned. Unknown files are deliberately left alone, even inside
    directories normally used by the generator.
    """

    manifest_path = root / "manifest.json"
    document = _load_json(manifest_path)
    if document is None:
        return set()
    generator = str(document.get("generator", ""))
    world = document.get("world")
    manifest_world = str(world.get("name", "")) if isinstance(world, dict) else ""
    if not generator.casefold().startswith("cwr-worldgen"):
        return set()
    if manifest_world and manifest_world.casefold() != world_name.casefold():
        return set()

    owned: set[str] = set()
    outputs = document.get("outputs")
    if isinstance(outputs, dict):
        for key in outputs:
            target = _manifest_output_path(root, world_name, str(key))
            if target is not None:
                relative = _relative_under(root, target)
                if relative:
                    owned.add(relative)
    # These are generator-owned bookkeeping files but historically were not
    # included in manifest["outputs"].
    for filename in ("manifest.json", "validation-report.txt", "cache-report.json"):
        if (root / filename).is_file():
            owned.add(filename)
    return owned


def _ownership_document(root: Path, world_name: str) -> tuple[dict[str, object] | None, set[str]]:
    ownership_path = root / OWNERSHIP_FILENAME
    document = _load_json(ownership_path)
    if document is None:
        return None, _legacy_owned_files(root, world_name)
    if int(document.get("schema", 0) or 0) != _OWNERSHIP_SCHEMA:
        return None, set()
    recorded_world = str(document.get("world", ""))
    if recorded_world and recorded_world.casefold() != world_name.casefold():
        return None, set()
    files = document.get("files")
    if not isinstance(files, list):
        return None, set()
    owned: set[str] = set()
    for value in files:
        relative = _safe_relative(str(value))
        if relative is not None:
            owned.add(relative.as_posix())
    # A valid ownership manifest is itself generated state.
    owned.add(OWNERSHIP_FILENAME)
    return document, owned


def _load_owned_files(root: Path, world_name: str) -> set[str]:
    _document, owned = _ownership_document(root, world_name)
    return owned


def _is_reserved_generated_path(relative: PurePosixPath, world_name: str) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if len(parts) == 1 and parts[0] in _RESERVED_ROOT_FILES:
        return True
    folded = tuple(part.casefold() for part in parts)
    world = world_name.casefold()
    # Dynamic generated assets live below this namespace, so any surviving
    # unowned file there could be overwritten later when model/texture names are
    # computed. Refuse the build instead of guessing.
    if len(parts) >= 2 and folded[0] == "source" and folded[1] == world:
        return True
    if len(parts) >= 2 and folded[0] == "missions" and folded[1] == f"test_mission.{world}":
        return True
    # Milestone/mod directory names may vary, but the generated island archive
    # and intro mission always include the world name.
    if len(parts) >= 3 and folded[-2] == "addons" and folded[-1] == f"{world}.pbo":
        return True
    if len(parts) >= 3 and folded[-2] == "anims" and folded[-1] == f"intro.{world}":
        return True
    if any(part == f"intro.{world}" for part in folded):
        return True
    return False


def _unowned_reserved_conflicts(root: Path, world_name: str, owned: set[str]) -> list[str]:
    conflicts: list[str] = []
    if not root.exists():
        return conflicts
    for path in root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative_text = _relative_under(root, path)
        if not relative_text or relative_text in owned:
            continue
        relative = _safe_relative(relative_text)
        if relative is not None and _is_reserved_generated_path(relative, world_name):
            conflicts.append(relative.as_posix())
    return sorted(conflicts)


def prepare_output_directory(root: Path, world_name: str, *, clean: bool) -> None:
    """Prepare a build root without touching files the generator does not own.

    When ``clean`` is true, only exact paths recorded by the previous
    cwr-worldgen ownership/legacy manifest are removed. Unknown files are never
    deleted. If an unknown file occupies a namespace this build may need to
    write, fail before generation rather than overwrite it.
    """

    root = root.resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return

    _ownership_doc, owned = _ownership_document(root, world_name)
    if clean:
        removable_parents: set[Path] = set()
        for relative_text in sorted(owned, key=lambda value: value.count("/"), reverse=True):
            relative = _safe_relative(relative_text)
            if relative is None:
                continue
            target = root / Path(*relative.parts)
            try:
                if target.is_symlink():
                    # Unlink the directory entry itself; never follow its target.
                    target.unlink()
                    removable_parents.update(target.parents)
                    continue
                # If a parent component was replaced by a symlink, do not follow
                # it outside the build root while cleaning an old ownership list.
                target.parent.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            try:
                if target.is_file():
                    target.unlink()
                    removable_parents.update(target.parents)
            except OSError:
                # Preserve the existing build if a generated file is locked. The
                # subsequent writer will surface the actionable error when it tries
                # to replace that exact output.
                continue

        for directory in sorted(
            (path for path in removable_parents if path != root),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                # Non-empty means it contains something we do not own. Leave it.
                pass

        # Every formerly owned file has now either been deleted or was locked.
        # Anything that remains in a generated namespace is therefore unsafe to
        # overwrite unless it is one of those still-owned locked files.
        surviving_owned = {
            rel for rel in owned
            if (root / Path(*PurePosixPath(rel).parts)).exists()
        }
        conflicts = _unowned_reserved_conflicts(root, world_name, surviving_owned)
    else:
        # Incremental builds may replace previously recorded generated files,
        # but still must not overwrite unrelated files.
        conflicts = _unowned_reserved_conflicts(root, world_name, owned)

    if conflicts:
        preview = "\n  - ".join(conflicts[:12])
        suffix = "\n  - ..." if len(conflicts) > 12 else ""
        raise FileExistsError(
            "refusing to overwrite files not owned by cwr-worldgen:\n  - "
            + preview
            + suffix
            + "\nMove those files elsewhere, choose another build directory, or delete them explicitly."
        )

    root.mkdir(parents=True, exist_ok=True)


def record_build_ownership(
    root: Path,
    world_name: str,
    manifest_path: Path,
    *,
    extra_files: Iterable[Path] = (),
    merge: bool = True,
) -> Path:
    """Record exact generated files for safe cleanup on the next build."""

    root = root.resolve()
    owned = _load_owned_files(root, world_name) if merge else set()
    document = _load_json(manifest_path)
    if document is not None:
        outputs = document.get("outputs")
        if isinstance(outputs, dict):
            for key in outputs:
                target = _manifest_output_path(root, world_name, str(key))
                if target is not None and (target.is_file() or target.is_symlink()):
                    relative = _relative_under(root, target)
                    if relative:
                        owned.add(relative)

    standard_files = (
        manifest_path,
        root / "validation-report.txt",
        root / "cache-report.json",
    )
    for path in (*standard_files, *tuple(extra_files)):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = _relative_under(root, path)
        if relative:
            owned.add(relative)

    ownership_path = root / OWNERSHIP_FILENAME
    payload = {
        "schema": _OWNERSHIP_SCHEMA,
        "generator": GENERATOR_VERSION,
        "world": world_name,
        "files": sorted(value for value in owned if value != OWNERSHIP_FILENAME),
    }
    ownership_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ownership_path
