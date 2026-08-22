from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from cwr_worldgen.output_ownership import (
    OWNERSHIP_FILENAME,
    prepare_output_directory,
    record_build_ownership,
)


def _manifest(path: Path, world: str, outputs: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "generator": "cwr-worldgen test",
                "world": {"name": world},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )


def test_clean_removes_everything_in_existing_build_folder() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "build"
        world = "safe_world"
        generated = root / "source" / world / "config.cpp"
        generated.parent.mkdir(parents=True)
        generated.write_text("generated", encoding="utf-8")
        unrelated = root / "notes.txt"
        unrelated.write_text("delete me too", encoding="utf-8")
        manifest = root / "manifest.json"
        _manifest(manifest, world, {"source/config.cpp": "deadbeef"})
        record_build_ownership(root, world, manifest, merge=False)

        prepare_output_directory(root, world, clean=True)

        assert root.is_dir()
        assert list(root.iterdir()) == []


def test_clean_allows_unowned_file_inside_generated_world_namespace() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "build"
        world = "safe_world"
        user_file = root / "source" / world / "custom-user-file.txt"
        user_file.parent.mkdir(parents=True)
        user_file.write_text("mine", encoding="utf-8")

        prepare_output_directory(root, world, clean=True)

        assert root.is_dir()
        assert not user_file.exists()


def test_clean_allows_unowned_fixed_build_artifact() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "build"
        world = "safe_world"
        root.mkdir()
        preview = root / "preview.png"
        preview.write_bytes(b"not-worldgen")

        prepare_output_directory(root, world, clean=True)

        assert root.is_dir()
        assert not preview.exists()


def test_incremental_build_allows_owned_and_unowned_collisions() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        world = "safe_world"
        generated = root / "source" / world / "config.cpp"
        generated.parent.mkdir(parents=True)
        generated.write_text("generated", encoding="utf-8")
        manifest = root / "manifest.json"
        _manifest(manifest, world, {"source/config.cpp": "deadbeef"})
        record_build_ownership(root, world, manifest, merge=False)

        # Previously generated files are safe overwrite targets when clean=False.
        prepare_output_directory(root, world, clean=False)

        user_file = root / "source" / world / "user.ini"
        user_file.write_text("mine", encoding="utf-8")
        prepare_output_directory(root, world, clean=False)
        assert user_file.read_text(encoding="utf-8") == "mine"
