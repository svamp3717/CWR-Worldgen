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


def test_clean_removes_only_recorded_worldgen_files() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        world = "safe_world"
        generated = root / "source" / world / "config.cpp"
        generated.parent.mkdir(parents=True)
        generated.write_text("generated", encoding="utf-8")
        unrelated = root / "notes.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        manifest = root / "manifest.json"
        _manifest(manifest, world, {"source/config.cpp": "deadbeef"})
        record_build_ownership(root, world, manifest, merge=False)

        prepare_output_directory(root, world, clean=True)

        assert not generated.exists()
        assert not manifest.exists()
        assert not (root / OWNERSHIP_FILENAME).exists()
        assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_clean_refuses_unowned_file_inside_generated_world_namespace() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        world = "safe_world"
        user_file = root / "source" / world / "custom-user-file.txt"
        user_file.parent.mkdir(parents=True)
        user_file.write_text("mine", encoding="utf-8")

        with pytest.raises(FileExistsError, match="not owned by cwr-worldgen"):
            prepare_output_directory(root, world, clean=True)

        assert user_file.read_text(encoding="utf-8") == "mine"


def test_clean_refuses_unowned_fixed_build_artifact() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        world = "safe_world"
        preview = root / "preview.png"
        preview.write_bytes(b"not-worldgen")

        with pytest.raises(FileExistsError, match="preview.png"):
            prepare_output_directory(root, world, clean=True)

        assert preview.read_bytes() == b"not-worldgen"


def test_incremental_build_may_replace_owned_but_not_unowned_files() -> None:
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
        with pytest.raises(FileExistsError, match="user.ini"):
            prepare_output_directory(root, world, clean=False)
        assert user_file.read_text(encoding="utf-8") == "mine"
