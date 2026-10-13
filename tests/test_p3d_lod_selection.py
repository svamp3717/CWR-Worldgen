"""Per-model LOD listing and visual preview selection across P3D generations."""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import p3d_lod_options as lods
import p3d_preview_geometry as preview
import p3d_asset_browser_app as browser
import p3d_categoriser_app as categoriser


def _legacy_odol(
    version: int,
    specs: tuple[tuple[float, int], ...],
    *,
    proxy: bool = False,
) -> bytes:
    def lod(point_count: int, texture: str) -> bytes:
        assert point_count >= 3
        out = bytearray()
        out.extend(struct.pack("<I", point_count))
        out.extend(b"\0" * (point_count * 4))
        out.extend(struct.pack("<I", point_count))
        for index in range(point_count):
            out.extend(struct.pack("<ff", index / point_count, 0.5))
        out.extend(struct.pack("<I", point_count))
        for index in range(point_count):
            out.extend(struct.pack("<fff", float(index), float(index % 3), 0.0))
        out.extend(struct.pack("<I", point_count))
        out.extend(struct.pack("<fff", 0.0, 0.0, 1.0) * point_count)
        out.extend(b"\0" * 48)
        out.extend(struct.pack("<I", 1))
        out.extend(texture.encode("ascii") + b"\0")
        out.extend(struct.pack("<II", 0, 0))  # two edge arrays
        out.extend(struct.pack("<II", 1, 0))  # one face and section offset
        out.extend(struct.pack("<IhBHHH", 0, 0, 3, 0, 1, 2))
        out.extend(struct.pack("<IIII", 0, 0, 0, 0))  # sections, selections, properties, frames
        out.extend(b"\0" * 12)
        out.extend(struct.pack("<I", 1 if proxy else 0))
        if proxy:
            out.extend(b"proxy\\wheel\0" + b"\0" * 56)
        return bytes(out)

    out = bytearray(b"ODOL" + struct.pack("<II", version, len(specs)))
    for index, (_resolution, points) in enumerate(specs):
        out.extend(lod(points, f"data\\detail{index}.paa"))
    out.extend(struct.pack("<" + "f" * len(specs), *(value for value, _ in specs)))
    return bytes(out)


def _mlod(specs: tuple[tuple[float, int], ...]) -> bytes:
    out = bytearray(struct.pack("<4sBBHI", b"MLOD", 1, 1, 0, len(specs)))
    for number, (resolution, point_count) in enumerate(specs):
        out.extend(struct.pack("<4siiiiii", b"SP3X", 28, 1, point_count, 0, 1, 0))
        for i in range(point_count):
            out.extend(struct.pack("<fffi", float(i), float(i % 3), 0.0, 0))
        verts = (
            (0, 0, 0.0, 0.0),
            (1, 0, 1.0, 0.0),
            (2, 0, 0.0, 1.0),
            (0, 0, 0.0, 0.0),
        )
        face = struct.Struct("<32si" + "iiff" * 4 + "i")
        out.extend(face.pack(
            f"data\\lod{number}.paa".encode("ascii").ljust(32, b"\0"),
            3, *(value for vertex in verts for value in vertex), 0,
        ))
        out.extend(b"TAGG")
        out.extend(struct.pack("<64si", b"#EndOfFile#", 0))
        out.extend(struct.pack("<f", resolution))
    return bytes(out)


@pytest.mark.parametrize("version", [6, 7])
def test_ofp_cwa_odol_fast_and_explicit_lods(version: int) -> None:
    data = _legacy_odol(version, ((1.0, 6), (7.0, 3), (1.0e13, 4)), proxy=True)
    options = lods.list_lods(data)
    assert len(options) == 3
    assert [option.visual for option in options] == [True, True, False]
    assert options[2].label.endswith("[special]")
    fast = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="fast",
    )
    full = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="full",
    )
    chosen = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="lod:0",
    )
    assert fast.measurement.lod == "resolution 7"
    assert fast.original_vertex_count == 3
    assert fast.texture_paths == (r"data\detail1.paa",)
    assert fast.selected_lod_quality == "fast"
    assert full.original_vertex_count == 6
    assert chosen.original_vertex_count == 6
    assert fast.available_lods == options
    with pytest.raises(ValueError, match="unavailable"):
        preview.build_preview_model(
            data, model_path="house.p3d", source="models.pbo",
            lod_quality="lod:99",
        )


def test_ofp_mlod_can_preview_each_visual_and_special_lod() -> None:
    data = _mlod(((1.0, 6), (9.0, 3), (1.0e13, 4)))
    options = lods.list_lods(data)
    assert len(options) == 3
    fast = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="fast",
    )
    full = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="full",
    )
    special = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="lod:2",
    )
    assert fast.measurement.lod == "resolution 9"
    assert fast.original_vertex_count == 3
    assert fast.texture_paths == (r"data\lod1.paa",)
    assert full.original_vertex_count == 6
    assert special.original_vertex_count == 4
    assert special.measurement.lod.startswith("resolution ")
    assert special.available_lods[2].visual is False


def test_legacy_fast_falls_back_to_next_visual_lod(monkeypatch) -> None:
    data = _mlod(((1.0, 6), (7.0, 3)))
    original = preview.extract_first_lod_geometry
    seen = []

    def reject_coarse(*args, **kwargs):
        seen.append(kwargs["lod_quality"])
        if kwargs["lod_quality"] == "lod:1":
            raise preview.measure.ModelReadError("corrupt coarse LOD")
        return original(*args, **kwargs)

    monkeypatch.setattr(preview, "extract_first_lod_geometry", reject_coarse)
    model = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="fast",
    )
    assert seen == ["lod:1", "lod:0"]
    assert model.original_vertex_count == 6


def test_browser_lists_all_model_lods_and_resolves_selected_level() -> None:
    model = preview.build_preview_model(
        _mlod(((1.0, 6), (9.0, 3), (1.0e13, 4))),
        model_path="house.p3d", source="models.pbo",
        lod_quality="fast",
    )
    app = browser.AssetBrowserApp.__new__(browser.AssetBrowserApp)
    app.lod_picker = Mock()
    app.preview_lod_var = SimpleNamespace(set=Mock(), get=lambda: "LOD 2 (9)")
    app._lod_current_quality = "fast"
    app._update_lod_picker(model)
    assert len(app._lod_choices) == 5
    assert app._lod_choices["LOD 2 (9)"] == "lod:1"
    assert app._lod_choices["LOD 3 (1e+13) [special]"] == "lod:2"
    assert "LOD 2 (9)" in app.lod_picker.configure.call_args.kwargs["values"]


def test_categoriser_reloads_only_current_lod_without_touching_classifications(
    monkeypatch,
) -> None:
    data = _mlod(((1.0, 6), (9.0, 3)))
    original = preview.build_preview_model(
        data, model_path="house.p3d", source="models.pbo", lod_quality="fast",
    )
    app = categoriser.CategoriserApp.__new__(categoriser.CategoriserApp)
    app.current = original
    app.models = [original]
    app.index = 0
    app.state = {"house.p3d": object()}
    before = app.state["house.p3d"]
    app.model_iter = iter(())
    original_iterator = app.model_iter
    app._lod_choices = {"LOD 1 (1)": "lod:0", "Fast (lowest)": "fast"}
    app.preview_lod_var = SimpleNamespace(get=lambda: "LOD 1 (1)")
    app._show_model = Mock()
    monkeypatch.setattr(
        categoriser.world_assets, "read_asset_record_bytes",
        lambda record: data,
    )
    app._change_preview_lod()

    assert app.current is original  # Replaced in UI by _show_model normally.
    assert app.models[0].original_vertex_count == 6
    assert app.models[0].selected_lod_quality == "lod:0"
    assert app.state["house.p3d"] is before
    assert app.model_iter is original_iterator
    app._show_model.assert_called_once_with(app.models[0])
