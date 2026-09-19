from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_p3d_models as measure


def test_measurement_accepts_numpy_vertex_array() -> None:
    points = np.array(
        [
            [-2.0, -1.5, -4.0],
            [3.0, 2.5, 6.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    result = measure._measurement(
        model_path=r"o\hous\example.p3d",
        source="fixture",
        format_name="ODOL",
        version=7,
        lod="first",
        points=points,
    )

    assert result.vertex_count == 3
    assert result.width_m == pytest.approx(5.0)
    assert result.height_m == pytest.approx(4.0)
    assert result.length_m == pytest.approx(10.0)
    assert result.origin_to_bottom_m == pytest.approx(1.5)

def test_count_models_reads_pbo_header_without_parsing_payloads(tmp_path: Path) -> None:
    entry = struct.Struct("<IIIII")
    members = (
        ("a.p3d", b"not actually a valid p3d"),
        ("textures\\leaf.paa", b"texture"),
        ("sub\\b.p3d", b"also deliberately invalid"),
    )
    header = bytearray()
    payload = bytearray()
    for name, data in members:
        header += name.encode("latin-1") + b"\0"
        header += entry.pack(0, len(data), 0, 0, len(data))
        payload += data
    header += b"\0" + entry.pack(0, 0, 0, 0, 0)

    pbo = tmp_path / "archive.pbo"
    pbo.write_bytes(bytes(header + payload))

    assert measure.count_models((pbo,)) == 2
    assert measure.count_models((pbo,), (r"archive\\sub\\*.p3d",)) == 1



def test_cprs_empty_name_record_is_accepted_as_pbo_terminator(tmp_path: Path) -> None:
    entry = struct.Struct("<IIIII")
    members = (
        ("MAP_Shed_brown.p3d", b"brown"),
        ("metal.paa", b"texture"),
        ("MAP_Shed_green.p3d", b"green"),
        ("MAP_Shed_grey.p3d", b"grey"),
    )
    header = bytearray()
    payload = bytearray()
    for name, data in members:
        header += name.encode("latin-1") + b"\0"
        header += entry.pack(measure._PBO_COMPRESSED, len(data), 0, 0, len(data))
        payload += data

    # MAP_Shed.pbo and some other legacy/addon archives use an empty-name Cprs
    # record with zero sizes as the header terminator.
    header += b"\0" + entry.pack(measure._PBO_COMPRESSED, 0, 0, 0, 0)

    pbo = tmp_path / "MAP_Shed.pbo"
    pbo.write_bytes(bytes(header + payload))

    assert measure.count_models((pbo,)) == 3
    assert [model_path for model_path, _data in measure._pbo_entries(pbo)] == [
        r"map_shed\map_shed_brown.p3d",
        r"map_shed\map_shed_green.p3d",
        r"map_shed\map_shed_grey.p3d",
    ]


def test_unknown_empty_name_pbo_extension_record_remains_rejected(tmp_path: Path) -> None:
    entry = struct.Struct("<IIIII")
    pbo = tmp_path / "bad-extension.pbo"
    pbo.write_bytes(b"\0" + entry.pack(0x12345678, 1, 0, 0, 0))

    with pytest.raises(measure.ModelReadError, match="unsupported PBO extension record"):
        measure.count_models((pbo,))
