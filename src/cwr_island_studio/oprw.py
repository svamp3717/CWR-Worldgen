# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only OFP / Cold War Assault binarized OPRW v2/v3 terrain reader.

Packed grids are decoded independently; original object records remain mapped
in place (56 bytes per placement). No conversion to 4WVR or write support.
See https://community.bistudio.com/wiki/Wrp_File_Format_-_OPRW2_%26_3
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
import math
import mmap
import struct
import threading
from typing import BinaryIO

import numpy as np

from .core import IslandDocument, MapObject, _BUCKET_METRES
from .object_layers import is_forest_model
from .roads import is_road_model

_UINT = struct.Struct("<I")
_DIMS = struct.Struct("<IIII")
_OBJECT_SIZE = 56
# OFP object placement: uint32 ID, uint32 model table index, 12 float matrix.
_RECORD_DTYPE = np.dtype([
    ("object_id", "<u4"),
    ("model", "<u4"),
    ("transform", "<f4", (12,)),
])
assert _RECORD_DTYPE.itemsize == _OBJECT_SIZE
_MAX_CELLS = 32 * 1024 * 1024
_MAX_NAMES = 2_000_000
_MAX_STRING = 4096


def _exact(stream: BinaryIO, amount: int, label: str) -> bytes:
    data = stream.read(amount)
    if len(data) != amount:
        raise ValueError(f"truncated OPRW {label}")
    return data


def _u32(stream: BinaryIO, label: str) -> int:
    return _UINT.unpack(_exact(stream, 4, label))[0]


def _name(stream: BinaryIO, label: str) -> str:
    result = bytearray()
    for _ in range(_MAX_STRING):
        value = _exact(stream, 1, label)[0]
        if value == 0:
            return result.decode("latin-1")
        result.append(value)
    raise ValueError(f"OPRW {label} is too long")


def _read_packed(
    stream: BinaryIO, expected: int, label: str,
    cancel_event: threading.Event | None = None,
) -> bytes:
    """Read one OFP packed grid (LZSS with a trailing additive checksum).

    Packed grid sizes are not stored in the file. The next grid starts
    immediately after the checksum for the preceding grid.
    """
    if not 0 < expected <= _MAX_CELLS * 4:
        raise ValueError(f"invalid OPRW {label} size")
    # Poseidon SerializeBinStream writes arrays below 1024 bytes directly.
    if expected < 1024:
        return _exact(stream, expected, label)
    output = bytearray(expected)
    dictionary = bytearray(b" " * 4096)
    write_at = dict_at = 0
    while write_at < expected:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        flags = _exact(stream, 1, label)[0]
        for bit in range(8):
            if write_at >= expected:
                break
            if flags & (1 << bit):
                value = _exact(stream, 1, label)[0]
                output[write_at] = dictionary[dict_at] = value
                write_at += 1
                dict_at = (dict_at + 1) & 4095
            else:
                pair = _exact(stream, 2, label)
                distance = pair[0] | ((pair[1] & 0xF0) << 4)
                length = min(3 + (pair[1] & 15), expected - write_at)
                source_at = dict_at - distance
                for index in range(length):
                    value = dictionary[(source_at + index) & 4095]
                    output[write_at] = dictionary[dict_at] = value
                    write_at += 1
                    dict_at = (dict_at + 1) & 4095
    checksum = _u32(stream, f"{label} checksum")
    if checksum != (sum(output) & 0xFFFFFFFF):
        raise ValueError(f"OPRW {label} checksum mismatch")
    return bytes(output)


class OprwIslandDocument(IslandDocument):
    """OPRW v2/v3 document using Island Studio's existing read-only viewer."""

    def __init__(
        self, path: Path | str, *, cell_size: float = 50.0,
        height_scale: float = 1.0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if not (math.isfinite(cell_size) and cell_size > 0):
            raise ValueError("cell size must be positive and finite")
        if not (math.isfinite(height_scale) and height_scale > 0):
            raise ValueError("height scale must be positive and finite")
        self.cell_size = float(cell_size)
        self.height_scale = float(height_scale)
        self.source_label = self.path.name
        self._read_only = self._read_only_forced = True
        self._archive_owner = None
        self._records = self._raw_records = None
        self.heights = self.textures = None
        self._map = None
        self._file = self.path.open("rb")
        try:
            self._parse(cancel_event)
        except BaseException:
            self._records = self._raw_records = None
            self.heights = self.textures = None
            try:
                if self._map is not None:
                    self._map.close()
                    self._map = None
            finally:
                self._file.close()
            raise

        self.height_edits: dict[int, int] = {}
        self.texture_edits: dict[int, int] = {}
        self.record_edits: dict[int, bytes] = {}
        self.deleted: set[int] = set()
        self.added: list[bytes] = []
        self._index_order = self._index_keys = None
        self._index_columns = 0
        self._road_mask = self._forest_mask = None
        self._last_object_id = None

    def _parse(self, cancel_event: threading.Event | None) -> None:
        stream = self._file
        if _exact(stream, 4, "signature") != b"OPRW":
            raise ValueError("not an OPRW file")
        version = _u32(stream, "version")
        if version not in (2, 3):
            raise ValueError(
                f"Unsupported binarized OPRW version {version}; "
                "Island Studio supports OFP/CWA OPRW 2 and 3 only"
            )
        self.oprw_version = version
        if version == 2:
            layer_x = layer_y = map_x = map_y = 256
        else:
            layer_x, layer_y, map_x, map_y = _DIMS.unpack(
                _exact(stream, _DIMS.size, "dimensions")
            )
        layer_count = layer_x * layer_y
        map_count = map_x * map_y
        if (min(layer_x, layer_y, map_x, map_y) <= 0
                or max(layer_count, map_count) > _MAX_CELLS):
            raise ValueError("invalid or excessive OPRW grid dimensions")
        self.width, self.height = map_x, map_y
        self.layer_width, self.layer_height = layer_x, layer_y
        self.cell_count = map_count

        _read_packed(stream, layer_count * 4, "cell flags", cancel_event)
        _read_packed(stream, layer_count, "environment sounds", cancel_event)
        peaks = _u32(stream, "peak count")
        if peaks > _MAX_CELLS:
            raise ValueError("excessive OPRW peak count")
        if peaks * 12 > self.path.stat().st_size - stream.tell():
            raise ValueError("truncated OPRW peaks")
        stream.seek(peaks * 12, 1)

        tex_bytes = _read_packed(
            stream, layer_count * 2, "texture indexes", cancel_event,
        )
        _read_packed(stream, layer_count * 4, "extended flags", cancel_event)
        height_bytes = _read_packed(
            stream, map_count * 4, "terrain elevations", cancel_event,
        )
        self.textures = np.frombuffer(
            tex_bytes, dtype="<u2",
        ).reshape(layer_y, layer_x)
        self.heights = np.frombuffer(
            height_bytes, dtype="<f4",
        ).reshape(map_y, map_x)

        texture_count = _u32(stream, "texture count")
        if not 0 < texture_count <= _MAX_NAMES:
            raise ValueError("invalid OPRW texture count")
        texture_paths: list[str] = []
        for index in range(texture_count):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()
            texture_paths.append(_name(stream, f"texture {index}"))
            _exact(stream, 1, "texture colour")
        self.texture_paths = tuple(texture_paths)

        model_count = _u32(stream, "model count")
        if model_count > _MAX_NAMES:
            raise ValueError("invalid OPRW model count")
        self.model_names = tuple(
            _name(stream, f"model {index}") for index in range(model_count)
        )
        self.object_offset = stream.tell()
        self._map = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        remainder = len(self._map) - self.object_offset
        if remainder < 4 or (remainder - 4) % _OBJECT_SIZE != 0:
            raise ValueError("invalid OPRW object record alignment")
        if self._map[-4:] != b"\xff\xff\xff\xff":
            raise ValueError("OPRW is missing the object terminator")
        self.object_count = (remainder - 4) // _OBJECT_SIZE
        self.has_terminator = True
        self._raw_records = np.ndarray(
            (self.object_count,), dtype=_RECORD_DTYPE,
            buffer=self._map, offset=self.object_offset,
        )
        self._records = self._raw_records
        if self.object_count and (
            not model_count or np.any(self._records["model"] >= model_count)
        ):
            raise ValueError("OPRW object references an invalid model index")

    def object_at(self, index: int) -> MapObject:
        if not 0 <= index < self.object_count:
            raise IndexError(index)
        record = self._records[index]
        transform = record["transform"]
        heading = math.degrees(
            math.atan2(-float(transform[2]), float(transform[0]))
        ) % 360.0
        return MapObject(
            index=index,
            object_id=int(record["object_id"]),
            model_path=self.model_names[int(record["model"])],
            x=float(transform[9]), y=float(transform[10]),
            z=float(transform[11]), heading=heading,
            basis_xx=float(transform[0]), basis_xz=float(transform[6]),
            basis_zx=float(transform[2]), basis_zz=float(transform[8]),
        )

    def terrain_at(self, column: int, row: int) -> float:
        if not 0 <= column < self.width or not 0 <= row < self.height:
            raise IndexError((column, row))
        return float(self.heights[row, column]) * self.height_scale

    def texture_at(self, column: int, row: int) -> int:
        if not 0 <= column < self.width or not 0 <= row < self.height:
            raise IndexError((column, row))
        layer_col = min(self.layer_width - 1, column * self.layer_width // self.width)
        layer_row = min(self.layer_height - 1, row * self.layer_height // self.height)
        return int(self.textures[layer_row, layer_col])

    def sample_viewport(
        self, left: float, top: float, right: float, bottom: float,
        width: int, height: int, *, textures: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = max(1, min(1600, int(width)))
        height = max(1, min(1200, int(height)))
        world_x = np.linspace(left, right, width, endpoint=False)
        world_z = np.linspace(top, bottom, height, endpoint=False)
        if top > bottom:
            # Island Studio displays geographic north at the top, opposite
            # the raw WRP south-to-north row order. Sample pixel centres on
            # descending views so both edge rows appear exactly once.
            world_z -= (top - bottom) / (2 * height)
        if textures:
            # Layer and terrain dimensions can differ in Resistance OPRW.
            columns = np.clip(
                (world_x / self.world_width * self.layer_width).astype(np.int64),
                0, self.layer_width - 1,
            )
            rows = np.clip(
                (world_z / self.world_height * self.layer_height).astype(np.int64),
                0, self.layer_height - 1,
            )
            sampled = self.textures[np.ix_(rows, columns)].astype(np.int32)
        else:
            columns = np.clip((world_x / self.cell_size).astype(np.int64), 0, self.width - 1)
            rows = np.clip((world_z / self.cell_size).astype(np.int64), 0, self.height - 1)
            sampled = self.heights[np.ix_(rows, columns)].astype(np.float32)
        return sampled, world_x, world_z

    def _category_filter(self, hits: np.ndarray, *, forest: bool) -> np.ndarray:
        """Apply model-table metadata overrides without a placement-wide scan."""
        mask = self._forest_mask if forest else self._road_mask
        if mask is not None:
            selected = mask[hits].copy()
        else:
            predicate = is_forest_model if forest else is_road_model
            selected = np.fromiter(
                (predicate(self.model_names[int(self._records["model"][int(index)])])
                 for index in hits),
                dtype=np.bool_, count=len(hits),
            )
        categories = getattr(self, "_model_categories", {})
        if categories:
            # OPRW stores one model-table index per placement, so resolving
            # metadata against model types is much cheaper than per-object paths.
            positive = []
            negative = []
            for index, path in enumerate(self.model_names):
                override = categories.get(self._model_key(path))
                if override is not None:
                    (positive if (override.forest if forest else override.road)
                     else negative).append(index)
            models = self._records["model"][hits]
            if positive:
                selected |= np.isin(models, positive)
            if negative:
                selected &= ~np.isin(models, negative)
        return hits[selected]

    def build_object_index(
        self, cancel_event: threading.Event | None = None,
    ) -> None:
        """Spatially index compact OPRW object placements, cancellably."""
        def check_cancel() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()

        check_cancel()
        transforms = self._records["transform"]
        columns = max(1, math.ceil(self.world_width / _BUCKET_METRES))
        x, z = transforms[:, 9], transforms[:, 11]
        valid = np.isfinite(x) & np.isfinite(z)
        ix = np.maximum(0, np.floor(x / _BUCKET_METRES)).astype(np.int64)
        iz = np.maximum(0, np.floor(z / _BUCKET_METRES)).astype(np.int64)
        keys = np.where(valid, iz * columns + ix, np.iinfo(np.int64).max)
        order = np.argsort(keys, kind="stable")
        check_cancel()
        self._index_keys = keys[order]
        self._index_order = order.astype(
            np.int32 if self.object_count < 2_147_483_647 else np.int64,
        )
        self._index_columns = columns
        models = self._records["model"]
        def classify(predicate) -> np.ndarray:
            flags = np.empty(len(self.model_names), dtype=np.bool_)
            for index, model in enumerate(self.model_names):
                if index % 4096 == 0:
                    check_cancel()
                flags[index] = predicate(model)
            return flags

        road_types = classify(is_road_model)
        forest_types = classify(is_forest_model)
        check_cancel()
        self._road_mask = road_types[models]
        self._forest_mask = forest_types[models]
        check_cancel()


def open_island_document(
    path: Path | str, *, cell_size: float = 50.0,
    height_scale: float = 1.0, read_only: bool = False,
    cancel_event: threading.Event | None = None,
    cwa_runtime_scale: bool = False,
) -> IslandDocument:
    """Dispatch by signature; optional game-accurate GUI scaling.

    In CWR-CE, OPRW stores float heights in metres, while 4WVR needs
    the user's supplied (normally 0.045) int16-unit height scale.
    Library callers retain the old explicit scaling behavior by default.
    """
    filename = Path(path)
    with filename.open("rb") as stream:
        magic = stream.read(4)
    if magic == b"OPRW":
        return OprwIslandDocument(
            filename, cell_size=cell_size,
            height_scale=1.0 if cwa_runtime_scale else height_scale,
            cancel_event=cancel_event,
        )
    if magic == b"4WVR":
        return IslandDocument(
            filename, cell_size=cell_size, height_scale=height_scale,
            read_only=read_only,
        )
    raise ValueError(
        f"Unsupported WRP format ({magic!r}): expected 4WVR or OPRW v2/v3"
    )
