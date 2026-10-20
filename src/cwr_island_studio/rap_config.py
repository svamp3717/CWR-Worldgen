# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, bounded RAP (config.bin) metadata reader for Island Studio.

Only CfgWorlds/<world>/LandGrid and Names are decoded. Other classes and
values are skipped without constructing an entire game configuration.
This supports the OFP/CWA and common RV RAP hierarchy, not arbitrary future
engine-specific config formats.
"""
from __future__ import annotations

import math
import struct

_MAGIC = b"\x00raP"
_MAX_CONFIG = 16 * 1024 * 1024
_MAX_ENTRIES = 100000
_MAX_DEPTH = 24
_MAX_STRING = 4096
_MAX_LOCATIONS = 5000


class _Rap:
    def __init__(self, data: bytes) -> None:
        if not 16 <= len(data) <= _MAX_CONFIG or data[:4] != _MAGIC:
            raise ValueError("not a supported RAP configuration")
        self.data = data
        self.pos = 16
        self.enum_offset = struct.unpack_from("<I", data, 12)[0]
        # Known RAP signature variants, including OFP/CWA style headers.
        # Do not assume an enum list is present.
        self._active: set[int] = set()

    def _take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise ValueError("truncated RAP value")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def byte(self) -> int:
        return self._take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def cstring(self) -> str:
        end = self.data.find(b"\0", self.pos, min(len(self.data), self.pos + _MAX_STRING + 1))
        if end < 0:
            raise ValueError("unterminated or oversized RAP string")
        text = self.data[self.pos:end].decode("latin-1", "replace")
        self.pos = end + 1
        return text

    def count(self) -> int:
        # BIS compressed integers retain the high bit in the initial byte.
        value = self.byte()
        current = value
        shift = 7
        while current & 0x80:
            current = self.byte()
            value += (current - 1) << shift
            shift += 7
            if shift > 35:
                raise ValueError("invalid compressed RAP integer")
        if not 0 <= value <= _MAX_ENTRIES:
            raise ValueError("implausible RAP entry count")
        return value

    def scalar(self, subtype: int) -> str | float | int:
        if subtype in (0, 4):
            return self.cstring()
        if subtype == 1:
            return struct.unpack("<f", self._take(4))[0]
        if subtype == 2:
            return struct.unpack("<i", self._take(4))[0]
        if subtype == 6:
            return struct.unpack("<q", self._take(8))[0]
        raise ValueError("unknown RAP value type")

    def array(self, depth: int = 0) -> tuple[object, ...]:
        if depth > _MAX_DEPTH:
            raise ValueError("deeply nested RAP array")
        result: list[object] = []
        for _ in range(self.count()):
            subtype = self.byte()
            value = self.array(depth + 1) if subtype == 3 else self.scalar(subtype)
            result.append(value)
        return tuple(result)

    def body(self, offset: int) -> tuple[str, dict[str, object]]:
        if not 16 <= offset < len(self.data) or offset in self._active:
            raise ValueError("invalid cyclic RAP class offset")
        self._active.add(offset)
        try:
            self.pos = offset
            parent = self.cstring()
            values: dict[str, object] = {}
            for _ in range(self.count()):
                tag = self.byte()
                if tag == 0:
                    name = self.cstring()
                    values[name.casefold()] = ("class", self.u32())
                elif tag == 1:
                    subtype = self.byte()
                    name = self.cstring()
                    values[name.casefold()] = self.scalar(subtype)
                elif tag == 2:
                    name = self.cstring()
                    values[name.casefold()] = self.array()
                elif tag in (3, 4):
                    self.cstring()  # extern / delete class
                elif tag == 5:
                    # Array append flags precede the field name in RAP.
                    self.u32()
                    name = self.cstring()
                    values[name.casefold()] = self.array()
                else:
                    raise ValueError("unknown RAP class entry type")
            return parent, values
        finally:
            self._active.discard(offset)


def parse_rap_world(
    source: bytes, world_name: str,
) -> tuple[float | None, tuple[tuple[str, float, float], ...]] | None:
    """Return LandGrid and labels, or None if no matching world class exists.

    A corrupt config is rejected by ValueError, not silently treated as text.
    """
    rap = _Rap(source)
    _base, root = rap.body(16)
    worlds = root.get("cfgworlds")
    if not isinstance(worlds, tuple) or worlds[0] != "class":
        return None
    _base, all_worlds = rap.body(worlds[1])
    world = all_worlds.get(world_name.casefold())
    if not isinstance(world, tuple) or world[0] != "class":
        return None

    # Config inheritance is local to CfgWorlds here. The actual engine can
    # merge classes across addons, which cannot be resolved from one PBO.
    def inherited(offset: int) -> dict[str, object]:
        chain: list[dict[str, object]] = []
        seen: set[int] = set()
        for _ in range(_MAX_DEPTH):
            if offset in seen:
                raise ValueError("cyclic RAP world inheritance")
            seen.add(offset)
            parent, fields = rap.body(offset)
            chain.append(fields)
            previous = all_worlds.get(parent.casefold())
            if not parent or not isinstance(previous, tuple) or previous[0] != "class":
                break
            offset = previous[1]
        combined: dict[str, object] = {}
        for item in reversed(chain):
            combined.update(item)
        return combined

    data = inherited(world[1])
    grid = data.get("landgrid")
    if isinstance(grid, (int, float, str)):
        try:
            grid = float(grid)
            if not math.isfinite(grid) or not 0 < grid <= 10000:
                grid = None
        except (ValueError, OverflowError):
            grid = None
    else:
        grid = None
    names = data.get("names")
    places: list[tuple[str, float, float]] = []
    if isinstance(names, tuple) and names[0] == "class":
        _parent, entries = rap.body(names[1])
        for value in entries.values():
            if not isinstance(value, tuple) or value[0] != "class":
                continue
            _parent, marker = rap.body(value[1])
            name = marker.get("name")
            position = marker.get("position")
            if isinstance(name, str) and name and len(name) <= 160 and isinstance(position, tuple) and len(position) >= 2:
                try:
                    x, z = float(position[0]), float(position[1])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(x) and math.isfinite(z):
                    places.append((name, x, z))
            if len(places) >= _MAX_LOCATIONS:
                break
    return grid, tuple(places)
