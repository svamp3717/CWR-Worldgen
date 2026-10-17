# SPDX-License-Identifier: GPL-3.0-or-later
"""Scan OFP/CWA, Arma and DayZ WRP files for P3D model dependencies.

The scanner accepts loose WRPs, PBOs and Iron Front IFA archives.
Unbinarized 4WVR, 8WVR and DayZ 0WZD objects are counted from their records.
Binarized OPRW worlds use conservative P3D string discovery: those counts
are model-name occurrences, NOT object placement counts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path
import argparse
import csv
import io
import json
import queue
import re
import struct
import threading
from typing import Iterable, Sequence

from .assets import _decompress_lzss_stream
from .pbo import (
    _is_pbo_header_terminator,
    _read_exact,
    _skip_exact,
    is_pbo_path,
    open_pbo_stream,
)

_PBO_ENTRY = struct.Struct("<IIIII")
_PBO_PROPERTIES = 0x56657273  # 'Vers'
_PBO_COMPRESSED = 0x43707273  # 'Cprs'
_RVW4_HEADER = struct.Struct("<4sii")
_RVW4_OBJECT = struct.Struct("<12fi76s")
_RVW4_TEXTURE_BYTES = 512 * 32
_8WVR_HEADER = struct.Struct("<4sIIIIf")
_8WVR_OBJECT_PREFIX = struct.Struct("<12fII")
DEFAULT_STOCK_ROOTS = ("data3d", "o")
GAME_STOCK_ROOTS = {
    "cwa": DEFAULT_STOCK_ROOTS,
    "arma1": ("ca",),
    # Queen's Gambit (Arma Gold) adds DBE1 namespaces on top of Arma 1.
    "arma1qg": ("ca", "dbe1", "models_dbe1", "sara_dbe1"),
    "arma2": ("ca",),
    "arma2oa": ("ca", "ca_e"),
    "arma3": ("a3",),
    "dayz": ("dz",),
    # Original 2012 Iron Front uses Arma 2 technology and LIB_ addons.
    # lib_* intentionally matches addon namespace segments, not full paths.
    "ironfront": ("lib", "lib_*", "ca", "ca_e"),
}
GAME_LABELS = {
    "CWA / OFP": "cwa",
    "Arma 1": "arma1",
    "Arma 1 + Queen's Gambit": "arma1qg",
    "Arma 2": "arma2",
    "Arma 2 OA": "arma2oa",
    "Arma 3": "arma3",
    "DayZ Standalone": "dayz",
    "Iron Front": "ironfront",
}

_ASCII_RUN = re.compile(rb"[ -~]{5,}")
_P3D_PATH = re.compile(
    r"(?i)(?:[a-z0-9_.$@()+\-]+[\\/])"
    r"(?:(?:[a-z0-9_.$@()+\- ]+)[\\/])*"
    r"(?:[a-z0-9_.$@()+\- ]+)\.p3d"
)


@dataclass(frozen=True, slots=True)
class DependencyRow:
    wrp_name: str
    model_path: str
    namespace: str
    references: int
    is_stock: bool
    parser: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    input_path: str
    input_kind: str
    stock_roots: tuple[str, ...]
    wrp_names: tuple[str, ...]
    rows: tuple[DependencyRow, ...]
    warnings: tuple[str, ...] = ()

    @property
    def wrp_count(self) -> int:
        return len(self.wrp_names)

    @property
    def unique_models(self) -> int:
        return len({row.model_path for row in self.rows})

    @property
    def unique_mod_models(self) -> int:
        return len({row.model_path for row in self.rows if not row.is_stock})


def canonical_model_path(value: object) -> str:
    path = str(value or "").replace("/", "\\").strip().lstrip("\\")
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path.casefold()


def normalize_stock_roots(values: Iterable[str] | None) -> tuple[str, ...]:
    roots: list[str] = []
    for raw in values or DEFAULT_STOCK_ROOTS:
        for part in re.split(r"[;,]", str(raw)):
            root = canonical_model_path(part).strip("\\")
            if "\\" in root:
                root = root.split("\\", 1)[0]
            if root and root not in roots:
                roots.append(root)
    return tuple(roots or DEFAULT_STOCK_ROOTS)


def model_namespace(model_path: str) -> str:
    canonical = canonical_model_path(model_path)
    if "\\" not in canonical:
        return "(root)"
    return canonical.split("\\", 1)[0]


def _is_stock_model(model_path: str, stock_roots: Sequence[str]) -> bool:
    namespace = model_namespace(model_path)
    return any(
        fnmatchcase(namespace, root.casefold()) if "*" in root or "?" in root
        else namespace == root.casefold()
        for root in stock_roots
    )


def _read_cstring(stream: io.BytesIO) -> str:
    value = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("truncated PBO string")
        if byte == b"\0":
            return value.decode("latin-1")
        value.extend(byte)


def _pbo_wrp_entries(path: Path) -> tuple[tuple[str, bytes], ...]:
    metadata: list[tuple[str, int, int, int]] = []
    worlds: list[tuple[str, bytes]] = []

    with open_pbo_stream(path) as stream:
        while True:
            name = _read_cstring(stream)
            fields = _read_exact(stream, _PBO_ENTRY.size, "PBO entry header")
            packing, original_size, reserved, timestamp, data_size = _PBO_ENTRY.unpack(fields)

            if not name:
                if packing == _PBO_PROPERTIES:
                    while True:
                        key = _read_cstring(stream)
                        if not key:
                            break
                        _read_cstring(stream)
                    continue
                if _is_pbo_header_terminator(
                    packing, original_size, reserved, timestamp, data_size
                ):
                    break
                raise ValueError(
                    "unsupported PBO extension record "
                    f"(packing={packing:#x}, original_size={original_size}, "
                    f"reserved={reserved}, timestamp={timestamp}, data_size={data_size})"
                )

            metadata.append((name, packing, original_size, data_size))

        for name, packing, original_size, data_size in metadata:
            if Path(name.replace("\\", "/")).suffix.casefold() != ".wrp":
                _skip_exact(stream, data_size, f"PBO entry {name!r}")
                continue

            stored = _read_exact(stream, data_size, f"PBO entry {name!r}")
            if packing == 0:
                data = stored
            elif packing == _PBO_COMPRESSED:
                if original_size <= 0:
                    raise ValueError(f"compressed WRP entry {name!r} has no original size")
                packed = io.BytesIO(stored)
                data = _decompress_lzss_stream(packed, original_size)
                if packed.read():
                    raise ValueError(f"compressed WRP entry {name!r} has trailing bytes")
            else:
                raise ValueError(
                    f"unsupported PBO packing method {packing:#x} for WRP entry {name!r}"
                )
            worlds.append((name.replace("/", "\\"), data))

    return tuple(worlds)

def _rvw4_model_counts(data: bytes) -> Counter[str] | None:
    if len(data) < _RVW4_HEADER.size or data[:4] != b"4WVR":
        return None

    magic, width, height = _RVW4_HEADER.unpack_from(data, 0)
    if magic != b"4WVR" or width <= 0 or height <= 0:
        raise ValueError("invalid RVW4 header")
    cells = width * height
    cursor = _RVW4_HEADER.size + cells * 4 + _RVW4_TEXTURE_BYTES
    if cursor > len(data):
        raise ValueError("truncated RVW4 terrain/texture data")

    counts: Counter[str] = Counter()
    while cursor < len(data):
        end = cursor + _RVW4_OBJECT.size
        if end > len(data):
            raise ValueError("trailing partial RVW4 object record")
        values = _RVW4_OBJECT.unpack_from(data, cursor)
        cursor = end
        raw_model = values[13].split(b"\0", 1)[0]
        if not raw_model:
            break
        try:
            model = canonical_model_path(raw_model.decode("ascii"))
        except UnicodeDecodeError as exc:
            raise ValueError("RVW4 contains a non-ASCII model path") from exc
        if model.endswith(".p3d"):
            counts[model] += 1

    # Some real-world RVW4 files end immediately after the final complete
    # 128-byte object record instead of writing an all-zero terminator. Partial
    # records are still rejected above, so aligned EOF is safe for dependency
    # scanning and matches what tools such as WRPTool encounter in the wild.
    return counts


def _8wvr_model_counts(data: bytes) -> Counter[str] | None:
    """Count unbinarized Arma 8WVR / DayZ 0WZD model paths exactly.

    Both variants have terrain/texture grids, a length-prefixed material
    table (with a dummy entry zero), and length-prefixed object model names.
    A final zero-length model name terminates the object list.
    """
    magic = data[:4]
    if magic not in (b"8WVR", b"0WZD"):
        return None
    kind = magic.decode("ascii")
    if len(data) < _8WVR_HEADER.size:
        raise ValueError(f"truncated {kind} header")
    _, texture_x, texture_y, terrain_x, terrain_y, cell_size = _8WVR_HEADER.unpack_from(data)
    if min(texture_x, texture_y, terrain_x, terrain_y) == 0 or not 0 < cell_size < 1e6:
        raise ValueError(f"invalid {kind} grid sizes or cell size")

    cursor = _8WVR_HEADER.size + terrain_x * terrain_y * 4 + texture_x * texture_y * 2
    if cursor + 4 > len(data):
        raise ValueError(f"truncated {kind} terrain or material table")
    material_count = struct.unpack_from("<I", data, cursor)[0]
    cursor += 4
    if material_count == 0:
        raise ValueError(f"invalid {kind} material count")
    # The first material is an implicit dummy, without a name.
    for _ in range(1, material_count):
        while True:
            if cursor + 4 > len(data):
                raise ValueError(f"truncated {kind} material name")
            length = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
            if length == 0:
                break
            if length > len(data) - cursor:
                raise ValueError(f"truncated {kind} material name")
            cursor += length

    counts: Counter[str] = Counter()
    terminated = False
    while cursor < len(data):
        if cursor + _8WVR_OBJECT_PREFIX.size > len(data):
            raise ValueError(f"truncated {kind} object record")
        values = _8WVR_OBJECT_PREFIX.unpack_from(data, cursor)
        cursor += _8WVR_OBJECT_PREFIX.size
        length = values[-1]
        if length == 0:
            terminated = True
            break
        if length > len(data) - cursor:
            raise ValueError(f"truncated {kind} object model name")
        model = canonical_model_path(data[cursor : cursor + length].rstrip(b"\0").decode("latin-1"))
        cursor += length
        if model.endswith(".p3d"):
            counts[model] += 1
    if not terminated or cursor != len(data):
        raise ValueError(f"missing or invalid {kind} final dummy object")
    return counts


def _legacy_wrp_model_counts(data: bytes) -> Counter[str]:
    """Find model paths in legacy/addon WRP variants without parsing transforms.

    Old OPRW layouts vary enough that a dependency scanner is safer extracting
    their embedded model strings than guessing record offsets. Counts here are
    string occurrences, not guaranteed object-placement counts.
    """

    counts: Counter[str] = Counter()
    for raw_run in _ASCII_RUN.findall(data):
        try:
            run = raw_run.decode("ascii")
        except UnicodeDecodeError:
            continue
        for match in _P3D_PATH.finditer(run):
            model = canonical_model_path(match.group(0))
            if model:
                counts[model] += 1
    return counts


def scan_wrp_bytes(
    data: bytes,
    *,
    wrp_name: str,
    stock_roots: Sequence[str] = DEFAULT_STOCK_ROOTS,
) -> tuple[DependencyRow, ...]:
    roots = normalize_stock_roots(stock_roots)
    exact = _rvw4_model_counts(data)
    if exact is not None:
        parser = "RVW4 object records"
    else:
        exact = _8wvr_model_counts(data)
        if exact is not None:
            parser = "0WZD object records" if data[:4] == b"0WZD" else "8WVR object records"
        elif data[:4] == b"OPRW":
            parser = "OPRW P3D string occurrences"
        else:
            parser = "legacy P3D string scan"
    counts = exact if exact is not None else _legacy_wrp_model_counts(data)

    return tuple(
        DependencyRow(
            wrp_name=wrp_name,
            model_path=model,
            namespace=model_namespace(model),
            references=int(count),
            is_stock=_is_stock_model(model, roots),
            parser=parser,
        )
        for model, count in sorted(counts.items())
    )


def scan_dependencies(
    input_path: Path | str,
    *,
    stock_roots: Sequence[str] | None = None,
    game: str = "cwa",
) -> ScanResult:
    if game not in GAME_STOCK_ROOTS:
        raise ValueError(f"unknown game profile: {game!r}")
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    roots = normalize_stock_roots(stock_roots if stock_roots is not None else GAME_STOCK_ROOTS[game])
    suffix = path.suffix.casefold()
    warnings: list[str] = []
    rows: list[DependencyRow] = []

    if suffix == ".wrp":
        wrp_names = (path.name,)
        world_data = path.read_bytes()
        rows.extend(scan_wrp_bytes(world_data, wrp_name=path.name, stock_roots=roots))
        if game == "dayz" and world_data[:4] == b"OPRW":
            warnings.append("DayZ OPRW: only embedded P3D string occurrences can be counted, not object placements")
        kind = "wrp"
    elif is_pbo_path(path) or path.name.casefold().endswith((".ifa", ".ifa.zst")):
        worlds = _pbo_wrp_entries(path)
        wrp_names = tuple(name for name, _data in worlds)
        if not worlds:
            warnings.append("Archive contains no .wrp entries")
        for name, data in worlds:
            try:
                rows.extend(scan_wrp_bytes(data, wrp_name=name, stock_roots=roots))
                if game == "dayz" and data[:4] == b"OPRW":
                    warnings.append(f"{name}: OPRW scan counts P3D strings, not object placements")
            except ValueError as exc:
                warnings.append(f"{name}: {exc}")
        kind = "ifa" if path.name.casefold().endswith((".ifa", ".ifa.zst")) else "pbo"
    else:
        raise ValueError("input must be a .wrp, .pbo, .pbo.zst, .ifa, or .ifa.zst file")

    return ScanResult(
        input_path=str(path),
        input_kind=kind,
        stock_roots=roots,
        wrp_names=wrp_names,
        rows=tuple(rows),
        warnings=tuple(warnings),
    )


def rows_for_display(result: ScanResult, *, include_stock: bool = False) -> tuple[DependencyRow, ...]:
    if include_stock:
        return result.rows
    return tuple(row for row in result.rows if not row.is_stock)


def _sort_display_rows(
    rows: Sequence[DependencyRow], column: str, *, descending: bool = False
) -> tuple[DependencyRow, ...]:
    """Sort visible table rows, treating reference counts as numbers."""
    if column == "refs":
        key = lambda row: row.references
    elif column == "type":
        key = lambda row: row.is_stock  # Mod first, then Stock.
    else:
        attributes = {
            "wrp": "wrp_name",
            "namespace": "namespace",
            "model": "model_path",
            "parser": "parser",
        }
        attribute = attributes[column]
        key = lambda row: getattr(row, attribute).casefold()
    return tuple(sorted(rows, key=key, reverse=descending))


def result_document(
    result: ScanResult, *, include_stock: bool = False
) -> dict[str, object]:
    rows = rows_for_display(result, include_stock=include_stock)
    return {
        "schema": 1,
        "input_path": result.input_path,
        "input_kind": result.input_kind,
        "stock_roots": list(result.stock_roots),
        "wrp_names": list(result.wrp_names),
        "wrp_count": result.wrp_count,
        "unique_models": result.unique_models,
        "unique_mod_models": result.unique_mod_models,
        "unique_mod_dependencies": sorted(
            {row.model_path for row in result.rows if not row.is_stock}
        ),
        "warnings": list(result.warnings),
        "dependencies": [asdict(row) for row in rows],
    }


def write_json_report(
    result: ScanResult, output: Path | str, *, include_stock: bool = False
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result_document(result, include_stock=include_stock),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_csv_report(
    result: ScanResult, output: Path | str, *, include_stock: bool = False
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wrp", "namespace", "model_path", "references", "type", "parser"))
        for row in rows_for_display(result, include_stock=include_stock):
            writer.writerow((
                row.wrp_name,
                row.namespace,
                row.model_path,
                row.references,
                "stock" if row.is_stock else "mod",
                row.parser,
            ))
    return path


def _format_text(result: ScanResult, *, include_stock: bool) -> str:
    rows = rows_for_display(result, include_stock=include_stock)
    lines = [
        f"Input: {result.input_path}",
        f"WRPs: {result.wrp_count}",
        f"Unique P3Ds: {result.unique_models}",
        f"Unique mod P3Ds: {result.unique_mod_models}",
        f"Stock roots: {', '.join(result.stock_roots)}",
    ]
    if result.warnings:
        lines.extend(f"WARNING: {warning}" for warning in result.warnings)
    if not rows:
        lines.append("No matching P3D dependencies found.")
        return "\n".join(lines)

    current = None
    for row in rows:
        if row.wrp_name != current:
            current = row.wrp_name
            lines.append("")
            lines.append(f"[{current}]")
        kind = "stock" if row.is_stock else "mod"
        lines.append(
            f"{row.model_path}  [{kind}; root={row.namespace}; refs={row.references}; {row.parser}]"
        )
    return "\n".join(lines)


class ScannerGui:
    def __init__(self, initial_path: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("WRP Mod Dependency Scanner")
        self.root.geometry("1120x680")
        self.root.minsize(820, 500)

        self.input_var = tk.StringVar(value=str(initial_path or ""))
        self.game_var = tk.StringVar(value="CWA / OFP")
        self.stock_var = tk.StringVar(value="; ".join(DEFAULT_STOCK_ROOTS))
        self.include_stock_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose a PBO or WRP to scan.")
        self._result: ScanResult | None = None
        self._sort_column: str | None = None
        self._sort_descending = False
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build()
        self.root.after(100, self._poll_worker)

    def _build(self) -> None:
        from tkinter import ttk

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(outer, text="Input", padding=10)
        input_frame.pack(fill="x")
        ttk.Entry(input_frame, textvariable=self.input_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(input_frame, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(input_frame, text="Scan", command=self._scan).grid(row=0, column=2, padx=(8, 0))
        input_frame.columnconfigure(0, weight=1)

        options = ttk.Frame(outer)
        options.pack(fill="x", pady=(10, 8))
        ttk.Label(options, text="Game").pack(side="left")
        game_choice = ttk.Combobox(
            options, textvariable=self.game_var, values=tuple(GAME_LABELS), state="readonly", width=22
        )
        game_choice.pack(side="left", padx=(8, 14))
        game_choice.bind("<<ComboboxSelected>>", self._game_changed)
        ttk.Label(options, text="Stock model roots").pack(side="left")
        ttk.Entry(options, textvariable=self.stock_var, width=25).pack(side="left", padx=(8, 14))
        ttk.Checkbutton(
            options,
            text="Show stock models too",
            variable=self.include_stock_var,
            command=self._populate,
        ).pack(side="left")

        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        columns = ("wrp", "namespace", "model", "refs", "type", "parser")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "wrp": "WRP",
            "namespace": "Namespace / PBO root",
            "model": "P3D model",
            "refs": "Refs",
            "type": "Type",
            "parser": "Detection",
        }
        widths = {"wrp": 170, "namespace": 150, "model": 390, "refs": 65, "type": 75, "parser": 150}
        self._column_headings = headings
        for key in columns:
            self.tree.heading(
                key, text=headings[key], command=lambda column=key: self._sort_by(column)
            )
            self.tree.column(key, width=widths[key], anchor="w")
        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Copy P3D paths", command=self._copy_paths).pack(side="left")
        ttk.Button(actions, text="Export CSV…", command=self._export_csv).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Export JSON…", command=self._export_json).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

    def _sort_by(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = False

        for key, label in self._column_headings.items():
            arrow = (" ▼" if self._sort_descending else " ▲") if key == column else ""
            self.tree.heading(key, text=label + arrow)
        self._populate()

    def _game_changed(self, _event: object = None) -> None:
        game = GAME_LABELS[self.game_var.get()]
        self.stock_var.set("; ".join(GAME_STOCK_ROOTS[game]))

    def _browse(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose OFP / Arma / DayZ WRP, PBO or Iron Front IFA",
            filetypes=(("WRP / PBO / IFA", "*.wrp *.pbo *.pbo.zst *.ifa *.ifa.zst"), ("WRP", "*.wrp"), ("PBO / IFA", "*.pbo *.pbo.zst *.ifa *.ifa.zst"), ("All files", "*.*")),
        )
        if selected:
            self.input_var.set(selected)

    def _scan(self) -> None:
        raw_path = self.input_var.get().strip()
        if not raw_path:
            return
        path = Path(raw_path)
        roots = normalize_stock_roots((self.stock_var.get(),))
        game = GAME_LABELS[self.game_var.get()]
        self.status_var.set("Scanning…")
        worker = threading.Thread(target=self._scan_worker, args=(path, roots, game), daemon=True)
        worker.start()

    def _scan_worker(self, path: Path, roots: tuple[str, ...], game: str = "cwa") -> None:
        try:
            self._queue.put(("ok", scan_dependencies(path, stock_roots=roots, game=game)))
        except Exception as exc:  # GUI boundary: surface the error instead of crashing Tk.
            self._queue.put(("error", exc))

    def _poll_worker(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "ok":
                    self._result = payload  # type: ignore[assignment]
                    self._populate()
                else:
                    self.status_var.set("Scan failed.")
                    messagebox.showerror("WRP Mod Dependency Scanner", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def _populate(self) -> None:
        self.tree.delete(*self.tree.get_children())
        result = self._result
        if result is None:
            return
        rows = rows_for_display(result, include_stock=bool(self.include_stock_var.get()))
        if self._sort_column is not None:
            rows = _sort_display_rows(
                rows, self._sort_column, descending=self._sort_descending
            )
        for row in rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    row.wrp_name,
                    row.namespace,
                    row.model_path,
                    row.references,
                    "Stock" if row.is_stock else "Mod",
                    row.parser,
                ),
            )
        suffix = f"; {len(result.warnings)} warning(s)" if result.warnings else ""
        self.status_var.set(
            f"{result.wrp_count} WRP(s), {result.unique_mod_models} unique mod P3D(s){suffix}"
        )

    def _visible_paths(self) -> tuple[str, ...]:
        result = self._result
        if result is None:
            return ()
        rows = rows_for_display(result, include_stock=bool(self.include_stock_var.get()))
        return tuple(sorted({row.model_path for row in rows}))

    def _copy_paths(self) -> None:
        paths = self._visible_paths()
        if not paths:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(paths))
        self.status_var.set(f"Copied {len(paths)} P3D path(s).")

    def _export_csv(self) -> None:
        from tkinter import filedialog

        if self._result is None:
            return
        selected = filedialog.asksaveasfilename(
            title="Export dependency CSV",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
        )
        if selected:
            write_csv_report(
                self._result,
                selected,
                include_stock=bool(self.include_stock_var.get()),
            )
            self.status_var.set(f"Saved {Path(selected).name}")

    def _export_json(self) -> None:
        from tkinter import filedialog

        if self._result is None:
            return
        selected = filedialog.asksaveasfilename(
            title="Export dependency JSON",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
        )
        if selected:
            write_json_report(
                self._result,
                selected,
                include_stock=bool(self.include_stock_var.get()),
            )
            self.status_var.set(f"Saved {Path(selected).name}")

    def run(self) -> int:
        self.root.update_idletasks()
        _dismiss_splash()
        self.root.mainloop()
        return 0


def _dismiss_splash() -> None:
    """Close the one-file bootloader loading screen, if enabled."""
    try:
        import pyi_splash
    except ImportError:
        return
    try:
        pyi_splash.close()
    except (OSError, RuntimeError):
        pass


def gui_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", nargs="?", type=Path)
    args, _unknown = parser.parse_known_args(argv)
    return ScannerGui(args.input).run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cwr-wrp-mod-scan",
        description="List P3D dependencies in OFP, Arma, DayZ and Iron Front WRP/PBO/IFA archives.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="WRP file, PBO or Iron Front IFA archive containing WRP files")
    parser.add_argument(
        "--game", choices=tuple(GAME_STOCK_ROOTS), default="cwa",
        help="Stock asset profile: cwa (default), arma1, arma1qg, arma2, arma2oa, arma3, dayz, ironfront",
    )
    parser.add_argument(
        "--stock-root",
        action="append",
        default=None,
        help="Override the selected game's stock roots; repeat as needed",
    )
    parser.add_argument("--include-stock", action="store_true", help="Show stock P3D references too")
    parser.add_argument("--json", type=Path, help="Write a JSON report")
    parser.add_argument("--csv", type=Path, help="Write a CSV report")
    parser.add_argument("--gui", action="store_true", help="Open the graphical scanner")
    args = parser.parse_args(argv)

    if args.gui or args.input is None:
        return ScannerGui(args.input).run()

    _dismiss_splash()  # CLI mode has no Tk window; close before scanning.
    result = scan_dependencies(args.input, game=args.game, stock_roots=args.stock_root)
    if args.json:
        write_json_report(result, args.json, include_stock=args.include_stock)
    if args.csv:
        write_csv_report(result, args.csv, include_stock=args.include_stock)
    print(_format_text(result, include_stock=args.include_stock))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
