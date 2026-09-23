# SPDX-License-Identifier: GPL-3.0-or-later
"""Scan CWA/OFP WRP files for P3D model dependencies.

The scanner accepts either a loose WRP or a PBO containing one or more WRP
members. RVW4/4WVR worlds are parsed exactly. Older/addon WRP variants fall back
to conservative ASCII P3D-path discovery so the utility remains useful for
legacy worlds without pretending to understand transforms it does not need.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
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

_PBO_ENTRY = struct.Struct("<IIIII")
_PBO_PROPERTIES = 0x56657273  # 'Vers'
_PBO_COMPRESSED = 0x43707273  # 'Cprs'
_RVW4_HEADER = struct.Struct("<4sii")
_RVW4_OBJECT = struct.Struct("<12fi76s")
_RVW4_TEXTURE_BYTES = 512 * 32
DEFAULT_STOCK_ROOTS = ("data3d", "o")

_ASCII_RUN = re.compile(rb"[ -~]{5,}")
_P3D_PATH = re.compile(
    r"(?i)(?:(?:[a-z0-9_.$@()+\- ]+)[\\/])+(?:[a-z0-9_.$@()+\- ]+)\.p3d"
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
    rows: tuple[DependencyRow, ...]
    warnings: tuple[str, ...] = ()

    @property
    def wrp_count(self) -> int:
        return len({row.wrp_name for row in self.rows})

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
    return model_namespace(model_path) in {root.casefold() for root in stock_roots}


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
    raw = path.read_bytes()
    stream = io.BytesIO(raw)
    metadata: list[tuple[str, int, int, int]] = []

    while True:
        name = _read_cstring(stream)
        fields = stream.read(_PBO_ENTRY.size)
        if len(fields) != _PBO_ENTRY.size:
            raise ValueError("truncated PBO entry header")
        packing, original_size, reserved, timestamp, data_size = _PBO_ENTRY.unpack(fields)

        if not name:
            if packing == _PBO_PROPERTIES:
                while True:
                    key = _read_cstring(stream)
                    if not key:
                        break
                    _read_cstring(stream)
                continue
            if (
                packing in {0, _PBO_COMPRESSED}
                and original_size == 0
                and reserved == 0
                and timestamp == 0
                and data_size == 0
            ):
                break
            raise ValueError(
                "unsupported PBO extension record "
                f"(packing={packing:#x}, original_size={original_size}, "
                f"reserved={reserved}, timestamp={timestamp}, data_size={data_size})"
            )

        metadata.append((name, packing, original_size, data_size))

    cursor = stream.tell()
    worlds: list[tuple[str, bytes]] = []
    for name, packing, original_size, data_size in metadata:
        end = cursor + data_size
        if end > len(raw):
            raise ValueError(f"truncated PBO entry {name!r}")
        stored = raw[cursor:end]
        cursor = end
        if Path(name.replace("\\", "/")).suffix.casefold() != ".wrp":
            continue

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
    found_terminator = False
    while cursor < len(data):
        end = cursor + _RVW4_OBJECT.size
        if end > len(data):
            raise ValueError("trailing partial RVW4 object record")
        values = _RVW4_OBJECT.unpack_from(data, cursor)
        cursor = end
        raw_model = values[13].split(b"\0", 1)[0]
        if not raw_model:
            found_terminator = True
            break
        try:
            model = canonical_model_path(raw_model.decode("ascii"))
        except UnicodeDecodeError as exc:
            raise ValueError("RVW4 contains a non-ASCII model path") from exc
        if model.endswith(".p3d"):
            counts[model] += 1

    if not found_terminator:
        raise ValueError("RVW4 object list is missing its terminator")
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
    parser = "RVW4 object records" if exact is not None else "legacy P3D string scan"
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
    stock_roots: Sequence[str] = DEFAULT_STOCK_ROOTS,
) -> ScanResult:
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    roots = normalize_stock_roots(stock_roots)
    suffix = path.suffix.casefold()
    warnings: list[str] = []
    rows: list[DependencyRow] = []

    if suffix == ".wrp":
        rows.extend(scan_wrp_bytes(path.read_bytes(), wrp_name=path.name, stock_roots=roots))
        kind = "wrp"
    elif suffix == ".pbo":
        worlds = _pbo_wrp_entries(path)
        if not worlds:
            warnings.append("PBO contains no .wrp entries")
        for name, data in worlds:
            try:
                rows.extend(scan_wrp_bytes(data, wrp_name=name, stock_roots=roots))
            except ValueError as exc:
                warnings.append(f"{name}: {exc}")
        kind = "pbo"
    else:
        raise ValueError("input must be a .wrp or .pbo file")

    return ScanResult(
        input_path=str(path),
        input_kind=kind,
        stock_roots=roots,
        rows=tuple(rows),
        warnings=tuple(warnings),
    )


def rows_for_display(result: ScanResult, *, include_stock: bool = False) -> tuple[DependencyRow, ...]:
    if include_stock:
        return result.rows
    return tuple(row for row in result.rows if not row.is_stock)


def result_document(result: ScanResult) -> dict[str, object]:
    return {
        "schema": 1,
        "input_path": result.input_path,
        "input_kind": result.input_kind,
        "stock_roots": list(result.stock_roots),
        "wrp_count": result.wrp_count,
        "unique_models": result.unique_models,
        "unique_mod_models": result.unique_mod_models,
        "warnings": list(result.warnings),
        "dependencies": [asdict(row) for row in result.rows],
    }


def write_json_report(result: ScanResult, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_document(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_csv_report(result: ScanResult, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wrp", "namespace", "model_path", "references", "type", "parser"))
        for row in result.rows:
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
        self.stock_var = tk.StringVar(value="; ".join(DEFAULT_STOCK_ROOTS))
        self.include_stock_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose a PBO or WRP to scan.")
        self._result: ScanResult | None = None
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
        ttk.Label(options, text="Stock model roots").pack(side="left")
        ttk.Entry(options, textvariable=self.stock_var, width=30).pack(side="left", padx=(8, 14))
        ttk.Checkbutton(
            options,
            text="Show stock CWA/OFP models too",
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
        for key in columns:
            self.tree.heading(key, text=headings[key])
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

    def _browse(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose WRP or PBO",
            filetypes=(("WRP / PBO", "*.wrp *.pbo"), ("WRP", "*.wrp"), ("PBO", "*.pbo"), ("All files", "*.*")),
        )
        if selected:
            self.input_var.set(selected)

    def _scan(self) -> None:
        path = Path(self.input_var.get().strip())
        roots = normalize_stock_roots((self.stock_var.get(),))
        if not str(path):
            return
        self.status_var.set("Scanning…")
        worker = threading.Thread(target=self._scan_worker, args=(path, roots), daemon=True)
        worker.start()

    def _scan_worker(self, path: Path, roots: tuple[str, ...]) -> None:
        try:
            self._queue.put(("ok", scan_dependencies(path, stock_roots=roots)))
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
            write_csv_report(self._result, selected)
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
            write_json_report(self._result, selected)
            self.status_var.set(f"Saved {Path(selected).name}")

    def run(self) -> int:
        self.root.mainloop()
        return 0


def gui_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", nargs="?", type=Path)
    args, _unknown = parser.parse_known_args(argv)
    return ScannerGui(args.input).run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cwr-wrp-mod-scan",
        description="List mod P3D dependencies referenced by a CWA/OFP WRP or PBO.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="WRP file or PBO containing WRP files")
    parser.add_argument(
        "--stock-root",
        action="append",
        default=None,
        help="Namespace treated as stock; repeat as needed (default: data3d, o)",
    )
    parser.add_argument("--include-stock", action="store_true", help="Show stock P3D references too")
    parser.add_argument("--json", type=Path, help="Write a JSON report")
    parser.add_argument("--csv", type=Path, help="Write a CSV report")
    parser.add_argument("--gui", action="store_true", help="Open the graphical scanner")
    args = parser.parse_args(argv)

    if args.gui or args.input is None:
        return ScannerGui(args.input).run()

    result = scan_dependencies(args.input, stock_roots=normalize_stock_roots(args.stock_root))
    if args.json:
        write_json_report(result, args.json)
    if args.csv:
        write_csv_report(result, args.csv)
    print(_format_text(result, include_stock=args.include_stock))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
