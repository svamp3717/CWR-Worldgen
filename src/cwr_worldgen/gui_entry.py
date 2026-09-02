# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import multiprocessing as _multiprocessing
import sys

# PyInstaller re-executes the frozen application to start multiprocessing workers.
# Dispatch those worker command lines before the GUI can be imported, otherwise
# each procedural-asset worker can start another application window. Keep normal
# source runs untouched; PyInstaller sets sys.frozen before user code starts.
if bool(getattr(sys, "frozen", False)):
    _multiprocessing.freeze_support()

import argparse
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable

FROZEN_CLI_MARKER = "--cwr-cli"
OVERTURE_CLI_MARKER = "--cwr-overture"
ROAD_INSPECTOR_CLI_MARKER = "--cwr-road-inspector"
ROAD_INSPECTOR_CHILD_CODE = (
    "import cwr_worldgen.gui_entry as g,sys;"
    "raise SystemExit(g._run_road_inspector_postbuild(sys.argv[1:]))"
)
CONSOLE_LOG_FILENAME = "cwr-worldgen-console.log"
WORLD_NAME_PREFIX = "wg_"
LEGACY_WORLD_NAME_PREFIXES = ("cwa_", "cwr_")


def generated_world_identifier(value: str) -> str:
    """Return the auto-managed world identifier using the current ``wg_`` prefix."""
    slug = str(value).strip().casefold()
    for prefix in (WORLD_NAME_PREFIX, *LEGACY_WORLD_NAME_PREFIXES):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
            break
    slug = slug or "my_world"
    return WORLD_NAME_PREFIX + slug


def storage_base_dir() -> Path:
    """Return the folder that owns build, source-data, and config."""
    if not bool(getattr(sys, "frozen", False)):
        return Path.cwd().resolve()
    executable = Path(sys.executable).resolve()
    if (
        sys.platform == "darwin"
        and executable.parent.name == "MacOS"
        and executable.parent.parent.name == "Contents"
    ):
        return executable.parents[3]
    return executable.parent


def managed_replacement(
    current: str,
    managed: str | None,
    replacement: str,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> tuple[str, str | None]:
    """Replace an auto-managed value, but stop tracking a user-edited value."""
    if managed is None:
        return current, None
    normalize = normalizer or (lambda value: value)
    if normalize(current) != normalize(managed):
        return current, None
    return replacement, replacement


def console_log_paths(
    source_dir: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, ...]:
    """Return de-duplicated console-log targets for source and build folders."""
    targets: list[Path] = []
    seen: set[str] = set()
    for raw_root in (source_dir, output_dir):
        if raw_root is None or not str(raw_root).strip():
            continue
        target = Path(raw_root).expanduser() / CONSOLE_LOG_FILENAME
        key = os.path.normcase(os.path.abspath(str(target)))
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return tuple(targets)


def mirror_console_log_fragment(
    targets: Iterable[Path],
    fragment: str,
    transcript: str,
    initialized: set[str],
) -> None:
    """Mirror a GUI log fragment, restoring the full transcript after folder cleanup."""
    for raw_target in targets:
        target = Path(raw_target)
        key = os.path.normcase(os.path.abspath(str(target)))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            reset = key not in initialized or not target.is_file()
            mode = "w" if reset else "a"
            payload = transcript if reset else fragment
            with target.open(mode, encoding="utf-8", newline="") as stream:
                stream.write(payload)
            initialized.add(key)
        except OSError:
            # A diagnostic mirror must never be allowed to break the build itself.
            continue


def generated_mod_folder(output_dir: Path) -> Path | None:
    """Find the generated runtime folder that directly contains Addons and Anims."""
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return None
    if (root / "Addons").is_dir() and (root / "Anims").is_dir():
        return root.resolve()
    try:
        candidates = [
            child.resolve()
            for child in root.iterdir()
            if child.is_dir()
            and (child / "Addons").is_dir()
            and (child / "Anims").is_dir()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.casefold() == "cwr-worldgen":
            return candidate
    candidates.sort(key=lambda path: path.name.casefold())
    return candidates[0]


def generated_world_pbo(build_dir: str | Path, world_name: str) -> Path | None:
    """Return the generated world PBO without depending on generator internals."""
    root = Path(build_dir).expanduser()
    name = str(world_name).strip()
    if not name:
        return None
    runtime = generated_mod_folder(root)
    if runtime is not None:
        candidate = runtime / "Addons" / f"{name}.pbo"
        if candidate.is_file():
            return candidate.resolve()
    for candidate in (root / f"{name}.pbo", root / "Addons" / f"{name}.pbo"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _write_road_inspector_map_report(result: Any, path: Path) -> None:
    """Write the compact interactive map used by GUI post-build inspection."""
    roads = []
    for road in result.road_objects:
        if not road.endpoints:
            continue
        if road.kind.startswith("junction_"):
            segments = [
                [road.x, road.z, endpoint.point[0], endpoint.point[1]]
                for endpoint in road.endpoints
            ]
        else:
            first, last = road.endpoints[0], road.endpoints[-1]
            segments = [[first.point[0], first.point[1], last.point[0], last.point[1]]]
        roads.append({
            "id": road.object_id,
            "kind": road.kind,
            "family": road.family,
            "segments": segments,
        })
    issues = [
        {
            "id": issue.issue_id,
            "severity": issue.severity,
            "score": issue.score,
            "category": issue.category,
            "x": issue.x,
            "z": issue.z,
            "objects": list(issue.object_ids),
            "message": issue.message,
        }
        for issue in result.issues
    ]
    summary = {
        "roads": result.road_object_count,
        "issues": len(result.issues),
    }

    def embedded_json(value: object) -> str:
        return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")

    title = html.escape(f"Road Inspector - {Path(result.input_path).name}")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{font-family:system-ui,Segoe UI,sans-serif;color-scheme:dark;background:#111;color:#eee}}
body{{margin:0;display:grid;grid-template-columns:minmax(360px,42vw) 1fr;height:100vh;overflow:hidden}}
#panel{{overflow:auto;padding:16px;border-right:1px solid #444;background:#171717}}
#mapwrap{{position:relative;min-width:0;background:#0c0f0c}}#map{{width:100%;height:100%;display:block}}
h1{{font-size:20px;margin:0 0 6px}}.muted{{color:#aaa}}.stats{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}}
.stat{{padding:6px 9px;border:1px solid #444;border-radius:6px;background:#202020}}
.controls{{position:sticky;top:-16px;z-index:3;padding:10px 0;background:#171717;display:flex;gap:8px;flex-wrap:wrap}}
select,button{{background:#222;color:#eee;border:1px solid #555;border-radius:5px;padding:6px 8px}}
.issue{{border:1px solid #3c3c3c;border-left-width:5px;border-radius:6px;margin:8px 0;padding:9px;cursor:pointer;background:#1d1d1d}}
.issue:hover,.issue.active{{background:#292929}}.issue.critical{{border-left-color:#ff4242}}.issue.high{{border-left-color:#ff9d32}}
.issue.medium{{border-left-color:#f1d54a}}.issue.low{{border-left-color:#6ca7ff}}.issuehead{{display:flex;justify-content:space-between;gap:8px;font-weight:700}}
.category{{font-family:ui-monospace,Consolas,monospace;color:#b8d6ff}}.coords{{font-family:ui-monospace,Consolas,monospace;color:#ccc}}
.issueactions{{display:flex;gap:8px;margin-top:8px}}.copycoords{{cursor:pointer}}
.road{{stroke:#777;stroke-width:1.2;vector-effect:non-scaling-stroke;opacity:.55}}.road.curve{{stroke:#aaa}}.road.junction{{stroke:#ddd;stroke-width:2}}
.road.selected{{stroke:#63d6ff;stroke-width:4;opacity:1}}.marker{{vector-effect:non-scaling-stroke;stroke:#111;stroke-width:1.2;cursor:pointer}}
.marker.critical{{fill:#ff4242}}.marker.high{{fill:#ff9d32}}.marker.medium{{fill:#f1d54a}}.marker.low{{fill:#6ca7ff}}
#mapinfo{{position:absolute;top:10px;left:10px;padding:8px 10px;border-radius:6px;background:#111d;border:1px solid #555;font-family:ui-monospace,Consolas,monospace;pointer-events:none}}
@media(max-width:900px){{body{{grid-template-columns:1fr;grid-template-rows:55vh 45vh}}#panel{{border-right:0;border-bottom:1px solid #444}}}}
</style></head><body>
<section id="panel"><h1>{title}</h1><div class="muted">Read-only audit of the actual emitted RVW4 stock-road geometry.</div>
<div class="stats" id="stats"></div><div class="controls">
<select id="severity"><option value="30" selected>medium +</option><option value="55">high +</option><option value="80">critical only</option><option value="0">all findings</option></select>
<select id="category"><option value="">all categories</option></select><button id="reset">Reset map</button></div><div id="issues"></div></section>
<section id="mapwrap"><svg id="map"></svg><div id="mapinfo">Click a finding to zoom to ±35 m</div></section>
<script>
const summary={embedded_json(summary)},issues={embedded_json(issues)},roads={embedded_json(roads)};
const svg=document.getElementById('map'),list=document.getElementById('issues'),ns='http://www.w3.org/2000/svg';
let minx=Infinity,maxx=-Infinity,minz=Infinity,maxz=-Infinity;
function addPoint(x,z){{minx=Math.min(minx,x);maxx=Math.max(maxx,x);minz=Math.min(minz,z);maxz=Math.max(maxz,z)}}
for(const r of roads)for(const s of r.segments){{addPoint(s[0],s[1]);addPoint(s[2],s[3])}}for(const i of issues)addPoint(i.x,i.z);
if(!Number.isFinite(minx)){{minx=0;maxx=1;minz=0;maxz=1}}
function view(x0,z0,x1,z1){{svg.setAttribute('viewBox',`${{x0}} ${{-z1}} ${{Math.max(1,x1-x0)}} ${{Math.max(1,z1-z0)}}`)}}
function resetView(){{view(minx-10,minz-10,maxx+10,maxz+10)}}resetView();
const roadEls=new Map(),markerEls=new Map();
for(const r of roads)for(const s of r.segments){{const line=document.createElementNS(ns,'line');line.setAttribute('x1',s[0]);line.setAttribute('y1',-s[1]);line.setAttribute('x2',s[2]);line.setAttribute('y2',-s[3]);line.classList.add('road');if(r.kind==='curve')line.classList.add('curve');if(r.kind.startsWith('junction_'))line.classList.add('junction');svg.appendChild(line);if(!roadEls.has(r.id))roadEls.set(r.id,[]);roadEls.get(r.id).push(line)}}
for(const i of issues){{const c=document.createElementNS(ns,'circle');c.setAttribute('cx',i.x);c.setAttribute('cy',-i.z);c.setAttribute('r',2.2);c.classList.add('marker',i.severity);c.onclick=()=>selectIssue(i);svg.appendChild(c);markerEls.set(i.id,c)}}
function esc(s){{return String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]))}}
function visible(i){{const score=Number(document.getElementById('severity').value),cat=document.getElementById('category').value;return i.score>=score&&(!cat||i.category===cat)}}
function consoleCommand(i){{return `player setPos [${{i.x.toFixed(3)}}, ${{i.z.toFixed(3)}}, 0]`}}
async function copyText(text){{
  if(navigator.clipboard&&window.isSecureContext){{try{{await navigator.clipboard.writeText(text);return true}}catch(_error){{}}}}
  const area=document.createElement('textarea');area.value=text;area.setAttribute('readonly','');area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.focus();area.select();
  let copied=false;try{{copied=document.execCommand('copy')}}catch(_error){{}}area.remove();return copied;
}}
function selectIssue(i){{document.querySelectorAll('.road.selected').forEach(e=>e.classList.remove('selected'));for(const id of i.objects)for(const e of roadEls.get(id)||[])e.classList.add('selected');view(i.x-35,i.z-35,i.x+35,i.z+35);document.getElementById('mapinfo').textContent=`${{i.id}}  ${{i.x.toFixed(2)}}, ${{i.z.toFixed(2)}}  ${{i.category}}`;document.querySelectorAll('.issue.active').forEach(e=>e.classList.remove('active'));const row=document.querySelector(`[data-row="${{i.id}}"]`);if(row){{row.classList.add('active');row.scrollIntoView({{block:'nearest'}})}}}}
function render(){{list.innerHTML='';let shown=0;for(const i of issues){{const show=visible(i),marker=markerEls.get(i.id);if(marker)marker.style.display=show?'':'none';if(!show)continue;shown++;const d=document.createElement('div'),command=consoleCommand(i);d.className=`issue ${{i.severity}}`;d.dataset.row=i.id;d.innerHTML=`<div class="issuehead"><span>${{esc(i.id)}} · ${{esc(i.severity.toUpperCase())}}</span><span>${{i.score.toFixed(1)}}</span></div><div class="category">${{esc(i.category)}}</div><div class="coords">${{i.x.toFixed(2)}}, ${{i.z.toFixed(2)}} · objects ${{esc(i.objects.join(', '))}}</div><div>${{esc(i.message)}}</div><div class="coords">${{esc(command)}}</div><div class="issueactions"><button type="button" class="copycoords">Copy coords</button></div>`;d.onclick=()=>selectIssue(i);const copy=d.querySelector('.copycoords');copy.onclick=async event=>{{event.stopPropagation();const copied=await copyText(command);copy.textContent=copied?'Copied':'Copy failed';window.setTimeout(()=>{{copy.textContent='Copy coords'}},1200)}};list.appendChild(d)}}document.getElementById('stats').innerHTML=`<span class="stat">${{summary.roads}} road objects</span><span class="stat">${{summary.issues}} findings</span><span class="stat">${{shown}} shown</span>`}}
const cats=[...new Set(issues.map(i=>i.category))].sort(),catSel=document.getElementById('category');for(const c of cats){{const o=document.createElement('option');o.textContent=c;o.value=c;catSel.appendChild(o)}}
document.getElementById('severity').onchange=render;catSel.onchange=render;document.getElementById('reset').onclick=()=>{{resetView();document.querySelectorAll('.road.selected').forEach(e=>e.classList.remove('selected'));document.getElementById('mapinfo').textContent='Click a finding to zoom to ±35 m'}};render();
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def road_inspector_postbuild_command(
    build_dir: str | Path,
    world_name: str,
    *,
    frozen: bool | None = None,
    executable: str | None = None,
) -> list[str]:
    """Return the optional post-build inspector child-process command."""
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    launcher = executable or sys.executable
    command = [launcher, ROAD_INSPECTOR_CLI_MARKER] if frozen else [
        launcher,
        "-c",
        ROAD_INSPECTOR_CHILD_CODE,
    ]
    command.extend(("--build-dir", str(build_dir), "--world-name", str(world_name)))
    return command


def _run_road_inspector_postbuild(args: list[str]) -> int:
    """Run the read-only inspector after generation; diagnostics never fail a build."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--world-name", required=True)
    options = parser.parse_args(args)
    build_dir = options.build_dir.expanduser().resolve()
    report_dir = build_dir / "road-inspector"
    try:
        if report_dir.exists():
            shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        pbo = generated_world_pbo(build_dir, options.world_name)
        if pbo is None:
            raise FileNotFoundError(f"generated PBO for {options.world_name!r} was not found")
        from .road_inspector import inspect_road_geometry, write_inspection_report

        result = inspect_road_geometry(pbo)
        paths = write_inspection_report(result, report_dir)
        _write_road_inspector_map_report(result, paths["html"])
        print(
            f"Road Inspector: {result.road_object_count:,} road objects, "
            f"{len(result.issues):,} issues."
        )
        print(f"Road Inspector report: {paths['html']}")
    except Exception as exc:
        message = f"Road Inspector warning: {type(exc).__name__}: {exc}"
        print(message)
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "error.txt").write_text(message + "\n", encoding="utf-8")
        except OSError:
            pass
    return 0


def _install_frozen_dem_cache(base_dir: Path) -> None:
    """Make dem-stitcher localize remote DEM tiles before merging them."""
    import dem_stitcher

    original_stitch_dem = dem_stitcher.stitch_dem
    if bool(getattr(original_stitch_dem, "_cwr_local_cache", False)):
        return

    def stitch_dem_with_local_cache(*args: Any, **kwargs: Any):
        if kwargs.get("dst_tile_dir") is None:
            dem_name = kwargs.get("dem_name")
            if dem_name is None and len(args) > 1:
                dem_name = args[1]
            cache_dir = base_dir / "source-data" / ".dem-stitcher-cache" / str(dem_name or "dem")
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["dst_tile_dir"] = cache_dir
        return original_stitch_dem(*args, **kwargs)

    stitch_dem_with_local_cache._cwr_local_cache = True  # type: ignore[attr-defined]
    dem_stitcher.stitch_dem = stitch_dem_with_local_cache


def _ensure_cli_streams() -> None:
    """Provide harmless stdio streams in PyInstaller windowed processes."""
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _run_bundled_overture_cli(args: list[str]) -> int:
    """Run CWR's isolated Overture Python-API worker without the upstream CLI."""
    _ensure_cli_streams()
    from .overture import run_overture_worker

    try:
        return run_overture_worker(args)
    except Exception as exc:
        print(f"CWR Worldgen Overture worker failed: {exc}", file=sys.stderr)
        return 1


def _configure_gui(gui: Any, base_dir: Path) -> None:
    """Apply stable storage and world-name synchronization to the GUI module."""
    gui.application_base_dir = lambda: base_dir

    original_default_gui_values = gui.default_gui_values
    original_defaults_with_recent_source = gui.defaults_with_recent_source

    def default_gui_values() -> dict[str, object]:
        values = original_default_gui_values()
        values["name"] = generated_world_identifier("my_world")
        values["run_road_inspector_after_build"] = False
        return values

    def defaults_with_recent_source(
        defaults: dict[str, object], state: dict[str, object]
    ) -> dict[str, object]:
        values = original_defaults_with_recent_source(defaults, state)
        name = str(values.get("name", "")).strip()
        folded = name.casefold()
        if any(folded.startswith(prefix) for prefix in LEGACY_WORLD_NAME_PREFIXES):
            values["name"] = generated_world_identifier(name)
            if not str(state.get("last_world_output", "")).strip():
                output = Path(str(values.get("output", "")))
                values["output"] = str(output.parent / str(values["name"])[len(WORLD_NAME_PREFIX):])
        return values

    def suggested_world_values(
        display_name: str,
        *,
        source_mode: str,
        source_dir: str,
    ) -> dict[str, str]:
        slug = gui.slugify_world_name(display_name)
        values = {
            "name": generated_world_identifier(slug),
            "output": gui.default_gui_path(Path("build") / slug),
            "source_dir": source_dir,
        }
        if source_mode == "new":
            values["source_dir"] = gui.default_gui_path(Path("source-data") / slug)
        return values

    gui.default_gui_values = default_gui_values
    gui.defaults_with_recent_source = defaults_with_recent_source
    gui.suggested_world_values = suggested_world_values

    def normalize_path(value: str) -> str:
        if not value:
            return ""
        return os.path.normcase(str(gui.resolve_gui_path(value)))

    def generated_values(display_name: str) -> dict[str, str]:
        slug = gui.slugify_world_name(display_name)
        return {
            "name": generated_world_identifier(slug),
            "output": gui.default_gui_path(Path("build") / slug),
            "source_dir": gui.default_gui_path(Path("source-data") / slug),
        }

    original_class = gui.WorldgenGui

    class SyncedWorldgenGui(original_class):
        """Keep generated paths aligned, mirror logs, and expose finished diagnostics."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._auto_world_guard = True
            self._auto_display_name = ""
            self._console_log_text = ""
            self._console_log_initialized: set[str] = set()
            self._managed_world_values: dict[str, str | None] = {
                "name": None,
                "output": None,
                "source_dir": None,
            }
            super().__init__(*args, **kwargs)
            self._clear_remembered_deploy_default()
            self._auto_display_name = str(self.vars["display_name"].get())
            self._arm_auto_world_values()
            self._auto_world_guard = False
            self.open_generated_mod_button = gui.ttk.Button(
                self.page_frames[gui.PROGRESS_STEP_INDEX],
                text="Open generated files folder",
                command=self._open_generated_mod_folder,
            )
            self._update_navigation()

        def _build_world_page(self) -> None:
            super()._build_world_page()
            page = self.page_frames[-1]
            children = page.winfo_children()
            body = getattr(children[0], "body", None) if children else None
            if body is None:
                return
            checks = gui.ttk.LabelFrame(
                body,
                text="Post-build checks",
                style="Section.TLabelframe",
                padding=12,
            )
            checks.pack(fill="x", pady=(12, 0))
            gui.ttk.Checkbutton(
                checks,
                text="Run Road Inspector after a successful build",
                variable=self._var("run_road_inspector_after_build", False, boolean=True),
            ).pack(anchor="w")
            gui.ttk.Label(
                checks,
                text=(
                    "Writes the read-only road report to <world build folder>/road-inspector/. "
                    "Inspector errors are warnings and do not fail the completed world build."
                ),
                style="Hint.TLabel",
                wraplength=700,
            ).pack(anchor="w", pady=(5, 0))

        def _clear_remembered_deploy_default(self) -> None:
            """Start deployment disabled and migrate away any remembered machine path."""
            self.vars["deploy_to_mod_folder"].set(False)
            self.vars["deploy_mod_dir"].set("")
            try:
                state = gui.load_gui_state(self.state_path)
                had_deploy_path = bool(str(state.get("last_deploy_mod_dir", "")).strip())
                deploy_was_enabled = bool(state.get("deploy_to_mod_folder", False))
                if had_deploy_path or deploy_was_enabled:
                    state.pop("state_version", None)
                    state.pop("last_deploy_mod_dir", None)
                    state["deploy_to_mod_folder"] = False
                    gui.save_gui_state(self.state_path, state)
            except OSError:
                pass

        def _console_log_targets(self) -> tuple[Path, ...]:
            vars_map = getattr(self, "vars", {})
            source_var = vars_map.get("source_dir")
            output_var = vars_map.get("output")
            source_text = str(source_var.get()).strip() if source_var is not None else ""
            output_text = str(output_var.get()).strip() if output_var is not None else ""
            source_dir = gui.resolve_gui_path(source_text) if source_text else None
            output_dir = gui.resolve_gui_path(output_text) if output_text else None
            return console_log_paths(source_dir, output_dir)

        def _start_pipeline(
            self,
            jobs: list[tuple[list[str], str]],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            if self.process is None and not self._pipeline_active:
                self._console_log_text = ""
                self._console_log_initialized.clear()
            inspector_var = self.vars.get("run_road_inspector_after_build")
            if (
                kwargs.get("kind") == "build"
                and inspector_var is not None
                and bool(inspector_var.get())
            ):
                jobs = list(jobs)
                already_added = any(
                    ROAD_INSPECTOR_CLI_MARKER in command
                    or ROAD_INSPECTOR_CHILD_CODE in command
                    for command, _description in jobs
                )
                if not already_added:
                    output_text = str(self.vars["output"].get()).strip()
                    world_name = str(self.vars["name"].get()).strip()
                    if output_text and world_name:
                        jobs.append(
                            (
                                road_inspector_postbuild_command(
                                    gui.resolve_gui_path(output_text),
                                    world_name,
                                ),
                                "Running Road Inspector",
                            )
                        )
            super()._start_pipeline(jobs, *args, **kwargs)

        def _append_log(self, text: str) -> None:
            super()._append_log(text)
            self._console_log_text += text
            mirror_console_log_fragment(
                self._console_log_targets(),
                text,
                self._console_log_text,
                self._console_log_initialized,
            )

        def _generated_mod_folder(self) -> Path | None:
            output_text = str(self.vars["output"].get()).strip()
            if not output_text:
                return None
            return generated_mod_folder(gui.resolve_gui_path(output_text))

        def _open_generated_mod_folder(self) -> None:
            runtime = self._generated_mod_folder()
            if runtime is None:
                gui.messagebox.showinfo(
                    gui.APP_TITLE,
                    "No completed generated folder containing both Addons and Anims was found.",
                )
                return
            self._open_path(str(runtime))

        def _update_navigation(self) -> None:
            super()._update_navigation()
            button = getattr(self, "open_generated_mod_button", None)
            if button is None:
                return
            show_button = (
                self._operation_success
                and self._pipeline_kind == "build"
                and self.process is None
                and not self._pipeline_active
                and self._generated_mod_folder() is not None
            )
            if show_button:
                if not button.winfo_manager():
                    button.pack(anchor="e", padx=4, pady=(8, 4))
            elif button.winfo_manager():
                button.pack_forget()

        def _arm_auto_world_values(self) -> None:
            display_name = str(self.vars["display_name"].get())
            generated = generated_values(display_name)
            defaults = gui.default_gui_values()

            current_name = str(self.vars["name"].get()).strip()
            if current_name in {str(defaults["name"]), generated["name"]}:
                self._managed_world_values["name"] = current_name
            else:
                self._managed_world_values["name"] = None

            current_output = str(self.vars["output"].get()).strip()
            if normalize_path(current_output) in {
                normalize_path(str(defaults["output"])),
                normalize_path(generated["output"]),
            }:
                self._managed_world_values["output"] = current_output
            else:
                self._managed_world_values["output"] = None

            current_source = str(self.vars["source_dir"].get()).strip()
            if str(self.vars["source_mode"].get()) == "new" and normalize_path(current_source) in {
                normalize_path(str(defaults["source_dir"])),
                normalize_path(generated["source_dir"]),
            }:
                self._managed_world_values["source_dir"] = current_source
            else:
                self._managed_world_values["source_dir"] = None

        def _sync_auto_world_values(self) -> None:
            if self._auto_world_guard or "display_name" not in self.vars:
                return
            display_name = str(self.vars["display_name"].get())
            if display_name == self._auto_display_name:
                return

            generated = generated_values(display_name)
            current_name = str(self.vars["name"].get()).strip()
            new_name, managed_name = managed_replacement(
                current_name,
                self._managed_world_values["name"],
                generated["name"],
            )
            self._managed_world_values["name"] = managed_name
            if new_name != current_name:
                self.vars["name"].set(new_name)

            current_output = str(self.vars["output"].get()).strip()
            new_output, managed_output = managed_replacement(
                current_output,
                self._managed_world_values["output"],
                generated["output"],
                normalizer=normalize_path,
            )
            self._managed_world_values["output"] = managed_output
            if normalize_path(new_output) != normalize_path(current_output):
                self.vars["output"].set(new_output)

            if str(self.vars["source_mode"].get()) == "new":
                current_source = str(self.vars["source_dir"].get()).strip()
                new_source, managed_source = managed_replacement(
                    current_source,
                    self._managed_world_values["source_dir"],
                    generated["source_dir"],
                    normalizer=normalize_path,
                )
                self._managed_world_values["source_dir"] = managed_source
                if normalize_path(new_source) != normalize_path(current_source):
                    self.vars["source_dir"].set(new_source)
                    self._sync_source_paths()
            else:
                self._managed_world_values["source_dir"] = None

            self._auto_display_name = display_name

        def _refresh_views(self) -> None:
            self._sync_auto_world_values()
            super()._refresh_views()

        def _suggest_names(self) -> None:
            super()._suggest_names()
            self._auto_display_name = str(self.vars["display_name"].get())
            generated = generated_values(self._auto_display_name)
            self._managed_world_values["name"] = str(self.vars["name"].get()).strip()
            self._managed_world_values["output"] = str(self.vars["output"].get()).strip()
            self._managed_world_values["source_dir"] = (
                str(self.vars["source_dir"].get()).strip()
                if str(self.vars["source_mode"].get()) == "new"
                else None
            )
            # Keep this reference useful even if a future GUI revision changes
            # how the suggestion button derives its values.
            _ = generated

    gui.WorldgenGui = SyncedWorldgenGui


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    base_dir = storage_base_dir()
    os.environ.setdefault("CWR_WORLDGEN_GUI_STATE", str(base_dir / "config" / "gui-state.json"))
    os.environ.setdefault("CWR_WORLDGEN_RUNTIME_DIR", "CWR-Worldgen")

    if args and args[0] == ROAD_INSPECTOR_CLI_MARKER:
        return _run_road_inspector_postbuild(args[1:])
    if args and args[0] == OVERTURE_CLI_MARKER:
        return _run_bundled_overture_cli(args[1:])

    frozen = bool(getattr(sys, "frozen", False))
    if frozen and args and args[0] == FROZEN_CLI_MARKER:
        _install_frozen_dem_cache(base_dir)

    from . import gui

    _configure_gui(gui, base_dir)
    return gui.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
