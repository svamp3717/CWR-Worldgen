# Standalone Windows asset tools

The [Build Windows Asset Tools](../.github/workflows/build-windows-tools.yml) GitHub
Actions workflow packages these utilities independently of the main
CWR-Worldgen executable:

| Download artifact | Windows executable | Purpose |
| --- | --- | --- |
| `Categorise-P3D-Models-Windows` | `Categorise-P3D-Models.exe` | Review and categorise P3D models, saving a JSON catalogue |
| `Browse-P3D-Assets-Windows` | `Browse-P3D-Assets.exe` | Browse P3D models and PAA/PAC textures |
| `WRP-Mod-Dependency-Scanner-Windows` | `WRP-Mod-Dependency-Scanner.exe` | Scan WRPs, PBOs or IFAs for model dependencies |

## Downloading a build

Both Windows tool workflows are **manual-only**. Neither starts on commits,
pull requests, or tags. Once merged into `main`, choose one:

- **Actions → Build Windows Asset Tools → Run workflow:** build all three
  executables and download them from that run's **Artifacts** section.
  Artifacts are retained for 14 days.
- **Actions → Build and Publish Windows Asset Tools → Run workflow:** build all
  three tools and publish them as downloads on a separate GitHub Release,
  with a `tools-v<project version>` tag by default. An optional
  `release_tag` input (for example `tools-v0.9.262-patch1`) lets you
  choose a unique tag. Publishing requires `main`, and refuses to reuse an
  existing tag or overwrite an existing release.

Publishing the tools will **not** update the separate main CWR Worldgen release
or mark the tools release as the repository's latest release. No build or
release action is started automatically.

The programs package their Python runtime and dependencies. The input game
files are **not** bundled; select your own P3D/WRP/PBO/IFA files.

All three executables show a branded loading screen **during PyInstaller
one-file extraction** and while their startup Python imports finish. The
screen closes when the GUI is ready (or before a command-line WRP scan starts).
This gives visible feedback without introducing an artificial delay. It does
not make the one-file extraction faster. The splash is an experimental
Windows-supported PyInstaller feature and needs a Windows desktop to display.
On the WRP splash, only **WRP** is white; **MOD DEPENDENCY SCANNER** and the
other on-screen labels use the amber accent.

The categoriser and asset browser open directly as GUI applications. The WRP
scanner opens its GUI when run without arguments but keeps a console window so
it can also be used with CLI options such as:

```powershell
.\WRP-Mod-Dependency-Scanner.exe "C:\path\to\island.wrp" --json "dependencies.json"
```

### DayZ Standalone support (asset tools only)

All three standalone utilities can now be used with DayZ's PBO/P3D/WRP
assets. **No changes are made to the CWR-Worldgen terrain generator.**

- **Browse P3D Assets / Model Categoriser:** select DayZ model PBOs or loose
  P3Ds as usual. Models from virtual `DZ\\...` paths are identified as DayZ
  assets, and the existing modern ODOL reader handles common DayZ ODOL
  versions (v53/v54/v55). If an individual model's mesh cannot be decoded,
  its bounding-box preview is shown rather than pretending the box is a mesh.
- **Texture lookup:** a model referencing
  `dz\\structures\\data\\wall_co.paa` now prefers the sibling
  `structures.pbo` over a generic `dz.pbo`, honoring PBO prefix properties.
  Additional texture archives remain selectable manually.
- **WRP Mod Dependency Scanner:** select **DayZ Standalone** in the Game
  dropdown, or use `--game dayz` in the CLI. `DZ` is then treated as a
  stock asset root. Unbinarized DayZ Terrain Builder `0WZD` WRPs are read
  from their object records, so repeated P3Ds have real object counts.
  Binarized `OPRW` WRPs use a conservative embedded-string scan instead.
  Those counts are **string occurrences**, not object-placement counts, and
  not all object dependencies are guaranteed discoverable.

Example:

```powershell
.\\WRP-Mod-Dependency-Scanner.exe "C:\\DayZMaps\\world.wrp" --game dayz --json "dayz-dependencies.json"
```

This is **read-only inspection and categorisation**, not DayZ terrain export
or ODOL conversion. The automated tests use synthetic DayZ-compatible
format fixtures; actual game archives and Windows executables still need
smoke-testing against a legitimate DayZ installation.

### Choosing individual texture archives

Both P3D tools can load **multiple individual texture PBO/IFA archives**
(including `.pbo.zst` and `.ifa.zst`) without selecting or scanning an entire
texture folder:

- **Model & Texture Browser:** first choose the model PBO/IFA, then click
  **Add texture PBOs / IFAs…** and multi-select the archives containing its
  PAA/PAC files. You may add further archives with the same button later.
  **Clear texture PBOs** removes only the extra texture lookup sources, not
  the model catalogue. Replacing the model PBO keeps your extra texture choices.
- **Model Categoriser:** select the model archives in the startup dialog,
  then add individual **texture PBOs / IFAs** in the separate list before
  starting. You may also add archives from the main categoriser window or
  its View menu. Texture-only archives are not included in the model count
  or classification stream. The CLI continues to support repeated
  `--texture-source C:\\path\\to\\texture.pbo` arguments.

For Windows multi-selection, use Ctrl-click or Shift-click in the archive
picker. The original folder selection remains available when desired.

The **Related assets** panel in the Model & Texture Browser distinguishes
textures in the model catalogue (**Texture**) from textures located by the
preview resolver in another PBO/IFA (**External**). **Missing** means no
texture asset was located in the currently available sources. **Unreadable**
means an archive entry exists but its texture failed to decode during preview,
and **Unchecked** means texture loading was skipped. Extra texture archives
are deliberately not added to the model catalogue; the external status is
updated when you add or clear them.

### Copyable Selected asset details in the explorer

The browser's **Selected asset** panel is a read-only, scrollable text view.
Drag to highlight any model path, source archive, dimensions, texture
information, or load-time values and press **Ctrl+C** to copy only that
selection. **Ctrl+A** selects the entire readout. The existing **Copy asset
path** button remains available for quickly copying just the in-game asset
path. Text selection never changes a model, texture, or saved file.

### Faster CPU-rendered 3D previews

Both the **Model & Texture Browser** (in the Selected asset panel) and the
**Model Categoriser** (in the View sidebar, below Texture mipmap) now offer
a **Render quality** selector. This is independent from **Preview LOD**,
texture mipmap choice, and the Skip textures option:

- **Fast (60%)** (default in both tools): Rasterize at 60% of the adaptive preview
  width and height (about 36% of the pixel workload), then upscale smoothly.
- **Balanced (80%)**: Approximately 64% of the full pixel workload.
- **Quality (100%)**: Retain the previous full preview resolution and sampling
  behavior.

Changing render quality redraws the same loaded model without touching the
archive, model order, classification state, or chosen model/texture LOD. The
categoriser retains all selected categories, placement, current model, and
session progress; the startup render quality is Fast in both applications.

The shared CPU rasterizer now computes coverage using broadcast coordinate
vectors, tests depth only for triangle-covered pixels, and performs expensive
UV interpolation and bilinear/trilinear sampling **only on visible fragments**.
Transparent pixels keep their existing alpha-test behavior; the result still
uses perspective-correct UVs, repeating texture coordinates, and the same
sampling filters. The categoriser also benefits from the shared rasterizer
and from its own selectable, reduced render resolution.

The browser and categoriser model timing readouts both separate **Textures**
(lookup/decode/pyramid), **CPU raster** (remaining CPU geometry/pixel work),
and **Matplotlib prep** (synchronous plotting/figure setup). These sum with
**P3D read** and **Mesh** to the reported **Total**. For bounds-only and
vertex-only previews that skip CPU rasterization, the categoriser instead
shows the combined **Preview** time. Matplotlib's deferred UI painting remains
outside that total. Lower render resolution reduces CPU pixel workload, but
neither the selected model LOD nor embedded texture mip changes automatically.

### Explorer raw-model cache and load-time diagnostics

The **P3D Model & Texture Browser** now has the same model/texture timing
visibility as the categoriser, shown in its **Selected asset** panel:

- **P3D read**, **Mesh**, **Textures**, **Preview**, and **Total** for 3D
  model previews, including whether a LOD came from the decoded-mesh cache
  or the raw **P3D RAM cache**.
- **Texture read/decode**, **Preview/details**, and **Total** for standalone
  PAA/PAC texture previews. Re-selecting an already decoded texture/mipmap
  identifies the **RGBA cache** hit.

The browser retains its existing 96 MiB (by default) decoded-mesh cache
and adds a separate **64 MiB / four-model** raw P3D byte LRU. On first load
the browser reads the archive only once and keeps recent P3D bytes inside
that bounded cache. Selecting a previously unseen LOD can decode it from
RAM rather than reopening the original PBO/IFA (including nested archives).
Already-decoded LODs still use the faster existing mesh cache. Changing or
clearing model sources invalidates **both** caches, while changing texture-only
sources preserves the models but invalidates texture decodes and mip pyramids.

Preview numbers are measured around synchronous parsing and drawing; deferred
Matplotlib painting and related-asset indexing are outside the per-model total.
Read/mesh numbers describe the most recent model selection, whereas texture and
preview numbers are refreshed whenever the preview redraws.

### Categoriser load-time diagnostics and faster LOD switching

The **Model Categoriser** now displays a per-preview wall-clock breakdown in
the **View** panel, directly below its LOD and texture mipmap settings:

- **P3D read/scan:** Time to obtain the next model from the input archive
  stream. For the first model, this can include archive header enumeration;
  after a LOD change, a successful **RAM cache** hit reports zero reread time.
- **Mesh:** Time spent indexing/decoding the P3D and preparing preview
  geometry, separate from archive I/O.
- **Textures:** Time spent resolving/loading textures, decoding the selected
  embedded PAA/PAC mipmap, and preparing filtered texture levels. Previously
  cached textures usually make this step much shorter.
- **CPU raster:** Synchronous renderer time excluding the **Textures** stage.
- **Matplotlib prep:** Time to prepare the preview plot and schedule a redraw;
  Tk/Matplotlib deferred canvas painting is not included. For bounds-only
  and vertex-only fallbacks, **Preview** reports the combined duration.
- **Total:** Sum of the displayed model-read, mesh, and most recent preview
  stages. When rotating or zooming, the read/mesh values still refer to the
  loaded model; only the preview stages are newly measured.

The categoriser keeps a **64 MiB bounded LRU** of recently scanned raw model
bytes (up to four models). Changing the current P3D LOD can decode another mesh
from that memory instead of reopening the containing PBO/IFA. The raw bytes are
removed from the long-lived model list after entering the cache and the cache
evicts older models. Models too large for the budget still use the archive
reader. Browsing model order, categories, and saved JSON are unchanged.

PAA/PAC decoding also stops parsing embedded mip headers as soon as the chosen
level is found. Full-resolution selection no longer scans the entire mip chain,
and explicitly selecting a small embedded mip only inspects the necessary
headers before decoding it.

### Selecting embedded texture mipmaps

Both P3D tools now offer a **Texture mipmap** dropdown independently of model
LOD selection and **Skip textures**:

- **Full (mip 0)** (default): Decode the largest embedded PAA/PAC image, with
  the existing screen-space mipmapped filtering in the 3D renderer.
- **Max 512 px / Max 256 px / Max 128 px / Max 64 px:** Decode the first
  embedded mipmap whose width and height fit within that limit.
- **Mip 1 (1/2), Mip 2 (1/4), ...:** Decode that numbered mipmap directly.
  If a texture has fewer embedded levels, use its smallest available mip.

A single model can reference several texture files with different mip counts.
The dropdown is therefore a **per-preview setting**, applied separately to
each referenced texture. The Model & Texture Browser uses it when displaying
an individual texture, too. Changing mipmap settings redraws the currently
selected preview without rescanning or reclassifying models. The categoriser's
**Preview texture** popup uses the same mipmap setting.

Only actual embedded mips are chosen. A single-mip image cannot be made
faster to decode by requesting a smaller level; it remains full resolution.
The PBO/IFA payload still has to be retrieved, but smaller embedded DXT/BCn or
PAC/legacy PAA mipmaps can avoid expensive full-resolution image decoding.
Texture decoding and generated filtering pyramids are cached separately
per mip setting to prevent mixing resolutions.

### Selecting all model LODs

The **Preview LOD** dropdown lists the actual LODs of the currently selected
model for Arma-era ODOL, OFP/CWA **ODOL v6/v7**, and older **MLOD/SP3X** P3Ds.

- **Fast (lowest):** Automatically uses the lowest-detail available *visual*
  LOD, falling back toward higher visual detail if it cannot be decoded.
- **Full (highest):** Uses the highest-detail available visual LOD.
- **LOD 1 (1)**, **LOD 2 (9)**, and so on: Explicitly select that model's
  real LOD resolution. Entries labelled **[special]** are nonvisual LODs
  (geometry, roadway, shadow, etc.). They are listed for inspection but may
  not support textured preview.

Each model has its own list. Changing the selected model refreshes the
choices automatically. Switching LOD in the **Model & Texture Browser** reloads
only that model's preview and keeps separate cached meshes for each quality,
without rescanning the catalogue.

In the **Model Categoriser**, LOD selection is now located **beside Skip
textures** in the main View controls, rather than the startup setup window.
You can switch to another LOD while categorising without changing the model
order or saved categories. The CLI `--preview-lod fast|full` sets the *initial*
quality. The per-model dropdown offers individual LODs once the model loads.

Selecting a lower LOD reduces geometry parsing and preview rasterization but
does not avoid reading the P3D from its archive. OFP/CWA ODOL files store
LOD bodies sequentially, so their LOD index must be read before a later LOD
can be selected. If the index cannot be parsed, legacy models retain their
previously supported first-LOD fallback. Measurements and referenced textures
shown for a preview are based on its selected LOD and may differ between LODs.

Categoriser classification files and companion `.unclassified.json` files
are saved to the output path selected in its GUI, not inside the executable.

If the P3D GUIs fail at startup, review their persistent logs in
`%LOCALAPPDATA%\CWR-Worldgen\logs\` (or under
`~/.cwr-worldgen/CWR-Worldgen/logs/` if LOCALAPPDATA is unset). No console is
shown for these two applications.

## Building locally on Windows

From a checkout of the repository, using PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[sources]" matplotlib pyinstaller
python -m unittest discover -s tests -p test_p3d_frozen_support.py -v
python -m unittest discover -s tests -p test_pyinstaller_splash.py -v
```

Run these from the repository root. Generate the splash images first (the
GitHub Actions workflow does this automatically):

```powershell
python tools/generate_pyinstaller_splash.py --name Categorise-P3D-Models --output build/splash/Categorise-P3D-Models.png
python tools/generate_pyinstaller_splash.py --name Browse-P3D-Assets --output build/splash/Browse-P3D-Assets.png
python tools/generate_pyinstaller_splash.py --name WRP-Mod-Dependency-Scanner --output build/splash/WRP-Mod-Dependency-Scanner.png
```

Then package the executables:

```powershell
python -m PyInstaller --noconfirm --clean --onefile --splash build/splash/Categorise-P3D-Models.png --windowed --paths tools --paths src --collect-all cwr_worldgen --collect-all dem_stitcher --collect-all rasterio --collect-all pyarrow --collect-all duckdb --name Categorise-P3D-Models tools/categorise_p3d_models.py
python -m PyInstaller --noconfirm --clean --onefile --splash build/splash/Browse-P3D-Assets.png --windowed --paths tools --paths src --collect-all cwr_worldgen --collect-all dem_stitcher --collect-all rasterio --collect-all pyarrow --collect-all duckdb --name Browse-P3D-Assets tools/browse_p3d_assets.py
python -m PyInstaller --noconfirm --clean --onefile --splash build/splash/WRP-Mod-Dependency-Scanner.png --console --paths tools --paths src --collect-all cwr_worldgen --collect-all dem_stitcher --collect-all rasterio --collect-all pyarrow --collect-all duckdb --name WRP-Mod-Dependency-Scanner tools/wrp_mod_dependency_scanner.py
```

The executables appear in `dist/`. Build Windows executables on Windows, not
on a Linux/macOS runner. `--collect-all cwr_worldgen` intentionally favors
reliable packaging over smallest-file-size optimization, since importing the
project package initializes much of the world generator.

The workflow verifies the generated executables exist and smoke-tests the
scanner's CLI entry point; the two GUI applications should additionally be
tested manually on a Windows desktop with representative game archives.
