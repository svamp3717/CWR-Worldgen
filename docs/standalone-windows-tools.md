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
