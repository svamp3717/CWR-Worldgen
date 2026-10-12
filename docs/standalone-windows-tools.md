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

Go to **Actions → Build Windows Asset Tools → Run workflow** (after the workflow
is merged into the default branch), then open the completed run and download
each artifact from the **Artifacts** section. Pull requests touching the tools
also trigger builds. Artifacts are retained for 14 days.

The programs package their Python runtime and dependencies. The input game
files are **not** bundled; select your own P3D/WRP/PBO/IFA files.

All three executables show a branded loading screen **during PyInstaller
one-file extraction** and while their startup Python imports finish. The
screen closes when the GUI is ready (or before a command-line WRP scan starts).
This gives visible feedback without introducing an artificial delay. It does
not make the one-file extraction faster. The splash is an experimental
Windows-supported PyInstaller feature and needs a Windows desktop to display.

The categoriser and asset browser open directly as GUI applications. The WRP
scanner opens its GUI when run without arguments but keeps a console window so
it can also be used with CLI options such as:

```powershell
.\WRP-Mod-Dependency-Scanner.exe "C:\path\to\island.wrp" --json "dependencies.json"
```

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
