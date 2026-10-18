# Worldgen Windows and Linux startup splash

The main CWR Worldgen Windows one-file executable now uses a generated
**forest-green** loading splash. It follows the 640×360 technical-grid design
of the standalone P3D tools, but has its own moss-green accent and the single
white **WORLDGEN** wordmark. The old "Custom worlds for Operation Flashpoint"
tagline is intentionally omitted.

## Reproduce the artwork

Run from the repository root with Pillow installed:

```powershell
python tools/generate_pyinstaller_splash.py --name CWR-Worldgen --output build/splash/CWR-Worldgen.png
```

The PNG is created at build time, not stored as a binary source asset.
The short filled bar is decorative and **not** a reported extraction percentage.

## Windows builds

Both the manual **Build Windows EXE** action and the Windows job in **Build and
Publish Release** run the same image generator, call `pyi-makespec` with
`--splash`, apply `tools/configure_tool_splash_spec.py` to give the splash
a normal taskbar window, and build `CWR-Worldgen.spec`.

The splash appears while the PyInstaller one-file bootloader unpacks and
imports its dependencies. The GUI closes it after the Tk window has been laid
out and before entering the event loop. It starts in front, releases
always-on-top after 1.5 seconds, and can be selected from the Windows
taskbar while visible. It is not intended to track generation progress.

## Linux builds

The manual **Build Linux App** workflow and the Linux job in **Build and
Publish Release** generate the same forest-green `CWR-Worldgen.png` using
`tools/generate_pyinstaller_splash.py` and pass it to PyInstaller via
`--splash build/splash/CWR-Worldgen.png`. They also execute the existing
splash artwork regression tests during the build.

Linux uses PyInstaller's native Tcl/Tk splash behavior. Unlike the Windows
build, it does not apply `configure_tool_splash_spec.py`, which changes
Windows-specific window/taskbar behavior. Linux must run under a graphical
desktop with the necessary Tk/X11 libraries; a text-only terminal or headless
server cannot display the image. When the GUI window is ready, the shared
`pyi_splash.close()` handling dismisses the splash.

The Windows *loose-file/onedir* build and the macOS build are unchanged.
The release workflow remains manual and `main`-only, so a new executable will
not be published until the branch is merged and that action is run.
