# CWR Island Studio

Island Studio is a standalone, non-destructive map and object editor for
unbinarized RVW4 / 4WVR .wrp islands. It is designed to open large worlds
without eagerly decoding their object records or building a full-resolution
image of the entire terrain. The editor uses a lightweight Python package,
so opening the GUI does not initialize Worldgen's generation pipeline.

## Installation and launch

Install the project:

    python -m pip install -e .

Start the editor:

    cwr-island-studio-gui

Or open an island directly:

    cwr-island-studio /path/to/island.wrp

On Windows, the Island Studio validation and Windows build workflow creates
a CWR-Island-Studio-Windows artifact containing CWR-Island-Studio.exe.
The tool is not bundled with the generator executable.

## Workflow

1. Set Grid metres and Height scale to the values used to generate the map,
   then select Open WRP. The 4WVR header encodes neither scale value.
2. Use the wheel to zoom and right-drag to pan. Choose Elevation or Texture
   slots. Slot colours are index visualizations, not decoded .paa textures.
3. Select a world object to edit its model path, X/Y/Z position and heading.
   Vertical/shear transform components are retained on existing objects.
   Add object at map centre and Remove selected object are also available.
4. Use Raise terrain, Lower terrain and Paint texture with the chosen brush
   radius. Height edits quantize to the configured height scale. Texture
   indices must be between 0 and 511.
5. Save As writes a new WRP. The application refuses to overwrite the source.

Fit map returns to the complete island.

## Why loading scales well

- Terrain and fixed-size object records are read-only memory-mapped NumPy
  views. Opening does not allocate per-object Python tuples and strings.
- Background indexing sorts compact object-record indices into 256-metre
  spatial buckets after the terrain is already available.
- Rendering samples only visible screen pixels (up to 1600 by 1200). It never
  allocates a full-island RGB image.
- The viewport limits rendered object markers to 2700. Zoom in to inspect
  dense groups individually.
- Sparse in-memory edits avoid reconstructing original objects. Unchanged
  matrix bytes are preserved.
- Save As copies source bytes and patches changed fields; object deletion
  uses a separate compaction path.

No end-to-end loading benchmark against previous software has been published
yet. Performance still depends on object density and hardware, and background
index construction can take longer for million-object maps.

## Format support and current limits

This first version is a functional **2D editor**, not a 3D mesh viewport.
It supports terrain height samples, texture-slot IDs, and object positions,
heading and model paths. It does not yet offer advanced transform controls,
a full road editor, 3D scene rendering, texture-image decoding or undo/redo.

Open a loose, unbinarized 4WVR WRP. Binarized OPRW, 8WVR and direct PBO/IFA
archive editing are not supported. Extract a WRP from an archive first.
Memory availability and the target game engine impose practical limits beyond
the file reader's terrain-size handling.

The original is always left untouched. Export is atomic within the destination
directory. Retained object records preserve their untouched binary fields,
including terrain shear and tilt. Keep backups and validate changed maps in
the target engine.

## Tests

    python -m pytest -q tests/test_island_studio.py

The tests cover untouched byte-identical exports, sparse height/texture edits,
model and position changes, object deletion and insertion, spatial index
queries, malformed files and a 90,000-object input. GitHub Actions validates
the editor on Linux and Windows and packages a separate Windows executable.
