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

The Island Studio workflow runs tests on source changes. A dedicated Windows
executable can be built separately on request, using the manual workflow
option. The tool is not bundled with the generator executable.

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

## Roads as a separate editing layer

Road pieces are WRP **world objects**, not ground texture colours. Enable
**Show roads as lines** or choose the **Road preview** view to see road-centre
lines over the map. Known stock straight pieces display their approximate
footprint; unrecognized road models appear as small orange markers. This is
a 2D preview, not the complete game model or collision mesh.

Choose **Draw road line**, select a compatible straight road model and click
the starting point. Move the cursor to preview the path; click the endpoint
to place as many complete segments as fit between the two points. Press
Escape to cancel. Choose **Erase road** and click near a road object's
centre to remove just that object. The regular object editor remains
available for editing positions, headings and paths individually.

Drawing currently supports stock straight paved/dirt/asphalt road lengths
(25, 12.5, 6.25 metres) and compatible generated straight models. This first
version does **not** automatically connect intersections, snap segments to
nearby roads, compute graded terrain, or create bends/curves. It places
objects at sampled terrain height; check the connection and elevation in
the target game before using edited roads.

## Water and real surface images

The Elevation and Road preview views display water in blue wherever the
heightmap is at or below the chosen **Water level, metres** (default 0).
This is a height-based water preview, not hydrodynamic or coastline modelling.

The **Texture slots** view remains a lightweight false-colour diagnostic.
Choose **Actual textures** and supply a **texture assets folder** to display
real images referenced by the WRP texture table. The folder should contain
an extracted loose-file tree matching the virtual texture paths, for example
an asset at o/data/grass.paa for the texture table entry o\\data\\grass.paa.
Supported formats currently include DXT1/DXT5 .paa and ordinary image
files such as .png/.jpg. Missing or unsupported textures retain their
existing slot colour, so gaps are visible.

The preview loads only the visible area, with bounded decoded texture
caching. It does not resolve custom texture
coordinate transforms or reproduce every in-engine terrain blending effect.

## Zoom controls and mouse-pointer shortcuts

Use the **+** and **−** buttons next to Fit map, the mouse wheel, or the
`+` / `-` keys (also `=`, numeric keypad +/−, and Ctrl+plus/minus).
All methods zoom toward the mouse pointer while it is over the viewport,
keeping the same map location beneath the cursor. When the pointer is outside
the map, the buttons and keyboard shortcuts zoom around the map centre.
Shortcuts are suppressed while editing a text field.

## Select PBOs for real texture previews

Choose **Select texture PBOs…** to select one or more loose `.pbo` or
`.pbo.zst` archives. Also accepted: `.ifa` and `.ifa.zst`. Archive headers
are indexed in a background task; images are decoded *only if needed* by
the current **Actual textures** viewport. Archives with a declared PBO
`prefix` use that namespace, otherwise the archive filename is the
namespace. If several selected archives supply the same virtual texture
path, the last archive wins. Use **Clear selected PBOs** to release the
indexes and cached preview tiles.

You can combine selected archives with a loose texture folder. A loose
file is preferred when both sources contain the same virtual path, and
missing or unsupported images stay visibly rendered as slot colours.
This version supports ordinary first-level texture members, including
DXT1/DXT5 PAA. Nested archives inside a PBO are not indexed. Whole-file
Zstandard-wrapped archives are temporarily unpacked to a scratch file
for random-access indexing, then deleted when the selection is cleared.

PBO selection is read-only. The tool does not modify, repack or extract
game archives into the map's output folder.

## Builds

Source updates trigger validation tests only. Windows EXE packaging is
**manual opt-in** using the `build_windows_exe` workflow input and is
not performed automatically when this branch receives commits.

## True-size world object markers

In the **View** sidebar, enable **Show objects at model size** while
**Show world objects** is enabled. Visible objects with readable P3D
geometry then appear as outline rectangles of their model's real first-LOD
horizontal extents in world metres. The preview keeps the actual model
origin and each object's stored WRP horizontal rotation, scale and shear
instead of centring a guessed rectangle at the object position.

The editor uses the previously selected **texture assets folder** for loose
`.p3d` files and **selected PBOs** for archived models. There is no need
to extract entire archives. Model measurements load in small background
batches only for visible object types, and stay cached while browsing.
The original map and assets remain unchanged.

The first version measures OFP/CWA-era MLOD/SP3X and ODOL v6/v7 first
visual LODs. Unsupported or missing models continue to appear as regular
dots, and very small outlines collapse to dots at far zoom levels. These
are **2D visual bounding footprints**, not exact mesh silhouettes,
collision bounds or full 3D views. Object selection supports clicks inside
a visible outline.

This option is off by default so large-world overview navigation retains
its original lightweight fixed-size object markers.

## Multiple asset folders and PBO archives

Use **Add folder…** under **Texture / model folders** as many times as needed
to select multiple directories. The native folder dialog selects one folder
per invocation; every new selection is appended rather than replacing the
existing set. The selectable list displays the complete paths. Use **Remove**
to delete highlighted entries (Ctrl/Shift for multiple selections), or
**Clear** to deselect all directories.

Use **Add PBOs…** to select several PBOs in one dialog and add more in later
sessions. Previous archives remain selected. The adjacent list supports
the same **Remove** and **Clear** controls. Rebuilding the archive index
runs in the background and leaves the previous selection intact if loading
a new archive fails. Models and terrain images share the same source lists.

Asset precedence is deliberate: the **most recently added loose folder**
containing a matching virtual path wins; loose files take precedence over
PBO archive members. If no matching loose file exists, the **last selected
PBO** containing that path wins. Removal reveals the earlier matching file.
Duplicate folder and archive paths are ignored.

For example, a virtual model path `o\\buildings\\house.p3d` is looked up
under `<selected folder>/o/buildings/house.p3d`. A selected directory should
be the **parent** of the virtual namespace (`o` in this example), not the
namespace subdirectory itself.

Changing the selection invalidates affected decoded-image and footprint
caches. No world objects are regenerated or moved, and the WRP source is not
modified. Actual textures and model-size footprints continue to load only
the required visible assets.

## Hover readouts for elevation and textures

The bottom status area includes a separate live cursor readout while the
pointer is over the map. In **Elevation** view it shows world X/Z coordinates
and the height sample in metres, using the configured **Height scale**.
In **Texture slots** and **Actual textures** views it shows the terrain
texture-slot number and the name stored in the WRP texture table, for example
`Texture 1: o\\data\\rock.paa`. Unnamed or invalid slots are marked
`(unnamed)`. **Road preview** continues to show coordinates alone.

The readout uses the exact grid cell underneath the pointer and includes
unsaved height and texture brush edits. It changes on mouse movement,
panning, zooming and view switching, and clears when the pointer leaves the
map. This lookup reads only a single terrain or texture sample; it does not
re-render terrain, decode a texture, or load a model on pointer movement.
The ordinary build, save and selection messages remain in their own area.

## Show forests independently of other object markers

Select **Show forests (green overlay)** in the View sidebar to highlight
forest blocks, generated forest clusters, trees, and common vegetation models
recorded in the WRP. This can be enabled while **Show world objects** is off.
Forest markers are green; models whose first visual P3D LOD can be measured
show shaded green 2D footprints and other forestry assets show green dots.
The overlay reads WRP forest/vegetation objects, **not** original OSM forest
polygons. It is a placement preview and may not reproduce a model's full
in-game canopy, proxies, or collision geometry. The view uses a compact
background classification mask and a bounded visible-object query.

**Show objects at model size** is now independent of **Show world objects**.
Turning off ordinary white object dots while leaving model size enabled
keeps genuine model footprint outlines visible. If an asset is missing,
unsupported, or smaller than two screen pixels at the current zoom, no
generic fallback dot is drawn unless **Show world objects** is turned on.
The forest overlay retains its own green fallback dots when enabled.
The viewer does not invent dimensions for unmeasured models.

The two overlays can be combined with road previews or either terrain
background. The map source and exported object positions are unchanged.

## Arrow-key map navigation

Use the keyboard **Left**, **Right**, **Up**, and **Down** arrows to pan
the map in the indicated direction. Each press advances the viewport by
80 screen pixels at the current zoom level. Hold **Shift** with an arrow
to move three times as far, useful for crossing large islands quickly.
Holding an arrow lets the operating system's key repeat continue moving.

Panning updates the terrain/texture readout beneath the pointer and requests
only the normal viewport redraw. Arrow navigation is disabled during file
operations and while a text entry, dropdown, list or similar editable
widget has keyboard focus, so entering coordinates and choosing assets
continues to work normally. The arrow keys follow the same unclamped
navigation behavior as right-button map dragging.
