# Road Inspector

Road Inspector is a read-only post-build diagnostic for the road geometry that is actually stored in a generated CWA RVW4 world. It does not use RoadLab and it does not modify the WRP, PBO, source roads, terrain, fences, signs, buildings, or other generated objects.

## Quick start on Windows

After pulling `fix/stock-road-curves`, double-click:

```text
INSPECT-ROADS.cmd
```

With no arguments the launcher opens the Road Inspector GUI. Choose a generated `.pbo` or `.wrp`, optionally select `normalized\roads.geojson`, choose the report folder, and click **Run inspection**. The GUI automatically looks for a nearby `normalized\roads.geojson`, keeps the diagnostic thresholds editable, and can open the HTML report or its output folder when the audit finishes.

The existing drag-and-drop workflow still works: drag a generated `.pbo` or `.wrp` onto `INSPECT-ROADS.cmd` to run the inspector immediately. The launcher looks for a nearby `normalized\roads.geojson`. If the source bundle is elsewhere, run:

```bat
INSPECT-ROADS.cmd "G:\path\wg_lundby.pbo" "G:\path\lundby\normalized\roads.geojson"
```

The report is written beside the inspected world as `<world-name>-road-inspector\report.html` and is opened in the default browser.

The installed GUI can also be opened directly:

```bat
cwr-road-inspector-gui
```

The installed command-line inspector remains available:

```bat
cwr-road-inspector "G:\path\wg_lundby.pbo" --roads "G:\path\lundby\normalized\roads.geojson" --output "G:\path\road-inspector"
```

or from a source checkout:

```bat
python -m cwr_worldgen.road_inspector_entry "G:\path\wg_lundby.pbo" --roads "G:\path\lundby\normalized\roads.geojson" --output "G:\path\road-inspector"
```

## What it checks

The inspector reconstructs stock `sil`, `asf`, `kos`, and `ces` road objects from the actual RVW4 object transforms. It uses the measured stock straight, 10-degree curve, T-junction, and X-junction geometry already used by the road fitter.

It currently reports:

- `straight_miter`: touching straight pieces whose center connectors meet but whose headings open a visible edge wedge;
- `curve_transition`: curve/straight or curve/curve seams with incompatible rendered tangents or edges;
- `connector_gap`: physically separated connectors, including mutually-facing gaps outside the normal seam-clustering tolerance;
- `surface_family_mismatch`: connected stock pieces using different road families where the transition should be checked;
- `junction_connector_mismatch`: a native junction connector that does not line up with its emitted approach;
- `wrong_intersection_model`: T/X connector count does not match the normalized source degree;
- `intersection_connector_orientation`: rigid native junction headings do not match the normalized incident roads;
- `turning_intersection_cap`: a legacy straight cap is being used where the through road turns at the intersection;
- `intersection_approach_mismatch`: one or more final approach pieces do not follow the normalized source direction at the node;
- `intersection_missing_cap`: a normalized multi-arm intersection has no nearby stock cap/junction and the emitted approaches do not explain it cleanly.

For every issue the report includes world X/Z coordinates, object IDs, P3D names, a severity score, measured center/tangent/edge errors when applicable, and a suggested class of road-fitting fix.

When `normalized\roads.geojson` is supplied, WGS84 source coordinates are projected back into the generated world's metre coordinate system using the bundle's `bbox` and `cwr_world` metadata. Findings are then annotated with nearby normalized `road_id`, `highway`, and `surface` values so a bad WRP object can be traced back to the source feature.

## Output files

Each run creates:

```text
report.html
issues.json
issues.csv
summary.json
ingame-coordinates.csv
```

`ingame-coordinates.csv` is the upload-friendly coordinate log. It contains one row for every inspector finding and one row for every inspected road object, including world X/Z, road family/kind/model, related issue IDs, normalized source context where available, and a ready-to-paste `player setPos [...]` command. Upload this single CSV when you want the inspection coordinates reviewed without having to reconstruct them from the HTML report.

`report.html` is self-contained. Click an issue to zoom to its position and highlight the involved road objects. Filters can limit the list by severity or issue category.

The HTML report now defaults to **paved-only focus** for intersection diagnostics. Dirt-only intersections and mixed paved/dirt intersections are hidden from the issue list and map markers by default, because current in-game checks show those combinations are not visible defects. Use **Show dirt/mixed intersections** to reveal them. When normalized source `surface` metadata reports dirt, gravel, unpaved, or similar surfaces, the finding is de-emphasized; the report also falls back to involved stock road families, so any intersection involving a `ces` road is hidden from paved-only focus. Pure `sil`, `asf`, and `kos` intersections remain visible. This is only a report-view filter: `issues.json`, `issues.csv`, `ingame-coordinates.csv`, and the underlying measurements still contain every finding.

Every issue also includes copy-ready CWA debug-console teleport commands. **Finding** teleports to the measured defect coordinate, while each **Road N** entry teleports to the center of that specific involved road object. The generated command uses CWA/OFP syntax:

```text
player setPos [worldX, worldZ, 0]
```

The inspector's WRP X/Z plane corresponds to the game's horizontal X/Y position array, so the report deliberately writes the inspected world Z value into the second `setPos` coordinate. A height of `0` places the player on the terrain surface.

`issues.json` is intended for future automated road-repair experiments. The first versions of Road Inspector remain deliberately read-only until the measurements have been compared against enough in-game failures.

## Geometry accuracy

Road Inspector reads the yaw and pitch actually serialized in the RVW4 transform. This matters on graded roads: using nominal P3D length only in horizontal X/Z space can manufacture false connector gaps. Curves and native junction connectors are likewise projected through the stored yaw/pitch matrix before seam measurements are made.

## Current limitations

- It recognizes the stock road P3Ds and native junction models currently supported by the generator. Generated custom gravel geometry needs a separate footprint model before it can receive the same edge-level analysis.
- The HTML overview currently draws road center geometry and highlights the objects involved in a finding. Edge-gap numbers are measured numerically, but full rendered road-strip footprints are a planned visualization improvement.
- Candidate fixes are recommendations, not automatic WRP edits. The inspector should first prove that its top-ranked findings correspond to visible CWA defects.
