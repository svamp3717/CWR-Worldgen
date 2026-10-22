# Worldgen Debug Layers in Island Studio

Worldgen can optionally preserve its intermediate **OSM and terrain masks** in a
compressed companion file for Island Studio. This is diagnostic metadata,
**not** a game texture, and is never added to the world PBO.

## Exporting debug data

In the Worldgen GUI, expand **Advanced → Build behavior** and check
**Export Worldgen debug layers for Island Studio** before building the world.

CLI (Milestone 9): add `--worldgen-debug-layers` to the `milestone9`
subcommand options.

The build then creates:

```text
build/my_world/
  cwr_my_world-worldgen-debug.npz
  source/cwr_my_world/cwr_my_world.wrp
  @CWR-Milestone9/Addons/cwr_my_world.pbo
```

The NPZ sidecar is optional. Normal generation is unchanged when the option is
off, and the debug file is not part of the game-facing PBO.

## Viewing in Island Studio

Open the generated WRP or PBO while the sidecar remains in the build output
folder. Island Studio searches nearby parent folders and automatically loads
matching debug data. Alternatively use **Utilities → Load Worldgen Debug
Layers…** to select the NPZ manually.

Under **Worldgen debug overlay**, select:

- **OSM water, forest, roads, farmland, urban, buildings**: original
  rasterized OSM source masks, before the Worldgen surface classification.
- **Base terrain materials**: Worldgen's initial material classification.
- **Final terrain materials**: semantic materials after surface and object
  placement passes, before translation to stock texture filenames.
- **Final WRP texture slots**: actual slot numbers written into the WRP,
  which can differ from semantic material IDs for forest-border blends.
- **Placed forest objects**: per-cell density of generated forest-model
  placements. This is not the same as the OSM forest source mask.

Debug layers are translucent on the current view. Select **Off** to return to
the unmodified elevation, texture, or road preview. The cursor readout includes
the selected layer's raw cell value or material code. Map borders and optional
Elevation shorelines continue working.

Layers use the same WRP coordinates: X increases east, Z increases north and
row zero is the south of the island. The viewer samples them north-up.

## Safety and limitations

- The NPZ embeds a SHA-256 hash of the generated WRP. Island Studio refuses
  to apply a sidecar to a different WRP, even if dimensions happen to match.
- A debug companion belongs to a specific generation. Regenerating a world
  requires regenerating the sidecar too.
- These are **rasterized OSM features** at WRP cell resolution, not raw
  vector OSM geometry, and cannot replace the original OSM source JSON.
- The placed-forest layer counts recognized forest-model placements; it is
  not an exhaustive tree canopy or visibility simulation.
- Files from older builds won't contain this data. Island Studio still works
  normally without a debug sidecar.
