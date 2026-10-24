# Procedural unpaved parking with stock gravel colour

Worldgen's default stock gravel model is `o\\road\\asf25.p3d`.
Its opaque surface has a dark neutral grey colour, not brown.

**The parking generator matches that colour, not the original texture.**
Stock road P3D/PAA assets are **never loaded or sampled** when painting parking.
The user specifically does not want road-photo tiling on parking lots.

## Colour and detail

The central opaque region of the user-supplied stock `asf25` reference image
averages about **RGB (74, 74, 74)**. Common opaque intensities are 51, 68,
85 and 102. These *numeric colour statistics* establish the palette, not a
copied image pattern.

`stock_gravel_parking_texture.stock_gravel_colour_material` produces fresh
opaque grey gravel from deterministic, world-coordinate grain at several
physical scales. The result is monochrome and slightly mottled, with no
repeating road photograph, no transparent road margins, no material PAA
source dependencies and no brown tint. Existing subdued parking circulation
wear and tyre ruts darken the image without adding colour.

Each WRP terrain cell samples the same world-coordinate grain, preventing
patterns from restarting at cell borders. Pixel detail is limited by the
generated terrain-texture resolution, which can be lower than actual P3D
road texture resolution.

## Selection and preserved behaviour

- Explicit `surface=gravel`, `dirt`, `earth`, `ground`, `sand`,
  `unpaved`, `compacted`, `fine_gravel`, etc. remain unsealed.
- Untagged parking lots continue to infer paved/gravel from the nearest
  supported road.
- Paved parking surfaces and parking-bay markings remain unchanged.
- Runways and sports pitches retain priority over parking.
- No new P3D meshes or stock game assets are bundled.

## Validation and caching

The parking cache schema and recipe fingerprint were bumped, so older
photo-tiled and brown-gravel PAAs cannot hide this new procedural colour.

The generated `parking-lot-textures.json` report records:

- `gravel_material: "procedural-stock-asf25-gravel-grey-v1"`
- `gravel_reference_rgb: [74, 74, 74]`
- `gravel_asset_source: "procedural-no-stock-texture-used"`

CI tests assert average tone near the reference, all three RGB channels
equal, independent world-coordinate grain, unchanged paved appearance,
OSM parking selection, deterministic output, cache generation and zero
road-photo reads.

Regenerate the world PBO to inspect the updated colours in CWA. Python
tests verify the generator, not CWA's own texture filtering/rendering.
