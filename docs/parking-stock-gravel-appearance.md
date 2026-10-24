# Stock-style gravel appearance for unpaved parking lots

This branch makes non-paved OSM parking polygons use the **same neutral-grey
photographic aggregate** as the generated stock-style CWA gravel roads.

## Selection

The existing parking classifier is unchanged:

- `surface=asphalt` and lots near a paved road retain their paved appearance.
- Parking explicitly tagged `surface=gravel`, `dirt`, `earth`, `sand`,
  `unpaved`, `compacted`, `fine_gravel`, etc., remains gravel regardless
  of nearby asphalt, through `parking_dirt_surface_policy`.
- Parking without an explicit unsealed surface tag continues to take the
  nearest supported road's surface classification.

## Appearance

The prior gravel fill was an independent warm-brown random colour.
Its additional parking detail drew more brown speckles and warm tyre tracks,
making lots look different from adjoining neutral-grey gravel roads.

The revised renderer reuses
`procedural_infrastructure.create_gravel_junction_texture_image()`:
this is the **same road gravel photograph and neutral grading** used by
generated gravel junction P3Ds. This junction variant is deliberately opaque
and omits directional wheel tracks/transparent edge strips, which are
unsuitable for two-dimensional terrain parking surfaces.

- Gravel repeats at `GENERATED_GRAVEL_TEXTURE_REPEAT_METRES` (currently
  3 m), anchored to **world coordinates** rather than per-cell pixel positions.
- Texture sampling is bilinear from a prefiltered, cacheable road photo tile.
  At lower WRP terrain texture resolution, gravel stones cannot have the same
  visible pixel detail as P3D roads, but the material, tone and repetition
  come from the same source.
- Small parking-specific tyre ruts, aisle compaction and parking wear
  darken existing pixels neutrally without adding beige or dirt colours.
- Terrain pixels outside parking polygons remain untouched.
- Paved parking artwork, stall dividers, and priority versus runways and
  sports pitches are unchanged.

There is no separate original stock `gravel` model family in
`data/road_types.json`; the project uses a generated gravel P3D family
next to the original stock paved and dirt families. This implementation
matches that generated gravel family rather than taking a texture from
stock `asf` (paved) or `ces` (dirt) models.

## Caches and testing

`PARKING_TEXTURE_CACHE_SCHEMA` and the renderer's cache fingerprint have
been changed so old brown parking PAAs cannot be silently reused.
Build a fresh world PBO to see the change; old exported maps still contain
the previous parking PAA assets.

`tests/test_parking_stock_gravel_texture.py` and the existing parking
test suite verify the exact shared road artwork, world-coordinate tile
repetition, unsealed vs paved rendering, neutral tyre wear, OSM surface
overrides and cache generation. GitHub Actions validates the feature branch.

In-game appearance should still be checked in CWA at the desired world
texture resolution; Python tests do not simulate the game renderer.
