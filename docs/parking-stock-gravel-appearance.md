# Stock gravel-road texture for unpaved parking

**Confirmed from the actual generator configuration:** Worldgen's default
`gravel_road_model` is `o\\road\\asf25.p3d`, not an asphalt donor. Stock
`sil25.p3d` is paved, and `ces25.p3d` is dirt. A separate legacy OSM
mapping also mentions the Resistance `kos` family, but that is not the
configured default gravel donor.

The user-supplied stock `Road(1).zip` confirms that `asf25.p3d` references
**`o\\road\\asf_new.paa`**. This is a native **AI88** (`0x8080`) legacy
PAA with alpha, not the newer DXT1 format used for generated parking PAAs.

## Rendering

Unsealed parking polygons are composited as world terrain, not floating
roadway slabs. They now sample the actual stock `asf_new.paa` photo loaded
from the existing game asset roots (or a provided stock Road ZIP).

- The original stock `asf25` road is **7 m wide and 25 m long**.
- Its first visual LOD maps U=0..1 across 7 m and V=-2..2 along 25 m,
  meaning a **6.25 m** repeat along the road.
- The parking shader uses the same scale and orientates its material to the
  parking lot's main axis, anchored in world space so neighbouring terrain
  cells agree.
- The original PAA fades to transparent at the road's outside edges.
  Repeating those edge-faded sections inside a lot would reveal bands of
  grass, so parking uses only the opaque central 12%-88% of the actual photo.
  Adjacent tiles are mirrored to avoid abrupt joins.
- Existing parking-specific tyre tracks, circulation wear and parking-space
  compaction are subtle, neutral darkening on the original texture.
- The paved-lot renderer and painted markings are unchanged.

The earlier implementation erroneously used
`create_gravel_junction_texture_image()`, which produces a **different,
generated** neutral-grey gravel photograph. That remains only as an
explicitly reported compatibility fallback when the stock texture cannot
be found. It is *not* the intended result.

## Asset roots and diagnostics

The original stock PAA is proprietary and is **not checked into git**.
The asset loader resolves `o\\road\\asf_new.paa` using the same game asset
roots used to find stock roads (including `O.pbo`), or a supplied
`Road.zip`/extracted `Road` directory.

Regenerate the island. Inspect `parking-lot-textures.json` in the generated
world source directory. It reports:

- `stock_gravel_texture_status: "stock-asf-new-ai88"` when exact stock
  artwork was loaded.
- `stock_gravel_texture_source` and `stock_gravel_texture_sha256` to
  identify the actual source bytes.
- `stock_gravel_texture_status: "missing-stock-asf-photo-generated-fallback"`
  when no usable stock texture was found.

The cache key contains the source PAA SHA-256 and a bumped renderer schema,
so neither the old brown procedural texture nor the mistakenly generated
gravel artwork can be reused as the new material.

## Classification and testing

Existing surface classification is unchanged. Explicit unpaved/gravel/dirt
OSM parking remains unsealed regardless of the nearest road. Untagged lots
retain nearest-road inference; paved lots stay paved. The renderer continues
to respect runway and sports-field priorities.

Tests cover original AI88/LZSS decoding and signed checksum, exact default
model/texture mapping, physical UV repetition, paved-lot isolation, stock
archive discovery, shared terrain-cell behaviour, caching and the existing
OSM parking surface classification.

The Python and CI tests validate the image and outputs. **Visual appearance
inside CWA still requires a regenerated world PBO and an in-game check.**
