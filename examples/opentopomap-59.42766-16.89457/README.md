# OpenTopoMap 59.42766, 16.89457 example

This example turns the supplied map link into a 6.4 km square CWR-CE world:

```text
https://opentopomap.org/#map=13/59.42766/16.89457
```

Milestone 5 uses the URL only to select the centre and reference zoom. Elevation comes
from a real DEM, not from trying to decode contour-line colours like an archaeologist
with a graphics card.

## Install

```sh
python -m pip install -e '.[sources]'
```

## Fetch and build

```sh
python examples/opentopomap-59.42766-16.89457/build_example.py
```

The default source provider is Copernicus GLO-30 through `dem-stitcher`. A lighter SRTM3
fallback is available:

```sh
python examples/opentopomap-59.42766-16.89457/build_example.py --dem-provider hgt
```

To verify every referenced model against an installed CWR-CE tree:

```sh
python examples/opentopomap-59.42766-16.89457/build_example.py \
  --asset-root 'D:/Games/CWR-CE' \
  --strict-assets
```

The source snapshot is stored under `build/malaren-source-data`. Later builds validate and
reuse it without network access. Use `--refresh` only when deliberately replacing the OSM
and DEM snapshot. The two stages may also be run separately:

```sh
python examples/opentopomap-59.42766-16.89457/build_example.py --fetch-only
python examples/opentopomap-59.42766-16.89457/build_example.py --build-only
```

The exact selection is:

```text
south  59.398881748360814
west   16.837989602089053
north  59.456438251639190
east   16.951150397910950
```

Generated packages:

```text
build/malaren-example/cwr-malaren-runtime.zip
build/malaren-example/cwr-malaren-test-mission.zip
```

Install the complete mod folder from the runtime ZIP. The `Anims` directory and source
attribution files are part of the package, not optional decorative foliage.
