**CWR Worldgen**

A tool that generates playable Operation Flashpoint / Cold War Assault islands from real-world OpenStreetMap + elevation data, it also uses Overture as fallback for missing buildings in OSM.

It can generate terrain, roads, gravel roads, bridges, buildings, forests, beaches, parks, water and vegetation, then package everything into a CWA/OFP pbo file into a modfolder for you.

The goal is to turn real places into recognizable, playable OFP maps with as little manual work as possible.

## WRP mod dependency scanner

A standalone read-only tool can scan a loose `.wrp` or every `.wrp` inside a `.pbo` and list the P3D model namespaces it references. By default `data3d\\` and `o\\` are treated as stock CWA/OFP roots, so addon namespaces are shown as mod dependencies. The stock-root list is editable in the GUI and repeatable on the command line.

From a source checkout:

```text
python tools/wrp_mod_dependency_scanner.py
python tools/wrp_mod_dependency_scanner.py myisland.pbo
python tools/wrp_mod_dependency_scanner.py myisland.wrp --json dependencies.json --csv dependencies.csv
```

Installed entry points:

```text
cwr-wrp-mod-scan
cwr-wrp-mod-scan-gui
```

The GUI supports PBO/WRP browsing, optional display of stock references, copying unique P3D paths, and CSV/JSON export. RVW4 files use exact object-record counts. Legacy/addon WRP formats use conservative embedded P3D-path discovery and are labeled accordingly rather than claiming exact placement counts.
