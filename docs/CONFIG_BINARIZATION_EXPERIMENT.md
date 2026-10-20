# Experimental OFP/CWA config.cpp → config.bin binarization

The Worldgen GUI exposes **Advanced generation, cache and PBO settings →
Build behavior → Experimental: binarize addon config.cpp as CWA RAP v4
config.bin**. The option is **off by default** and remembered between GUI
sessions. It is independent of **Binarize generated world as OPRW v3** and
**Binarize procedural buildings as ODOL v7**.

CLI equivalent:

```text
python -m cwr_worldgen milestone9 --source-dir SOURCE ... --binarize-config
```

## Build output

When selected, Milestone 9 first generates its normal, editable source
`<world>/config.cpp`. After ordinary generation and validation:

- The full generated config is parsed strictly and encoded as **native CWA RAP
  version 4**, using the compact shared string table that CWR-CE's
  `ParamFile::SerializeBin` decoder expects.
- A standalone `<world>.config.bin` is preserved in the build output.
- The game PBO contains **`config.bin` instead of `config.cpp`**. The source
  directory retains its editable `config.cpp` and its original WRP.
- If world binarization is also enabled, a **single** staged game PBO is packed
  with both `config.bin` and the OPRW world. The original source files remain
  unchanged.
- The existing game PBO is replaced only after packing has succeeded. A failed
  conversion/repack reports an error and leaves the original game PBO and
  editable sources intact.
- The `build-manifest.json` records `config_binarization.enabled` and the
  independent generated binary's path.

## Compatibility and safeguards

The text parser supports the syntax Worldgen emits: nested class declarations
and inheritance, quoted strings (including doubled quotes), ints, floats,
nested arrays and comments. It intentionally **rejects unsupported
preprocessor directives and expressions**, rather than silently changing
important addon settings. This is a CWA-specific RAP v4 encoder, not the
modern ArmA RAP layout. It validates matching `CfgWorlds.LandGrid` and place
label counts after conversion; Island Studio can read the binary config.

The binary config may reduce addon parsing time and size, but does not change
terrain rendering or increase FPS while playing. Always test the resulting
PBO in OFP/CWA before distributing production builds. The encoder is
covered by synthetic format and PBO regression tests but has not undergone
a full real-game smoke test.
