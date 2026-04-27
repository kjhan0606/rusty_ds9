# ds9-rust

A Rust + [slint](https://slint.dev) port of [SAOImage DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9), the
astronomical FITS viewer originally written in Tcl/Tk by the Smithsonian Astrophysical Observatory.

This is a clean rewrite focused on the OGFinder-style workflow: open a FITS image, overlay a SExtractor catalog, draw and
edit DS9 regions, and switch between multiple frames.

## Build

```sh
cargo build -p ds9-app --release
```

Requires Rust ≥ 1.85. The vendored `fitsrs` patch in `crates/vendor/fitsrs` fixes an `f32` precision-loss bug in tile-
compressed (`.fz`) DES-style mosaics — keep the workspace `[patch.crates-io]` in place.

## Run

```sh
# empty session — open files via File ▸ Open…
./target/release/ds9-app

# preload an image (and optionally a region file + catalog)
./target/release/ds9-app image.fits regions.reg sources.cat
```

Supported inputs:

| Type    | Formats                                        |
|---------|------------------------------------------------|
| Image   | `.fits`, `.fit`, `.fts`, `.fz` (tile-compressed) |
| Region  | DS9 `.reg` — `circle / box / ellipse / annulus / point / line / polygon` in `image` coords |
| Catalog | TSV, whitespace, or SExtractor ASCII_HEAD       |

## Quick guide

### Mouse / keyboard

| Action                       | How                                                                     |
|------------------------------|-------------------------------------------------------------------------|
| Pan                          | Drag on the canvas (when not over a region)                            |
| Zoom                         | `Zoom ▸ Zoom In / Out / Fit / Reset` or pick a fixed level (`1× … 32×`)|
| Cursor readout (x/y/value/WCS) | Hover — shown in the info bar                                         |
| Select a region              | Click on it (any mode)                                                  |
| Drag a region                | `edit` mode + press-and-drag on the region                              |
| Drop a new circle            | `edit` mode + click on empty canvas                                     |
| Pick a catalog source        | Click near it on the image, **or** click its row in the table          |

Mode toggles live next to the info bar (`pan` / `edit` / `region` / `crosshair`). Catalog clicks recenter the view on
the selected source.

### Menus

| Menu      | Highlights                                                                       |
|-----------|----------------------------------------------------------------------------------|
| File      | `Open…`, `Quit`                                                                  |
| Frame     | `New Frame` (opens a new file alongside), `Delete Frame`, `Next` / `Previous`    |
| Zoom      | In / Out / Fit / Reset, fixed `1× / 2× / 4× / 8× / 16× / 32×`                    |
| Scale     | Stretch (`linear`, `log`, `sqrt`, `squared`, `asinh`, `sinh`) + limits (`zscale`, `minmax`) |
| Color     | Colormaps — `grey`, `red`, `green`, `blue`, `heat`, `cool`, `rainbow`, `viridis`, `cubehelix`, `aips0`, `sls` |
| Region    | `New`, `Load…`, `Save…`, `Delete Selected`, `Delete All`, `Info`                |
| Catalog   | `Load…`, `Clear`, `Info`                                                         |
| Analysis  | `Pixel Table…` (5×5 around cursor), `Statistics…`, `Histogram…`, `Contour Levels…` |

### Multi-frame

Each loaded image is its own *frame* — independent stretch, limits, colormap, regions, catalog, zoom, and pan. `Frame ▸
New Frame` opens a file dialog and pushes a new frame; `Next` / `Previous` cycles through them; the per-frame view state
is restored losslessly on switch. The `FRM` cell in the info bar shows `<active>/<total>`.

### Histogram + Contours

`Analysis ▸ Histogram…` toggles a floating 128-bin log-scale histogram (computed against the active frame's stretch
limits). `Analysis ▸ Contour Levels…` toggles a 5-level contour overlay (sign-change detection between zscale low/high)
drawn on top of the image at the same canvas geometry. Both follow the active frame and refresh when re-opened.

## Workspace layout

```
crates/
  ds9-app      — slint UI shell, event wiring, menubar, frame/state management
  ds9-fits     — FITS I/O (BITPIX 8/16/32/-32/-64, BSCALE/BZERO, BLANK, tile-compressed via fitsrs) + minimal WCS
  ds9-image    — stretch (linear/log/sqrt/squared/asinh/sinh), limits (zscale/minmax), colormaps, RGBA render
  ds9-marker   — DS9 .reg parser/writer + region/marker model (image-coord shapes)
  ds9-catalog  — TSV / SExtractor ASCII_HEAD parser, column lookup, sort
  vendor/fitsrs — local fitsrs patch (f64 unquantize for `.fz`)
```

## Status

Implemented:

- Single- and multi-extension FITS load (with WCS-aware sexagesimal readout when present).
- All seven DS9 stretches and the standard colormaps.
- DS9 `.reg` round-trip (image coords).
- SExtractor catalog overlay, click-to-select, table-row recentering.
- Region editing — click select, drag in `edit` mode, delete selected / all.
- Multiple frames with per-frame view state.
- Statistics, pixel table, histogram, contour levels.

Not yet ported (open candidates):

- WCS-coord regions (`fk5`, `galactic`, …) — currently parsed positionally as `image`.
- Smoothing / binning / blink, RGB composite frames.
- Print, Save Image (PNG/FITS export), Match (frame-to-frame alignment).
- Scripts / XPA-equivalent IPC.

## License

MIT OR Apache-2.0 — see crate metadata.
