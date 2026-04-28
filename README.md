# ds9-rust

A Rust + [slint](https://slint.dev) port of [SAOImage DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9), the
astronomical FITS viewer originally written in Tcl/Tk by the Smithsonian Astrophysical Observatory.

This is a clean rewrite focused on the OGFinder-style workflow: open a FITS image, overlay a SExtractor catalog, draw and
edit DS9 regions, and switch between multiple frames.

> **Full manual:** [`docs/manual.md`](docs/manual.md) — menubar reference, WCS-region syntax, IPC protocol, cookbook.

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
| File      | `Open…`, `Save Image…` (PNG of the current view), `Save FITS…` (basic FITS export), `Print…` (sends a PNG via `lpr`), `Quit` |
| Frame     | `New Frame`, `Delete Frame`, `Next` / `Previous`, `Match…` (sync zoom/pan to all frames), `Blink` (cycle every 500 ms), `RGB Composite` (frames 1–3 → R/G/B) |
| Bin       | `1 / 2 / 4 / 8 / 16 / 32` block-average for the active frame                     |
| Zoom      | In / Out / Fit / Reset, fixed `1× / 2× / 4× / 8× / 16× / 32×`                    |
| Scale     | Stretch (`linear`, `log`, `sqrt`, `squared`, `asinh`, `sinh`) + limits (`zscale`, `minmax`) |
| Color     | Colormaps — `grey`, `red`, `green`, `blue`, `a`, `b`, `bb`, `heat`, `cool`, `rainbow`, `sls`, `hsv` |
| Region    | `New`, `Load…`, `Save…`, `Delete Selected`, `Delete All`, `Info`                |
| Catalog   | `Load…`, `Clear`, `Run SExtractor…` (external `source-extractor` wrapper), `Info` |
| Analysis  | `Pixel Table…`, `Statistics…`, `Histogram…`, `Contour Levels…`, `Smooth (cycle)` (σ ∈ {0, 2, 4, 8} px), `Smooth Off` |

### Multi-frame

Each loaded image is its own *frame* — independent stretch, limits, colormap, regions, catalog, zoom, and pan. `Frame ▸
New Frame` opens a file dialog and pushes a new frame; `Next` / `Previous` cycles through them; the per-frame view state
is restored losslessly on switch. The `FRM` cell in the info bar shows `<active>/<total>`.

### Histogram + Contours

`Analysis ▸ Histogram…` toggles a floating 128-bin log-scale histogram (computed against the active frame's stretch
limits). `Analysis ▸ Contour Levels…` toggles a 5-level contour overlay (sign-change detection between zscale low/high)
drawn on top of the image at the same canvas geometry. Both follow the active frame and refresh when re-opened.

### Smoothing & binning

`Analysis ▸ Smooth (cycle)` cycles the active frame's gaussian σ through `0 → 2 → 4 → 8` pixels; `Smooth Off` resets to
0\. `Bin ▸ N` chunkifies the displayed image into NxN block averages (visualization-only; coordinates and WCS are
unchanged). Both filters are applied before stretch/colormap on every refresh and are persisted per-frame.

### WCS regions

DS9 region files (`fk5`, `icrs`, `galactic`) load correctly when the active frame has a WCS — coordinates may be
sexagesimal (`12:34:56.7`) or decimal degrees, and sizes use DS9's standard `"`/`'`/`d` suffixes. Without a WCS the
parser falls back to `image` semantics and the status bar warns you.

### SExtractor wrapper

`Catalog ▸ Run SExtractor…` shells out to the external SExtractor binary
(`source-extractor`, `sextractor`, or `sex`, in that PATH order) on the active frame's source FITS, parses the
resulting `ASCII_HEAD` catalog, and overlays it on the image. Defaults are `DETECT_THRESH=1.5`, `DETECT_MINAREA=5`,
`BACK_SIZE=64`. Override any SExtractor key via the `SEXTRACTOR_OPTS` environment variable, e.g.:

```sh
SEXTRACTOR_OPTS="-DETECT_THRESH 3.0 -DEBLEND_MINCONT 0.0001" ./target/release/ds9 image.fits
```

The IPC protocol exposes the same action as the `sextractor` verb.

### IPC (XPA-equivalent)

On startup the app opens a Unix-domain socket at `$XDG_RUNTIME_DIR/ds9-rust-$USER.sock` (or `/tmp/...`) and prints the
path to stderr. Pipe one command per line:

```sh
echo "frame next"            | nc -U /run/user/1000/ds9-rust-$USER.sock
echo "scale log"             | nc -U /run/user/1000/ds9-rust-$USER.sock
echo "cmap heat"             | nc -U /run/user/1000/ds9-rust-$USER.sock
echo "region load /tmp/x.reg"| nc -U /run/user/1000/ds9-rust-$USER.sock
echo "save png /tmp/out.png" | nc -U /run/user/1000/ds9-rust-$USER.sock
echo "value"                 | nc -U /run/user/1000/ds9-rust-$USER.sock
echo "help"                  | nc -U /run/user/1000/ds9-rust-$USER.sock
```

Supported verbs: `quit`, `frame next|previous|N`, `scale linear|log|...`, `cmap NAME`, `bin N`, `zoom in|out|fit|N`,
`region load|save PATH`, `file open PATH`, `save png|fits PATH`, `value`, `help`.

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
- DS9 `.reg` round-trip (image coords) **and WCS-coord regions** (`fk5`, `icrs`, `galactic`, sexagesimal or decimal).
- SExtractor catalog overlay, click-to-select, table-row recentering.
- Region editing — click select, drag in `edit` mode, delete selected / all.
- Multiple frames with per-frame view state, **Match**, **Blink**, **RGB Composite** (frames 1–3).
- **Smoothing** (gaussian σ ∈ {0, 2, 4, 8} px) and **binning** (NxN block-average) per frame.
- Statistics, pixel table, histogram, contour levels.
- **Save Image** (PNG of current view) and **Save FITS** (basic BITPIX=−32 export with WCS preserved).
- **Print** via `lpr` after rendering to a temp PNG.
- **Unix-socket IPC** with a small DS9-style verb language (`frame next`, `scale log`, `region load`, `save png`, …).
- **SExtractor wrapper** — `Catalog ▸ Run SExtractor…` (or IPC verb `sextractor`) runs the external `source-extractor` binary on the active frame's source FITS and overlays the resulting catalog.

Not yet ported (open candidates):

- Full XPA wire protocol (we ship a JSON-line / verb-line socket instead).
- DS9 macros, slice / blink-rate UI, frame contours saved to file.
- Print dialog with paper-size / orientation; only `lpr`-pipe is wired.
- Smoothing kernels other than gaussian (boxcar, top-hat, elliptical).

## License

MIT OR Apache-2.0 — see crate metadata.
