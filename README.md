# ds9-rust

A Rust + [slint](https://slint.dev) port of [SAOImage DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9), the
astronomical FITS viewer originally written in Tcl/Tk by the Smithsonian Astrophysical Observatory.

This is a clean rewrite focused on the OGFinder-style workflow: open a FITS image (or cube), overlay a SExtractor or
VizieR catalog, draw and edit DS9 regions, and switch between multiple frames — all scriptable via an XPA-style line
protocol on both Unix-domain socket and TCP.

> **Full manual:** [`docs/manual.pdf`](docs/manual.pdf) — typeset with LaTeX (xelatex). Source: [`docs/manual.tex`](docs/manual.tex).

## Build

```sh
cargo build -p ds9-app --release
```

Requires Rust ≥ 1.85. The vendored `fitsrs` patch in `crates/vendor/fitsrs` fixes an `f32` precision-loss bug in tile-
compressed (`.fz`) DES-style mosaics — keep the workspace `[patch.crates-io]` in place.

To rebuild the manual:

```sh
cd docs && xelatex manual.tex && xelatex manual.tex   # two passes for the TOC
```

## Run

```sh
# empty session — open files via File ▸ Open…
./target/release/ds9

# preload an image (and optionally a region file + catalog)
./target/release/ds9 image.fits regions.reg sources.cat
```

> The binary is named `ds9` (not `ds9-app`) — defined in `crates/ds9-app/Cargo.toml [[bin]]`.

Supported inputs:

| Type    | Formats                                                                  |
|---------|--------------------------------------------------------------------------|
| Image   | `.fits`, `.fit`, `.fts`, `.fz` (Rice tile-compressed); NAXIS=2 or NAXIS=3 |
| Region  | DS9 `.reg` — `circle / box / ellipse / annulus / point / line / polygon` in `image` or WCS coords (`fk5`, `icrs`, `galactic`) |
| Catalog | TSV, whitespace, CSV, SExtractor ASCII_HEAD, minimal VOTable             |

## Quick guide

### Mouse / keyboard

| Action                        | How                                                                     |
|-------------------------------|-------------------------------------------------------------------------|
| Pan                           | Drag on the canvas (when not over a region)                            |
| Zoom                          | `Zoom ▸ Zoom In / Out / Fit / Reset` or pick a fixed level (`1× … 16×`)|
| Cursor readout (x/y/value/WCS)| Hover — shown in the info bar                                          |
| Select a region               | Click on it (any mode)                                                  |
| Drag a region                 | `edit` mode + press-and-drag on the region                              |
| Drop a new circle             | `edit` mode + click on empty canvas                                     |
| Pick a catalog source         | Click near it on the image, **or** click its row in the table          |
| Crosshair                     | `crosshair` mode + click — pinned in WCS, follows active frame         |

Mode toggles live next to the info bar (`pan` / `edit` / `region` / `crosshair`).

### Menus (highlights)

| Menu      | Highlights                                                                                         |
|-----------|----------------------------------------------------------------------------------------------------|
| File      | `Open…`, `Save Image…` (PNG), `Save FITS…`, `Save TIFF…`, `Save EPS…`, `Print…` (via `lpr`), **`SAMP Send Image / VOTable`**, `Quit` |
| Edit      | **`Crop` / `Reset Crop`**, **`Preferences…`**                                                       |
| View      | Panner, Magnifier, Coordinate Grid, Crosshair, Info Panel, Buttons, Colorbar (toggles)             |
| Frame     | `New / Delete`, `Next / Previous`, `Match…`, `Blink`, `RGB Composite`, `HDU Movie`, **`Mosaic WCS`**, **`Tile Frames`**, **`3D Slice…`/`Max Intensity`/`Sum`/`Mean`**, `Rotate / Flip`, `Lock Zoom/Pan/Color/Scale` |
| Bin       | `1 / 2 / 4 / 8 / 16 / 32` factor + `Average / Sum / Sub-sample` reduction                          |
| Zoom      | In / Out / Fit / Reset, fixed `1× / 2× / 4× / 8× / 16×`                                            |
| Scale     | Stretch (`linear`, `log`, `sqrt`, `squared`, `asinh`, `sinh`) + limits (`zscale`, `minmax`)        |
| Color     | `grey`, `red`, `green`, `blue`, `a`, `b`, `bb`, `heat`, `cool`, `rainbow`, `sls`, `hsv` + custom LUT |
| Region    | `New`, `Load…`, `Save…`, `Delete Selected / All`, `Info`, **`Property…`** (editor)                 |
| Catalog   | `Load…`, `Clear`, `Run SExtractor…`, **`Online Query…`** (Sesame / VizieR / NED), `Info`           |
| Analysis  | `Pixel Table…`, `Statistics…`, `Histogram…`, `Contour Levels…`, `Smooth (cycle / kind)`, **`Centroid`**, **`Radial Profile`**, **`Projection`** |

### Multi-frame, mosaic, tile, blink

Each loaded image is its own *frame* — independent stretch, limits, colormap, regions, catalog, zoom, and pan. The
`FRM` cell shows `<active>/<total>`.

- **Match** broadcasts the active frame's view to siblings; the `Lock *` toggles pick which channels.
- **Blink** cycles every 500 ms.
- **RGB Composite** packs frames 1–3 into the R/G/B channels of a new frame.
- **Mosaic WCS** WCS-reprojects every open frame into one (output-driven nearest-neighbour, overlap averaged).
- **Tile Frames** shows all frames in a √n grid; click a tile to switch.

### 3-D cube rendering

When a NAXIS=3 cube is loaded, `Frame ▸ 3D …` projects the plane stack:

| Mode             | What                                                              |
|------------------|-------------------------------------------------------------------|
| `3D Slice…`      | one click advances to the next plane (wraps); IPC `3d slice N`    |
| `3D Max Intensity` | per-pixel max along the cube axis (NaN-safe)                    |
| `3D Sum`         | per-pixel sum (NaN-safe)                                          |
| `3D Mean`        | per-pixel mean (NaN-safe)                                         |

The chosen mode is sticky on the frame; smoothing, binning, stretch, and cmap all run downstream of the projection.

### Filters

`Analysis ▸ Smooth (cycle)` cycles Gaussian σ ∈ {0, 2, 4, 8} px; `Smooth Kind…` picks Gaussian / Boxcar / Median.
`Bin ▸ N` block-bins the displayed image NxN with the chosen reduction (`Average / Sum / Sub-sample`). All filters are
visualisation-only — coordinates and WCS are never modified.

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

The IPC verb `sextractor` runs the same code path.

### Online catalog query

`Catalog ▸ Online Query…` opens a small panel:

- **Resolve** — SIMBAD/Sesame name → (RA, Dec); recenters and drops a crosshair.
- **VizieR** — cone search around the cursor / crosshair (radius in arcmin); result loads as a VOTable catalog.
- **NED** — `objsearch` cone search (returns a bar-separated ASCII table).

All requests use HTTPS via `ureq + rustls` (no system OpenSSL).

### SAMP messaging

`File ▸ SAMP Send Image / VOTable` broadcasts the current FITS path (or a saved VOTable) to other SAMP clients
running on the same desktop — TOPCAT, Aladin, etc. The XML-RPC client is hand-rolled (no SOAP/glib dep). Outbound only.

### IPC (XPA-flavoured)

Two transports start at launch:

- Unix-domain socket: `$XDG_RUNTIME_DIR/ds9-rust-$USER.sock` (or `/tmp/...`).
- TCP loopback: a dynamic port on `127.0.0.1`.

Discovery file at `~/.ds9-rust/xpa.info`:

```
pid=12345
unix=/run/user/1000/ds9-rust-alice.sock
tcp=54861
```

Pipe one command per line:

```sh
SOCK=$(awk -F= '/^unix=/{print $2}' ~/.ds9-rust/xpa.info)

echo "frame next"             | nc -U "$SOCK"
echo "scale log"              | nc -U "$SOCK"
echo "cmap heat"              | nc -U "$SOCK"
echo "region load /tmp/x.reg" | nc -U "$SOCK"
echo "save png /tmp/out.png"  | nc -U "$SOCK"
echo "3d max"                 | nc -U "$SOCK"
echo "3d slice 12"            | nc -U "$SOCK"
echo "value"                  | nc -U "$SOCK"
echo "help"                   | nc -U "$SOCK"
```

Supported verbs: `quit`, `xpaaccess`, `version`, `mode M`, `frame next|prev|N`, `frame new|delete|mosaic|tile`,
`scale linear|log|...|limits LO HI`, `cmap NAME`, `bin N`, `zoom in|out|fit|reset|N`, `pan to X Y`,
`region load|save PATH`, `file open PATH`, `save png|fits|tiff|eps PATH`, `value`, `sextractor`,
`samp image|votable PATH`, `hdu next|list|N`, `movie on|off`, `3d max|sum|mean|slice N|depth`, `help`.

> TCP listens only on `127.0.0.1`. There is no auth on the IPC channel — don't expose it on a public interface.

### Preferences

`Edit ▸ Preferences…` toggles a panel for default cmap / stretch / limits / smoothing / bin / blink-ms.
**Apply** pushes onto the active frame; **Save** persists to `~/.config/ds9-rust/prefs.toml`. The prefs file is loaded
automatically at startup; new frames inherit those defaults.

## Workspace layout

```
crates/
  ds9-app      — slint UI shell, event wiring, menubar, frame/state management,
                 mosaic + tile + 3D + SAMP + online-query + IPC dispatch
  ds9-fits     — FITS I/O (BITPIX 8/16/32/-32/-64, BSCALE/BZERO, BLANK,
                 tile-compressed via fitsrs, NAXIS=3 cube loader) + minimal WCS
  ds9-image    — stretches (linear/log/sqrt/squared/asinh/sinh),
                 limits (zscale/minmax), colormaps + custom LUT,
                 RGBA render, smoothing (Gaussian/Boxcar/Median),
                 binning (Average/Sum/Subsample)
  ds9-marker   — DS9 .reg parser/writer + region/marker model
  ds9-catalog  — TSV / whitespace / CSV / SExtractor / VOTable parsers
  vendor/fitsrs — local fitsrs patch (f64 unquantize for `.fz`)
docs/
  manual.tex   — fancy LaTeX manual source (xelatex)
  manual.pdf   — built manual (~13 pages)
```

## Status

Implemented (Batch A → C complete):

- Single- and multi-extension FITS load incl. `.fz`; NAXIS=3 cube loader.
- All seven DS9 stretches + standard colormaps + 256-stop custom LUT.
- DS9 `.reg` round-trip in `image` and WCS coords (`fk5`, `icrs`, `galactic`, sexagesimal or decimal).
- Region editing — click select, drag in `edit` mode, **Property… editor**, delete selected / all.
- Multiple frames with per-frame view state, **Match**, **Blink**, **RGB Composite**, **Mosaic WCS**, **Tile Frames**.
- **3-D cube rendering** — Slice / Max Intensity / Sum / Mean projections per frame.
- **Smoothing** (Gaussian / Boxcar / Median) and **binning** (Average / Sum / Sub-sample) per frame.
- **HDU Navigator** + **HDU Movie** (auto-cycle every 800 ms).
- **Catalog** load + overlay (SExtractor / TSV / CSV / VOTable / whitespace), **SExtractor wrapper**,
  **Online Query** (Sesame / VizieR / NED via HTTPS).
- **Analysis** — Statistics, Pixel Table, Histogram, Contours, **Centroid**, **Radial Profile**, **Projection**.
- **Crop** + **Reset Crop**; **Preferences** persisted to `~/.config/ds9-rust/prefs.toml`.
- Exports — **PNG**, **FITS** (BITPIX=−32 + WCS), **TIFF** (in-house), **EPS**, **Print** via `lpr`.
- **IPC** — XPA-flavoured line protocol on both **Unix socket** and **TCP loopback**, with a
  `~/.ds9-rust/xpa.info` discovery file.
- **SAMP** outbound (XML-RPC over the standard `~/.samp` lockfile) — `image.load.fits` + `table.load.votable`.

Not (yet) ported:

- True XPA wire protocol (xpans, binary envelope, ACL, FIFO duplex). The line protocol covers most of the verb space.
- Inbound SAMP subscription; macros; saved frame contours.
- Slint print dialog (paper size / orientation); only `lpr`-pipe is wired.
- 3-D OpenGL volume rendering (we project to 2-D; the existing render path only handles 2-D textures).
- Tcl/Tk parity for niche dialogs (analysis-tool plug-ins, slice/blink-rate UI, tagged region groups).

## License

MIT OR Apache-2.0 — see crate metadata.
