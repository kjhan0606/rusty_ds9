<div align="center">

```
                    ╔═══════════════════════════════════════════╗
                    ║                                           ║
                    ║       d s 9 - r u s t   M A N U A L       ║
                    ║                                           ║
                    ║  a Rust + slint port of SAOImage DS9      ║
                    ║                v 0.1                      ║
                    ║                                           ║
                    ╚═══════════════════════════════════════════╝
```

*A pragmatic, OGFinder-style FITS viewer for working astronomers.*

</div>

---

## Contents

1. [Quick start](#1-quick-start)
2. [Core concepts](#2-core-concepts)
3. [Menubar reference](#3-menubar-reference)
4. [Mouse & keyboard](#4-mouse--keyboard)
5. [Working with regions](#5-working-with-regions)
6. [WCS-aware regions](#6-wcs-aware-regions)
7. [Source catalogs](#7-source-catalogs)
8. [Multi-frame, Blink, RGB](#8-multi-frame-blink-rgb)
9. [Filters: smoothing & binning](#9-filters-smoothing--binning)
10. [Analysis tools](#10-analysis-tools)
11. [Saving & printing](#11-saving--printing)
12. [IPC protocol](#12-ipc-protocol)
13. [Cookbook](#13-cookbook)
14. [Troubleshooting](#14-troubleshooting)
15. [Architecture](#15-architecture)

---

## 1. Quick start

```sh
# build (release recommended for large mosaics)
cargo build --release -p ds9-app

# run with no input — open files via the File menu
./target/release/ds9

# open an image (and optionally a region file + catalog) up front
./target/release/ds9 deep_field.fits regions.reg sources.cat
```

> **Note** — the binary is named `ds9` (not `ds9-app`). It's defined in
> `crates/ds9-app/Cargo.toml [[bin]] name = "ds9"`.

| Input class | Extensions                                               |
|-------------|----------------------------------------------------------|
| FITS image  | `.fits`, `.fit`, `.fts`, `.fz` (Rice tile-compressed)    |
| DS9 region  | `.reg` — circle / box / ellipse / annulus / point / line / polygon |
| Catalog     | TSV, whitespace-separated, or SExtractor `ASCII_HEAD`     |

---

## 2. Core concepts

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│   │ Frame 1  │   │ Frame 2  │   │ Frame 3  │   │   ...    │        │
│   │  ┌────┐  │   │  ┌────┐  │   │  ┌────┐  │   │  ┌────┐  │        │
│   │  │FITS│  │   │  │FITS│  │   │  │FITS│  │   │  │FITS│  │        │
│   │  └────┘  │   │  └────┘  │   │  └────┘  │   │  └────┘  │        │
│   │ stretch  │   │ stretch  │   │ stretch  │   │ stretch  │        │
│   │ limits   │   │ limits   │   │ limits   │   │ limits   │        │
│   │ cmap     │   │ cmap     │   │ cmap     │   │ cmap     │        │
│   │ regions  │   │ regions  │   │ regions  │   │ regions  │        │
│   │ catalog  │   │ catalog  │   │ catalog  │   │ catalog  │        │
│   │ zoom/pan │   │ zoom/pan │   │ zoom/pan │   │ zoom/pan │        │
│   │ smooth/  │   │ smooth/  │   │ smooth/  │   │ smooth/  │        │
│   │   bin    │   │   bin    │   │   bin    │   │   bin    │        │
│   └────┬─────┘   └──────────┘   └──────────┘   └──────────┘        │
│        │                                                           │
│        ▼  (active frame drives the canvas)                         │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                       canvas + overlay                     │   │
│   └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

**Frame.** A frame bundles one FITS image with *all* of its display state. Switching frames preserves your view —
nothing is mutated when you flip to the next frame.

**Active frame.** Exactly one frame is "active" at any time. The toolbar's `FRM` cell reads `<active>/<total>`.
Menus that act on an image (Scale, Color, Region, Bin, Analysis, …) act on the active frame.

**Mode.** Below the menubar a small toggle picks one of `pan` / `edit` / `region` / `crosshair`. Mode changes
how the canvas interprets clicks (drop a region in `edit`, pan the image in `pan`, …).

---

## 3. Menubar reference

```
File    Edit    View    Frame    Bin    Zoom    Scale    Color    Region    WCS    Analysis    Catalog    Help
```

### File

| Item            | What it does                                                                                  |
|-----------------|-----------------------------------------------------------------------------------------------|
| `Open…`         | File picker → load FITS into a new frame                                                      |
| `Save Image…`   | PNG export of the current rendered view (smooth/bin applied, full image resolution)           |
| `Save FITS…`    | Minimal `BITPIX=−32` FITS with WCS preserved                                                  |
| `Print…`        | Renders a temp PNG and runs `lpr` on it                                                       |
| `Quit`          | Close the application                                                                         |

### Frame

| Item              | What it does                                                                                |
|-------------------|---------------------------------------------------------------------------------------------|
| `New Frame`       | Open a file dialog and append the chosen FITS as a new frame                                |
| `Delete Frame`    | Drop the active frame; active index falls back to the previous frame                        |
| `Next` / `Previous` | Cycle through frames (wraps)                                                              |
| `Match…`          | Copy the active frame's `view_zoom`, `view_pan_x/y` to **every other** frame                |
| `Blink`           | Toggle a 500 ms timer that auto-cycles `Next`                                               |
| `RGB Composite`   | Render frames 1, 2, 3 as the R, G, B channels of one image (must share dimensions)          |

### Bin

`1` resets to the original. `2 … 32` block-averages the active frame into NxN cells *for display* — coordinates and
WCS are unchanged.

### Zoom

`Zoom In` / `Out` step by ×1.5. `Fit` recomputes the per-axis ratio that fits 800 × 600. `Reset` jumps back to
1×. The fixed levels (`1× … 32×`) snap to that exact zoom without changing pan.

### Scale

Top half — stretches: `linear`, `log`, `sqrt`, `squared`, `asinh`, `sinh`. Bottom half — limits: `zscale` (default,
clips the wildest 2.5 % on each tail) and `minmax`.

### Color

`grey`, `red`, `green`, `blue`, plus `a`, `b`, `bb` (black-body), `heat`, `cool`, `rainbow`, `sls`, `hsv`. The
colorbar strip beside the canvas updates instantly.

### Region

| Item                | What it does                                                         |
|---------------------|----------------------------------------------------------------------|
| `New`               | Drop a sample circle at image center                                 |
| `Load…`             | DS9 `.reg` file picker (WCS-aware if the active frame has WCS)       |
| `Save…`             | Write the current region list to a `.reg` file (image coords)        |
| `Delete Selected`   | Remove the currently selected region                                 |
| `Delete All`        | Clear every region in the active frame                               |
| `Info`              | Status-bar summary (count + selected index)                          |

### WCS

Coordinate-system labels (`FK5`, `ICRS`, `Galactic`) and readout format (`Degrees`, `Sexagesimal`). Currently the
readout follows the FITS header's `RADESYS`; format toggle is reserved.

### Analysis

| Item               | What it does                                                                              |
|--------------------|-------------------------------------------------------------------------------------------|
| `Pixel Table…`     | 5×5 numeric table around the cursor printed to the status bar                             |
| `Statistics…`      | n, min, max, mean, median, σ over finite samples                                          |
| `Histogram…`       | Toggle a floating 128-bin log-scale histogram on the active frame's stretch limits        |
| `Contour Levels…`  | Toggle a 5-level contour overlay (sign-change detection on right/down neighbors)          |
| `Smooth (cycle)`   | Cycle gaussian σ through `0 → 2 → 4 → 8` pixels                                           |
| `Smooth Off`       | Force σ back to 0                                                                         |

### Catalog

| Item     | What it does                                                  |
|----------|---------------------------------------------------------------|
| `Load…`  | Pick a TSV / SExtractor catalog and overlay points on the image |
| `Clear`  | Remove the catalog from the active frame                      |
| `Info`   | Status-bar summary (rows, columns, detected `X/Y` columns)    |

---

## 4. Mouse & keyboard

```
                                     ┌──────────────────┐
                                     │   info bar       │ x, y, value, WCS, FRM
   ┌──────────────────────────────┐  └──────────────────┘
   │                              │
   │           CANVAS             │  hover  → cursor readout
   │                              │  drag   → pan view (when not on a region)
   │                              │  click  → select a catalog source
   │                              │           or drop a region (in `edit`)
   │                              │  press  → grab a region (then drag in `edit`)
   └──────────────────────────────┘
```

| Action                              | How                                                                  |
|-------------------------------------|----------------------------------------------------------------------|
| Pan                                 | Drag on canvas (away from any region)                                |
| Zoom in / out                       | Menu → `Zoom ▸ Zoom In / Out`                                        |
| Snap zoom                           | Menu → `Zoom ▸ 1× / 2× / 4× / 8× / 16×`                              |
| Cursor readout (x, y, value, WCS)   | Hover — info bar updates live                                        |
| Select a region                     | Click the region (any mode)                                          |
| Drag a selected region              | `edit` mode + press-and-drag                                         |
| Drop a new circle                   | `edit` mode + click on empty canvas                                  |
| Pick a catalog source on the image  | Click within ~8 px of the source                                     |
| Pick a catalog source from the list | Click its row in the right-hand catalog table — view recenters       |
| Match all frames to current         | Menu → `Frame ▸ Match…`                                              |
| Blink frames                        | Menu → `Frame ▸ Blink` (toggle)                                      |

---

## 5. Working with regions

A region is a labeled, colored shape stored in image-pixel coordinates (1-based, FITS convention). Internally the
shape is one of:

```rust
enum Shape {
    Circle, Ellipse, Box, Annulus, Polygon, Point, Line, Compass, Text
}
```

### Drop, drag, delete

```
   ┌────────────────────────────┐
   │  1. switch to `edit` mode  │
   ├────────────────────────────┤
   │  2. click empty area       │  ← drops a 6-px circle at the click
   │  3. click the new region   │  ← marks it selected
   │  4. press-and-drag         │  ← moves the region in image space
   │  5. Region ▸ Delete Selected
   └────────────────────────────┘
```

The selected region is drawn slightly thicker so you can confirm it before deleting.

### Region file I/O

Files use DS9's familiar `.reg` text format. Without a WCS we read `image` coords; with a WCS we additionally accept
`fk5`, `icrs`, and `galactic` lines (see [§ 6](#6-wcs-aware-regions)). On save we always emit `image` so the file
round-trips losslessly through the parser.

```
# Region file format: DS9 ds9-rust
global color=green width=1 select=1 …
image
circle(1024.5, 1536.0, 12.0)
box(2048.0, 1024.0, 80.0, 40.0, 30.0)
annulus(512.0, 512.0, 6.0, 12.0)
```

---

## 6. WCS-aware regions

When the active frame's FITS header carries a tangent-plane WCS (`CRPIX1/2`, `CRVAL1/2`, and either a `CD` matrix or
`PC`/`CDELT`), `Region ▸ Load…` will project sky coordinates to pixels for you.

### Coordinate-system tokens

```
fk5            #  RA / Dec, J2000-equivalent for our precision
icrs           #  same — treated as FK5
fk4 / b1950    #  parsed as fk5 (no precession applied)
j2000          #  parsed as fk5
galactic       #  l, b → ICRS via IAU 1958 rotation, then projected
image          #  pixel coords (the default if no token seen)
physical       #  treated as image
```

### Coordinate formats

| Form                       | Example                  | Meaning                      |
|----------------------------|--------------------------|------------------------------|
| Sexagesimal hours (RA / l) | `12:34:56.78`            | hh:mm:ss.s → degrees × 15    |
| Sexagesimal degrees (Dec)  | `+12:34:56.7`, `-04:30`  | dd:mm:ss.s                   |
| Decimal degrees            | `188.7366`, `188.7366d`  | bare or with `d` suffix      |
| Decimal hours              | `12.583h`                | × 15 → degrees                |

### Size suffixes

For sky-coord shapes the size arguments take a unit suffix:

| Suffix | Unit     | Pixel conversion                        |
|--------|----------|-----------------------------------------|
| `"`    | arcsec   | size × pix_per_arcsec                    |
| `'`    | arcmin   | size × 60 × pix_per_arcsec               |
| `d`    | degrees  | size × 3600 × pix_per_arcsec             |
| _none_ | arcsec   | DS9 default                              |

The `pix_per_arcsec` factor is derived from `√|det(CD)|`.

### Example

A region file for an HST/ACS field:

```
fk5
circle(03:32:38.621, -27:46:29.36, 6")
ellipse(03:32:39.10, -27:46:35.0, 4.5", 3.0", 35)
polygon(03:32:38.0, -27:46:25.0, 03:32:38.5, -27:46:25.5, 03:32:38.3, -27:46:24.5)
galactic
circle(220.18d, -52.34d, 10")
```

Inline coord systems on a single line work too:

```
fk5; circle(03:32:38.621, -27:46:29.36, 6")
image; box(1024, 1536, 40, 40, 0)
```

---

## 7. Source catalogs

Three accepted formats:

```
TSV (with header)
    NUMBER<TAB>X_IMAGE<TAB>Y_IMAGE<TAB>MAG_AUTO
    1<TAB>123.5<TAB>456.7<TAB>22.34
    ...

SExtractor ASCII_HEAD
    #   1 NUMBER         Running object number
    #   2 X_IMAGE        Object position along x [pixel]
    #   3 Y_IMAGE        Object position along y [pixel]
    #   4 MAG_AUTO       Kron-like magnitude
       1   100.5  200.3  18.5
       2   300.0  150.7  19.2

Bare whitespace (no header → synthesizes col1, col2, …)
    100.5  200.3  18.5
    300.0  150.7  19.2
```

The viewer auto-detects which one you've thrown at it. It picks `X_IMAGE` / `XWIN_IMAGE` / `X` / `XCEN` (with `Y`
counterparts) for the points and one of `MAG_AUTO` / `MAG_BEST` / `MAG_APER` / `MAG_PSF` / `MAG` for the magnitude
column shown in the table.

### Click-to-select

Clicking on an image source within ~8 px highlights the matching catalog row. Clicking a row in the table recenters
the canvas on that source. Selected sources render as a slightly larger amber dot.

---

## 8. Multi-frame, Blink, RGB

### Loading frames

Each `File ▸ Open…` (or `Frame ▸ New Frame`) appends a new frame and makes it active. The previous frame's
zoom/pan is saved before the switch, and the new frame's saved view is restored after — switching back and forth is
lossless.

### Match

`Frame ▸ Match…` copies the active frame's `view_zoom` / `view_pan_x` / `view_pan_y` to every other frame. Useful
when you've zoomed into the same target on multiple bands and want them all aligned for blink.

### Blink

`Frame ▸ Blink` toggles a 500 ms `slint::Timer` that fires `Frame ▸ Next` until you toggle it off again. Combine
with `Match…` to A/B/C compare aligned exposures.

### RGB Composite

```
   frame 1 (luminance) ─┐
                        ├──→  R G B image  (replaces the canvas image)
   frame 2 (luminance) ─┤
                        │
   frame 3 (luminance) ─┘
```

`Frame ▸ RGB Composite` builds a single RGBA image from the first three frames' rendered (stretched, colormapped)
luminance and pushes it directly to the canvas. Frames must share `width × height`.

> **Tip.** Stretch each band independently (e.g. `log` for the long-wavelength frame, `linear` for the short) before
> running `RGB Composite` — the composite uses each frame's *current* render so you can preview by stretching a
> single band, then re-running the composite.

---

## 9. Filters: smoothing & binning

Both filters apply *before* the stretch / colormap pipeline and are persisted per-frame.

### Smoothing

```
   raw FITS  ──[ gaussian σ ]──  smoothed image  ──[ stretch + cmap ]──  canvas
```

`Analysis ▸ Smooth (cycle)` walks σ through `0 → 2 → 4 → 8 → 0` pixels. The kernel radius is `⌈3σ⌉` and NaNs are
treated as missing samples (skipped from both the convolution sum and the weight, so masked regions don't bleed).

### Binning

```
   raw FITS  ──[ NxN block average ]──  chunky image  ──[ stretch + cmap ]──  canvas
```

`Bin ▸ N` block-averages the image into NxN cells, then **expands back to the original size** so the WCS, cursor
readout, and click-to-select coordinates all stay valid. The result is a chunky-pixel preview useful for picking out
faint extended structure.

> **Order of operations.** When both filters are on, `bin_average` runs first, then `smooth_gaussian`. Smooth-then-
> bin would usually waste the smoothing.

---

## 10. Analysis tools

### Histogram

`Analysis ▸ Histogram…` toggles a floating 380 × 230 panel in the bottom-left of the canvas. It plots a 128-bin
log-scale histogram clipped to the active frame's stretch limits.

```
         ┃   bar height = log(count + 1) / log(max + 1)
         ┃   bins = 128 between [low, high]
   ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃
   ╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋
   low                                       high
```

### Contour levels

`Analysis ▸ Contour Levels…` toggles a 5-color overlay drawn at the same canvas geometry as the image. The five
levels are evenly spaced between the active frame's `low` and `high`, and a pixel is painted in that level's color
whenever its value crosses the level when compared to either its right or down neighbor.

### Statistics & pixel table

| Tool         | Output                                                                |
|--------------|-----------------------------------------------------------------------|
| Statistics…  | `n, min, max, mean, median, σ` over finite samples in the status bar |
| Pixel Table… | 5 × 5 numeric grid of values centered on the cursor                  |

---

## 11. Saving & printing

### Save Image (PNG)

`File ▸ Save Image…` writes a **full-resolution** PNG (post-stretch, post-cmap, post-smooth/bin). Use this for
publication figures.

### Save FITS

`File ▸ Save FITS…` writes a minimal FITS:

```
SIMPLE  =                    T
BITPIX  =                  -32
NAXIS   =                    2
NAXIS1  = <image width>
NAXIS2  = <image height>
BSCALE  = 1.0
BZERO   = 0.0
[CTYPE1, CTYPE2, CRPIX1/2, CRVAL1/2, CD1_1..CD2_2, RADESYS]   ← if the source had WCS
END
<width × height × f32 BE>
```

The data block is the raw `f32` floats from the loaded image (smooth/bin are *not* baked in — this exports the
underlying data).

### Print

`File ▸ Print…` renders a temp PNG and runs `lpr` on it. If `lpr` isn't on `$PATH` the temp file path is reported in
the status bar so you can hand it to your printing pipeline of choice.

---

## 12. IPC protocol

On startup, ds9-rust opens a Unix-domain socket at:

```
$XDG_RUNTIME_DIR/ds9-rust-$USER.sock      # if XDG_RUNTIME_DIR is set
${TMPDIR:-/tmp}/ds9-rust-$USER.sock       # otherwise
```

The socket path is also printed to stderr at startup and shown in the status bar.

### Verb table

| Verb form                    | Effect                                                            |
|------------------------------|-------------------------------------------------------------------|
| `quit`                       | Quit the application                                              |
| `frame next`                 | `Frame ▸ Next`                                                    |
| `frame previous`             | `Frame ▸ Previous`                                                |
| `frame N`                    | Switch to frame `N` (1-based)                                     |
| `scale linear|log|sqrt|...`  | `Scale ▸ <stretch>` (also accepts `minmax`, `zscale`)             |
| `cmap NAME`                  | `Color ▸ NAME`                                                    |
| `bin N`                      | `Bin ▸ N`                                                         |
| `zoom in|out|fit`            | `Zoom ▸ ...`                                                      |
| `zoom N`                     | Snap to a literal zoom level                                      |
| `region load PATH`           | Load `.reg` file                                                  |
| `region save PATH`           | Write current regions to file                                     |
| `file open PATH`             | Load FITS into a new frame                                        |
| `save png PATH`              | PNG export of the current view                                     |
| `save fits PATH`             | FITS export of the underlying data                                 |
| `value`                      | Print the value at the last cursor position                       |
| `help`                       | List all commands in one line                                     |

Replies are one line each: `ok`, `ok <message>`, or `err <reason>`.

### Examples

```sh
SOCK=${XDG_RUNTIME_DIR:-/tmp}/ds9-rust-$USER.sock

echo "frame next"           | nc -U "$SOCK"
echo "scale log"            | nc -U "$SOCK"
echo "cmap heat"            | nc -U "$SOCK"
echo "bin 4"                | nc -U "$SOCK"
echo "zoom 2"               | nc -U "$SOCK"
echo "region load /tmp/x.reg" | nc -U "$SOCK"
echo "save png /tmp/out.png" | nc -U "$SOCK"
echo "value"                | nc -U "$SOCK"
```

A multi-step pipeline (Python):

```python
import socket, os
sock = socket.socket(socket.AF_UNIX)
sock.connect(f"{os.environ['XDG_RUNTIME_DIR']}/ds9-rust-{os.environ['USER']}.sock")
sock.sendall(b"file open /tmp/deep.fits\nscale asinh\ncmap heat\nzoom fit\nsave png /tmp/preview.png\n")
print(sock.recv(4096).decode())
```

---

## 13. Cookbook

### A. Inspect a survey image with a SExtractor catalog

```sh
./target/release/ds9 deep_field.fits sources.cat
```

1. The image opens with `zscale` + `linear` stretch and the catalog over-plotted in amber.
2. Click any source → the catalog table on the right scrolls to the matching row.
3. Click a row in the table → canvas recenters.

### B. Edit a region file by hand, then iterate

```sh
./target/release/ds9 deep_field.fits old.reg
```

1. Switch to `edit` mode in the toolbar.
2. Click empty space to drop new circles; press-and-drag existing regions.
3. `Region ▸ Save…` → `new.reg`.

### C. Quick RGB color image from three bands

```sh
./target/release/ds9 r.fits
# then File ▸ Open… for g.fits and i.fits
```

1. Stretch each band individually (`Scale ▸ asinh` is usually a good starting point).
2. `Frame ▸ Match…` to align zoom/pan.
3. `Frame ▸ RGB Composite` — frames 1, 2, 3 → R, G, B.
4. `File ▸ Save Image…` → publication-ready PNG.

### D. Smoothing for low-S/N source detection

1. `Analysis ▸ Smooth (cycle)` — bumps σ to 2 px.
2. `Analysis ▸ Contour Levels…` overlays five contours on the smoothed image.
3. Cycle σ to 4 / 8 px to see how features survive at lower spatial frequencies.

### E. Drive the viewer from a script

```sh
#!/bin/bash
SOCK=${XDG_RUNTIME_DIR:-/tmp}/ds9-rust-$USER.sock
for stretch in linear log asinh sqrt; do
    echo "scale $stretch"               | nc -U "$SOCK"
    echo "save png /tmp/$stretch.png"   | nc -U "$SOCK"
done
```

---

## 14. Troubleshooting

| Symptom                                          | Likely cause / fix                                                  |
|--------------------------------------------------|---------------------------------------------------------------------|
| Image is uniform grey                            | `zscale` pulled the limits to a degenerate range — try `Scale ▸ minmax` |
| Sky-coord regions don't appear                   | The frame has no WCS (status bar mentions this on load). Open a FITS that carries `CRPIX/CRVAL/CD`. |
| `.fz` file looks blocky / wrong values           | Make sure the workspace `[patch.crates-io] fitsrs = …` entry is intact (fixes f32 unquantize precision) |
| `Print…` says "lpr unavailable"                  | Use the temp PNG path it reported and pipe it to your own printing tool |
| Catalog points off by one pixel                  | Catalog was written in 0-based image coords; ds9-rust expects 1-based FITS coords |
| IPC `nc -U` hangs                                | The protocol is request/response per line — close stdin (`echo "..." \| nc -q1 -U`) or use `socat - UNIX-CONNECT:$SOCK` |
| Blink doesn't advance                            | You need ≥ 2 frames loaded; `frame N` in IPC is 1-based                |

---

## 15. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                ds9-app                                      │
│                                                                             │
│   ┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐      │
│   │  slint UI      │←─→│  State / Frame     │←─→│  IPC server thread │      │
│   │  (declarative) │   │  (Rc<RefCell<…>>)  │   │  (UnixListener)    │      │
│   └───────┬────────┘   └──────┬─────────────┘   └────────────────────┘      │
│           │                   │                                             │
│           ▼                   ▼                                             │
│      menu actions      render_image / refresh_view                          │
│                                                                             │
└────────────┬───────────────────────────┬──────────────────────────┬─────────┘
             │                           │                          │
             ▼                           ▼                          ▼
       ┌──────────┐                 ┌──────────┐               ┌────────────┐
       │ ds9-fits │                 │ds9-image │               │ ds9-marker │
       │  - load  │                 │ - bin    │               │ - .reg I/O │
       │  - WCS   │                 │ - smooth │               │ - WCS proj │
       │ pix↔world│                 │ - render │               │            │
       └──────────┘                 └──────────┘               └────────────┘
                                          │
                                          ▼
                                    ┌──────────────┐
                                    │ ds9-catalog  │
                                    │ TSV / SExt   │
                                    └──────────────┘
```

| Crate         | Responsibility                                                                           |
|---------------|-------------------------------------------------------------------------------------------|
| `ds9-app`     | slint UI shell, menubar, frame/state management, IPC, save/print, blink, RGB composite   |
| `ds9-fits`    | FITS I/O (BITPIX 8/16/32/-32/-64, BSCALE/BZERO, BLANK, tile-compressed via fitsrs), TAN WCS forward + inverse, J2000 Galactic-to-ICRS rotation |
| `ds9-image`   | Stretch / limits, colormaps, gaussian smoothing, NxN binning, RGBA renderer              |
| `ds9-marker`  | DS9 `.reg` parser/writer, region/marker model, WCS-aware coord projection                |
| `ds9-catalog` | TSV / SExtractor `ASCII_HEAD` parser, column lookup, sort                                |
| `vendor/fitsrs` | Local fitsrs patch (f64 unquantize for `.fz` files where ZZERO is huge)                |

---

<div align="center">

*Made with rust 1.85, slint 1.16, and a healthy respect for SAOImage DS9.*

</div>
