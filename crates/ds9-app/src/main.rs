use anyhow::Result;
use ds9_fits::FitsImage;
use ds9_image::{Colormap, Limits, Orientation, Stretch};
use ds9_catalog::Catalog;
use ds9_marker::{Marker, Shape as MShape};
use slint::{ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::cell::RefCell;
use std::env;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Smoothing kernel choice. Gaussian is parameterised by `sigma`; Boxcar /
/// Median by an odd window size in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SmoothKind {
    Gaussian { sigma: f32 },
    Boxcar   { n: u32 },
    Median   { n: u32 },
}

impl Default for SmoothKind {
    fn default() -> Self { SmoothKind::Gaussian { sigma: 0.0 } }
}

impl SmoothKind {
    fn label(self) -> String {
        match self {
            SmoothKind::Gaussian { sigma } => {
                if sigma <= 0.0 { "off".into() } else { format!("gaussian σ={sigma:.1}") }
            }
            SmoothKind::Boxcar { n } => {
                if n <= 1 { "off".into() } else { format!("boxcar {n}×{n}") }
            }
            SmoothKind::Median { n } => {
                if n <= 1 { "off".into() } else { format!("median {n}×{n}") }
            }
        }
    }
}

/// Block-bin reduction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum BinMode {
    #[default]
    Average,
    Sum,
    Subsample,
}

impl BinMode {
    fn label(self) -> &'static str {
        match self {
            BinMode::Average   => "average",
            BinMode::Sum       => "sum",
            BinMode::Subsample => "sub-sample",
        }
    }
}

slint::include_modules!();

// ---------------------------------------------------------------- state --

#[derive(Clone, Copy, Default)]
enum LimitsMode {
    #[default]
    Zscale,
    MinMax,
    #[allow(dead_code)]
    User { low: f32, high: f32 },
}

/// One image plus its display state. DS9 calls these "frames" — a single
/// session can hold many of them and switch between them.
struct Frame {
    fits: FitsImage,
    name: String,
    /// Path on disk this frame was loaded from (None for synthetic / RGB / etc).
    /// External tools (SExtractor wrapper, …) need a real file to read.
    source_path: Option<PathBuf>,
    stretch: Stretch,
    limits_mode: LimitsMode,
    cmap: Colormap,
    markers: Vec<Marker>,
    catalog: Option<Catalog>,
    selected_catalog: Option<usize>,
    selected_marker: Option<usize>,
    /// Per-frame view state — switching frames restores these.
    view_zoom: f32,
    view_pan_x: f32,
    view_pan_y: f32,
    /// Gaussian-smooth kernel σ in pixels (0 = off). Kept for IPC back-compat
    /// with the cycle-through `smooth (cycle)` action; the kernel choice itself
    /// lives on `smooth_kind`.
    smooth_sigma: f32,
    /// Active smoothing kernel (Gaussian / Boxcar / Median).
    smooth_kind: SmoothKind,
    /// Block-bin factor (1 = off, 2/4/8/16/32 chunkify).
    bin_factor: u32,
    /// Block-bin reduction mode (Average / Sum / Subsample).
    bin_mode: BinMode,
    /// Display-time image orientation (no-op = `Identity`).
    orientation: Orientation,
    /// Optional user colormap loaded via `Color ▸ Load Custom…` — overrides
    /// `cmap` when present.
    custom_lut: Option<Box<[[u8; 3]; 256]>>,
    /// Disk path this frame was loaded from + the HDU index (0 = primary).
    /// Used by the HDU navigator dialog so we can re-load a different HDU
    /// without the user having to re-pick the file.
    hdu_idx: usize,
}

impl Frame {
    fn new(fits: FitsImage, name: String) -> Self {
        let (w, h) = (fits.width, fits.height);
        Self {
            fits,
            name,
            source_path: None,
            stretch: Stretch::Linear,
            limits_mode: LimitsMode::Zscale,
            cmap: Colormap::Grey,
            markers: Vec::new(),
            catalog: None,
            selected_catalog: None,
            selected_marker: None,
            view_zoom: fit_zoom(w, h),
            view_pan_x: 0.0,
            view_pan_y: 0.0,
            smooth_sigma: 0.0,
            smooth_kind: SmoothKind::default(),
            bin_factor: 1,
            bin_mode: BinMode::default(),
            orientation: Orientation::Identity,
            custom_lut: None,
            hdu_idx: 0,
        }
    }

    fn limits(&self) -> Limits {
        match self.limits_mode {
            LimitsMode::Zscale => Limits::zscale(&self.fits),
            LimitsMode::MinMax => Limits::minmax(&self.fits),
            LimitsMode::User { low, high } => Limits { low, high },
        }
    }

    fn limits_label(&self) -> &'static str {
        match self.limits_mode {
            LimitsMode::Zscale => "zscale",
            LimitsMode::MinMax => "minmax",
            LimitsMode::User { .. } => "user",
        }
    }

    fn stretch_label(&self) -> &'static str {
        match self.stretch {
            Stretch::Linear   => "linear",
            Stretch::Log      => "log",
            Stretch::Power(_) => "power",
            Stretch::Sqrt     => "sqrt",
            Stretch::Squared  => "squared",
            Stretch::Asinh    => "asinh",
            Stretch::Sinh     => "sinh",
        }
    }
}

/// Session-level crosshair. Pinned in world space when the frame the crosshair
/// was placed in had a WCS, otherwise pinned to that frame's pixel grid.
/// On render, the active frame projects this back into its own pixel space
/// (via `wcs.world_to_pix` if both world+wcs are available).
struct Crosshair {
    /// (RA, Dec) in degrees if the placing frame had a WCS.
    world: Option<(f64, f64)>,
    /// (frame_idx, x_fits, y_fits) — fallback when no WCS available.
    pixel: (usize, f64, f64),
}

/// Session-wide user preferences. Applied to every newly-created frame
/// (and optionally pushed onto the active frame via `Edit ▸ Preferences ▸
/// Apply`). Persisted to `~/.config/ds9-rust/prefs.toml` (simple
/// `key = value` text — no real TOML parser needed).
#[derive(Clone, Debug)]
struct Prefs {
    cmap:    String, // "grey" / "heat" / …
    stretch: String, // "linear" / "log" / …
    limits:  String, // "zscale" / "minmax"
    /// `off | gaussian:SIGMA | boxcar:N | median:N`
    smooth:  String,
    /// Block-bin factor (1 = off).
    bin:     u32,
    /// Blink interval in ms (informational only after startup; the timer
    /// is fixed once `main()` runs).
    blink_ms: u32,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            cmap: "grey".into(),
            stretch: "linear".into(),
            limits: "zscale".into(),
            smooth: "off".into(),
            bin: 1,
            blink_ms: 500,
        }
    }
}

struct State {
    frames: Vec<Frame>,
    /// Index into `frames`; only meaningful when `frames` is non-empty.
    active: usize,
    /// index into the active frame's markers (transient, not per-frame state)
    dragging_marker: Option<usize>,
    last_drag_fits: Option<(f64, f64)>,
    crosshair: Option<Crosshair>,
    /// Frame-lock toggles. When on, changes to the active frame are
    /// broadcast to every other loaded frame.
    lock_zoom:  bool,
    lock_pan:   bool,
    lock_cmap:  bool,
    lock_scale: bool,
    /// Session-wide defaults (applied to new frames; editable via
    /// `Edit ▸ Preferences…`).
    prefs: Prefs,
    /// Tile mode: when on, refresh_view renders every loaded frame in a
    /// grid instead of just the active one. Clicking a tile selects it
    /// and exits tile mode.
    tile_mode: bool,
}

impl State {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            active: 0,
            dragging_marker: None,
            last_drag_fits: None,
            crosshair: None,
            lock_zoom: false,
            lock_pan: false,
            lock_cmap: false,
            lock_scale: false,
            prefs: Prefs::default(),
            tile_mode: false,
        }
    }

    fn active_frame(&self) -> Option<&Frame> {
        self.frames.get(self.active)
    }
    fn active_frame_mut(&mut self) -> Option<&mut Frame> {
        self.frames.get_mut(self.active)
    }
}

// ---------------------------------------------------------------- render --

fn render_image(f: &Frame) -> Image {
    let rgba = render_rgba_for_frame(f);
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(f.fits.width as u32, f.fits.height as u32);
    buf.make_mut_bytes().copy_from_slice(&rgba);
    Image::from_rgba8(buf)
}

/// Decide a (cols, rows) grid for `n` tiles. cols = ceil(sqrt(n)), rows fills.
fn tile_layout(n: usize) -> (usize, usize) {
    if n == 0 { return (1, 1); }
    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = (n + cols - 1) / cols;
    (cols.max(1), rows.max(1))
}

/// Render every loaded frame in a grid into a single `out_w × out_h` RGBA
/// image. Each tile gets a 1-px border (`#2a3242` for inactive, accent gold
/// for the currently-active frame). Frame name is overlaid in the top-left
/// corner of each tile (small bitmap font from `render_line_plot`'s palette
/// — for simplicity we just colour-blit a couple of dots indicating the
/// frame index instead of full text).
fn render_tile_view(st: &State, out_w: usize, out_h: usize) -> Image {
    let n = st.frames.len();
    let (cols, rows) = tile_layout(n.max(1));
    let cell_w = out_w / cols;
    let cell_h = out_h / rows;

    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(out_w as u32, out_h as u32);
    {
        let bytes = buf.make_mut_bytes();
        // background `#0a0d14ff`
        for px in bytes.chunks_exact_mut(4) {
            px[0] = 0x0a; px[1] = 0x0d; px[2] = 0x14; px[3] = 0xff;
        }
        for (idx, f) in st.frames.iter().enumerate() {
            let cr = idx / cols;
            let cc = idx % cols;
            if cr >= rows { break; }
            let x0 = cc * cell_w;
            let y0 = cr * cell_h;
            let cw = cell_w.saturating_sub(2);
            let ch = cell_h.saturating_sub(2);
            if cw == 0 || ch == 0 { continue; }
            // render frame to its native size, then nearest-neighbour scale.
            let src = render_rgba_for_frame(f);
            let (sw, sh) = (f.fits.width, f.fits.height);
            // preserve aspect — fit into (cw, ch)
            let scale = (cw as f64 / sw as f64).min(ch as f64 / sh as f64);
            let tw = ((sw as f64 * scale).round() as usize).max(1).min(cw);
            let th = ((sh as f64 * scale).round() as usize).max(1).min(ch);
            let off_x = x0 + 1 + (cw - tw) / 2;
            let off_y = y0 + 1 + (ch - th) / 2;
            for ty in 0..th {
                let sy = (ty as f64 / scale) as usize;
                let sy = sy.min(sh - 1);
                let dst_row = (off_y + ty) * out_w * 4;
                let src_row = sy * sw * 4;
                for tx in 0..tw {
                    let sx = (tx as f64 / scale) as usize;
                    let sx = sx.min(sw - 1);
                    let di = dst_row + (off_x + tx) * 4;
                    let si = src_row + sx * 4;
                    bytes[di]   = src[si];
                    bytes[di+1] = src[si+1];
                    bytes[di+2] = src[si+2];
                    bytes[di+3] = 0xff;
                }
            }
            // border — accent gold for active, dim slate otherwise
            let (br, bg, bb) = if idx == st.active {
                (0xff, 0xc1, 0x07)
            } else {
                (0x2a, 0x32, 0x42)
            };
            let put = |b: &mut [u8], x: usize, y: usize| {
                if x < out_w && y < out_h {
                    let i = (y * out_w + x) * 4;
                    b[i] = br; b[i+1] = bg; b[i+2] = bb; b[i+3] = 0xff;
                }
            };
            for x in x0..(x0 + cell_w).min(out_w) {
                put(bytes, x, y0);
                put(bytes, x, (y0 + cell_h).min(out_h.saturating_sub(1)));
            }
            for y in y0..(y0 + cell_h).min(out_h) {
                put(bytes, x0, y);
                put(bytes, (x0 + cell_w).min(out_w.saturating_sub(1)), y);
            }
        }
    }
    Image::from_rgba8(buf)
}

/// Reverse of `render_tile_view`'s grid layout: given a click in canvas
/// coordinates and the same `out_w/out_h` used to render the tiles, return
/// the frame index whose tile cell contains the click.
fn tile_hit(st: &State, out_w: usize, out_h: usize, cx: f64, cy: f64) -> Option<usize> {
    let n = st.frames.len();
    if n == 0 { return None; }
    let (cols, rows) = tile_layout(n);
    let cell_w = (out_w / cols) as f64;
    let cell_h = (out_h / rows) as f64;
    if cell_w <= 0.0 || cell_h <= 0.0 { return None; }
    let cc = (cx / cell_w).floor() as i64;
    let cr = (cy / cell_h).floor() as i64;
    if cc < 0 || cr < 0 { return None; }
    let idx = (cr as usize) * cols + (cc as usize);
    if idx >= n { return None; }
    Some(idx)
}

/// Apply per-frame bin/smooth filters before stretching, returning the raw
/// flipped RGBA bytes. Caller wraps these into an Image / PNG / etc.
fn render_rgba_for_frame(f: &Frame) -> Vec<u8> {
    let mut owned: Option<FitsImage> = None;
    if f.bin_factor > 1 {
        owned = Some(match f.bin_mode {
            BinMode::Average   => ds9_image::bin_average(&f.fits, f.bin_factor),
            BinMode::Sum       => ds9_image::bin_sum(&f.fits, f.bin_factor),
            BinMode::Subsample => ds9_image::bin_subsample(&f.fits, f.bin_factor),
        });
    }
    let smoothed = match f.smooth_kind {
        SmoothKind::Gaussian { sigma } if sigma > 0.0 => Some(ds9_image::smooth_gaussian(
            owned.as_ref().unwrap_or(&f.fits), sigma,
        )),
        SmoothKind::Boxcar { n } if n > 1 => Some(ds9_image::smooth_boxcar(
            owned.as_ref().unwrap_or(&f.fits), n,
        )),
        SmoothKind::Median { n } if n > 1 => Some(ds9_image::smooth_median(
            owned.as_ref().unwrap_or(&f.fits), n,
        )),
        _ => None,
    };
    if smoothed.is_some() { owned = smoothed; }
    let img: &FitsImage = owned.as_ref().unwrap_or(&f.fits);
    let limits = match f.limits_mode {
        LimitsMode::Zscale => Limits::zscale(img),
        LimitsMode::MinMax => Limits::minmax(img),
        LimitsMode::User { low, high } => Limits { low, high },
    };
    let mut rgba = if let Some(lut) = &f.custom_lut {
        ds9_image::render_rgba_flipped_with_lut(img, limits, f.stretch, lut)
    } else {
        ds9_image::render_rgba_flipped(img, limits, f.stretch, f.cmap)
    };
    if f.orientation != Orientation::Identity {
        ds9_image::apply_orientation_rgba(&mut rgba, img.width, img.height, f.orientation);
    }
    rgba
}

fn make_colorbar_strip(cmap: Colormap) -> Image {
    let strip = cmap.rgba_strip();
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(1, 256);
    buf.make_mut_bytes().copy_from_slice(&strip);
    Image::from_rgba8(buf)
}

fn fit_zoom(w: usize, h: usize) -> f32 {
    let zw = 800.0 / (w as f32).max(1.0);
    let zh = 600.0 / (h as f32).max(1.0);
    zw.min(zh).clamp(0.05, 32.0)
}

/// Marker storage uses DS9 / FITS conventions (1-based, y up from bottom).
/// The slint canvas wants display-space coords (0-based, y down). Convert.
/// (No orientation applied — see [`fits_to_display_oriented`] for the
/// orientation-aware variant.)
fn fits_to_display(cx: f64, cy: f64, h: usize) -> (f32, f32) {
    ((cx - 1.0) as f32, (h as f32 - cy as f32))
}

/// Like [`fits_to_display`] but also applies the frame's display orientation.
fn fits_to_display_oriented(cx: f64, cy: f64, f: &Frame) -> (f32, f32) {
    let (dx, dy) = fits_to_display(cx, cy, f.fits.height);
    let (w, h) = (f.fits.width as f64, f.fits.height as f64);
    let (ox, oy) = f.orientation.apply_display(dx as f64, dy as f64, w, h);
    (ox as f32, oy as f32)
}

/// Inverse of [`fits_to_display_oriented`]: take a slint canvas display coord
/// (0-based, y-down, possibly oriented) and return the underlying FITS coord.
fn display_to_fits(dx: f64, dy: f64, f: &Frame) -> (f64, f64) {
    let (w, h) = (f.fits.width as f64, f.fits.height as f64);
    let (ux, uy) = f.orientation.invert_display(dx, dy, w, h);
    (ux + 1.0, h - uy)
}

fn marker_color(m: &Marker) -> slint::Color {
    slint::Color::from_argb_u8(m.color[3], m.color[0], m.color[1], m.color[2])
}

fn build_mark_model(f: &Frame) -> ModelRc<Mark> {
    let cat_count = f.catalog.as_ref().map(|c| c.len()).unwrap_or(0).min(5000);
    let mut out: Vec<Mark> = Vec::with_capacity(f.markers.len() + cat_count);

    // catalog points first so user-drawn regions paint on top
    if let Some(cat) = &f.catalog {
        let amber = slint::Color::from_argb_u8(0xff, 0xff, 0xc1, 0x07);
        for (i, (x, y)) in cat.xy_iter().enumerate() {
            if i >= 5000 { break; }
            let (cx, cy) = fits_to_display_oriented(x, y, f);
            let selected = f.selected_catalog == Some(i);
            // make the selected source visibly bigger so it stands out at low zoom
            let r = if selected { 8.0 } else { 4.0 };
            out.push(Mark {
                kind: 0, cx, cy, rx: r, ry: r,
                color: amber, selected,
            });
        }
    }

    for (i, m) in f.markers.iter().enumerate() {
        let color = marker_color(m);
        let sel = f.selected_marker == Some(i);
        let mark = match &m.shape {
            MShape::Circle { center: c, r } => {
                let (cx, cy) = fits_to_display_oriented(c.x, c.y, f);
                Some(Mark { kind: 0, cx, cy, rx: *r as f32, ry: *r as f32, color, selected: sel })
            }
            MShape::Box { center: c, w, h: bh, .. } => {
                let (cx, cy) = fits_to_display_oriented(c.x, c.y, f);
                Some(Mark { kind: 1, cx, cy, rx: (*w as f32) / 2.0, ry: (*bh as f32) / 2.0, color, selected: sel })
            }
            MShape::Ellipse { center: c, a, b, .. } => {
                let (cx, cy) = fits_to_display_oriented(c.x, c.y, f);
                Some(Mark { kind: 0, cx, cy, rx: *a as f32, ry: *b as f32, color, selected: sel })
            }
            MShape::Annulus { center: c, r_outer, .. } => {
                let (cx, cy) = fits_to_display_oriented(c.x, c.y, f);
                Some(Mark { kind: 0, cx, cy, rx: *r_outer as f32, ry: *r_outer as f32, color, selected: sel })
            }
            MShape::Point { center: c } => {
                let (cx, cy) = fits_to_display_oriented(c.x, c.y, f);
                Some(Mark { kind: 1, cx, cy, rx: 2.0, ry: 2.0, color, selected: sel })
            }
            // line / polygon / compass / text are baked into the line overlay
            // (see render_line_overlay) — they don't contribute Mark rectangles.
            _ => None,
        };
        if let Some(m) = mark { out.push(m); }
    }
    ModelRc::new(VecModel::from(out))
}

/// Bake DS9 line / polygon / compass markers as a frame-resolution RGBA strip
/// (same dims as the rendered image, oriented). Returns None if there's nothing
/// to draw, in which case callers should hide the overlay.
fn render_line_overlay(f: &Frame) -> Option<Image> {
    let (w, h) = (f.fits.width, f.fits.height);
    if w == 0 || h == 0 { return None; }

    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    let bytes = buf.make_mut_bytes();
    for px in bytes.chunks_exact_mut(4) { px.copy_from_slice(&[0, 0, 0, 0]); }

    let mut any = false;
    let put = |bytes: &mut [u8], x: i32, y: i32, c: [u8; 4]| {
        if x < 0 || y < 0 { return; }
        let (xu, yu) = (x as usize, y as usize);
        if xu >= w || yu >= h { return; }
        let i = (yu * w + xu) * 4;
        bytes[i..i+4].copy_from_slice(&c);
    };
    let draw_line = |bytes: &mut [u8], x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 4]| {
        let dx =  (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        let max_steps = (w + h) as i32 * 4;
        let mut steps = 0;
        loop {
            put(bytes, x, y, c);
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
            steps += 1;
            if steps > max_steps { break; }
        }
    };
    let to_xy = |fx: f64, fy: f64| -> (i32, i32) {
        let (dx, dy) = fits_to_display_oriented(fx, fy, f);
        (dx.round() as i32, dy.round() as i32)
    };

    for m in &f.markers {
        let c = m.color;
        match &m.shape {
            MShape::Line { from, to } => {
                let (x0, y0) = to_xy(from.x, from.y);
                let (x1, y1) = to_xy(to.x,   to.y);
                draw_line(bytes, x0, y0, x1, y1, c);
                // arrowhead — short bevel back from the endpoint
                let dx = (x1 - x0) as f64;
                let dy = (y1 - y0) as f64;
                let len = (dx*dx + dy*dy).sqrt().max(1.0);
                let (ux, uy) = (dx / len, dy / len);
                let head = 9.0_f64;
                for off in [(-uy, ux), (uy, -ux)] {
                    let bx = x1 as f64 - ux * head + off.0 * head * 0.4;
                    let by = y1 as f64 - uy * head + off.1 * head * 0.4;
                    draw_line(bytes, x1, y1, bx.round() as i32, by.round() as i32, c);
                }
                any = true;
            }
            MShape::Polygon { points } if points.len() >= 2 => {
                for win in points.windows(2) {
                    let (x0, y0) = to_xy(win[0].x, win[0].y);
                    let (x1, y1) = to_xy(win[1].x, win[1].y);
                    draw_line(bytes, x0, y0, x1, y1, c);
                }
                let (x0, y0) = to_xy(points[points.len()-1].x, points[points.len()-1].y);
                let (x1, y1) = to_xy(points[0].x, points[0].y);
                draw_line(bytes, x0, y0, x1, y1, c);
                any = true;
            }
            MShape::Compass { center, len } => {
                let (cx, cy) = to_xy(center.x, center.y);
                let l = *len as i32;
                draw_line(bytes, cx, cy, cx + l, cy, c); // east
                draw_line(bytes, cx, cy, cx, cy - l, c); // north (display y-up)
                any = true;
            }
            _ => {}
        }
    }

    if any { Some(Image::from_rgba8(buf)) } else { None }
}

fn build_text_marks(f: &Frame) -> ModelRc<TextMark> {
    let mut out: Vec<TextMark> = Vec::new();
    for m in &f.markers {
        if let MShape::Text { center, body } = &m.shape {
            let (dx, dy) = fits_to_display_oriented(center.x, center.y, f);
            out.push(TextMark {
                x: dx, y: dy,
                body: body.clone().into(),
                color: marker_color(m),
            });
        }
    }
    ModelRc::new(VecModel::from(out))
}

/// Hit-test user-drawn markers (FITS coords). Topmost (last-drawn) wins.
fn hit_test_markers(markers: &[Marker], fx: f64, fy: f64) -> Option<usize> {
    for (i, m) in markers.iter().enumerate().rev() {
        let inside = match &m.shape {
            MShape::Circle { center, r } => {
                let dx = fx - center.x; let dy = fy - center.y;
                dx*dx + dy*dy <= (*r) * (*r)
            }
            MShape::Box { center, w, h, .. } => {
                (fx - center.x).abs() <= w / 2.0 && (fy - center.y).abs() <= h / 2.0
            }
            MShape::Ellipse { center, a, b, .. } => {
                let dx = (fx - center.x) / a;
                let dy = (fy - center.y) / b;
                dx*dx + dy*dy <= 1.0
            }
            MShape::Annulus { center, r_outer, .. } => {
                let dx = fx - center.x; let dy = fy - center.y;
                dx*dx + dy*dy <= (*r_outer) * (*r_outer)
            }
            MShape::Point { center } => {
                let dx = fx - center.x; let dy = fy - center.y;
                dx*dx + dy*dy <= 16.0  // 4-px tolerance
            }
            _ => false,
        };
        if inside { return Some(i); }
    }
    None
}

/// Display-space axis-aligned bounding box of a marker, using the active
/// frame's orientation. Returns (min_dx, min_dy, max_dx, max_dy).
fn marker_display_bbox(m: &Marker, f: &Frame) -> (f64, f64, f64, f64) {
    // Sample the shape's outline (or center+extent) into FITS coords, then
    // project each point through fits_to_display_oriented.
    let mut pts: Vec<(f64, f64)> = Vec::new();
    match &m.shape {
        MShape::Circle { center, r } | MShape::Annulus { center, r_outer: r, .. } => {
            for k in 0..36 {
                let t = k as f64 * std::f64::consts::TAU / 36.0;
                pts.push((center.x + r * t.cos(), center.y + r * t.sin()));
            }
        }
        MShape::Box { center, w, h, theta_deg } => {
            let (hw, hh) = (w * 0.5, h * 0.5);
            let (s, c) = (theta_deg.to_radians().sin(), theta_deg.to_radians().cos());
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let (lx, ly) = (sx * hw, sy * hh);
                pts.push((center.x + lx * c - ly * s, center.y + lx * s + ly * c));
            }
        }
        MShape::Ellipse { center, a, b, theta_deg } => {
            let (s, c) = (theta_deg.to_radians().sin(), theta_deg.to_radians().cos());
            for k in 0..36 {
                let t = k as f64 * std::f64::consts::TAU / 36.0;
                let (lx, ly) = (a * t.cos(), b * t.sin());
                pts.push((center.x + lx * c - ly * s, center.y + lx * s + ly * c));
            }
        }
        MShape::Polygon { points } => {
            for p in points { pts.push((p.x, p.y)); }
        }
        MShape::Line { from, to } => {
            pts.push((from.x, from.y));
            pts.push((to.x, to.y));
        }
        MShape::Point { center } | MShape::Compass { center, .. } | MShape::Text { center, .. } => {
            // small surround so the box has area
            for d in [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)] {
                pts.push((center.x + d.0, center.y + d.1));
            }
        }
    }
    let mut mnx = f64::INFINITY;  let mut mxx = f64::NEG_INFINITY;
    let mut mny = f64::INFINITY;  let mut mxy = f64::NEG_INFINITY;
    for (fx, fy) in pts {
        let (dx, dy) = fits_to_display_oriented(fx, fy, f);
        let (dx, dy) = (dx as f64, dy as f64);
        if dx < mnx { mnx = dx; }  if dx > mxx { mxx = dx; }
        if dy < mny { mny = dy; }  if dy > mxy { mxy = dy; }
    }
    (mnx, mny, mxx, mxy)
}

/// Translate a marker's geometry by (dx, dy) in FITS pixels.
fn translate_marker(m: &mut Marker, dx: f64, dy: f64) {
    use ds9_marker::PixelPos;
    fn shift(p: &mut PixelPos, dx: f64, dy: f64) { p.x += dx; p.y += dy; }
    match &mut m.shape {
        MShape::Circle  { center, .. }
        | MShape::Box    { center, .. }
        | MShape::Ellipse{ center, .. }
        | MShape::Annulus{ center, .. }
        | MShape::Point  { center }
        | MShape::Compass{ center, .. }
        | MShape::Text   { center, .. } => shift(center, dx, dy),
        MShape::Line { from, to } => { shift(from, dx, dy); shift(to, dx, dy); }
        MShape::Polygon { points } => { for p in points { shift(p, dx, dy); } }
    }
}

/// Pick a magnitude column heuristically (SExtractor-style names first).
fn mag_column(cat: &Catalog) -> Option<usize> {
    const NAMES: &[&str] = &[
        "MAG_AUTO", "MAG_BEST", "MAG_APER", "MAG_PSF", "MAG", "mag",
    ];
    NAMES.iter().find_map(|n| cat.col_index(n))
}

fn build_catalog_model(f: &Frame) -> ModelRc<CatRow> {
    let Some(cat) = &f.catalog else {
        return ModelRc::new(VecModel::from(Vec::<CatRow>::new()));
    };
    let id_idx = cat.col_index("NUMBER").or_else(|| cat.col_index("ID"));
    let mag_idx = mag_column(cat);
    let xy = cat.xy_columns();
    let Some((xi, yi)) = xy else {
        return ModelRc::new(VecModel::from(Vec::<CatRow>::new()));
    };
    let mut out: Vec<CatRow> = Vec::with_capacity(cat.len().min(5000));
    let mut row_kept = 0usize;
    for (raw_idx, row) in cat.rows.iter().enumerate() {
        let x = row.get(xi).and_then(|s| s.parse::<f64>().ok());
        let y = row.get(yi).and_then(|s| s.parse::<f64>().ok());
        let (Some(x), Some(y)) = (x, y) else { continue };
        if row_kept >= 5000 { break; }
        let id = match id_idx {
            Some(i) => row.get(i).cloned().unwrap_or_else(|| (raw_idx + 1).to_string()),
            None => (raw_idx + 1).to_string(),
        };
        let mag = match mag_idx {
            Some(i) => row.get(i).and_then(|s| s.parse::<f64>().ok())
                .map(|v| format!("{v:>6.2}"))
                .unwrap_or_else(|| "—".into()),
            None => "—".into(),
        };
        out.push(CatRow { id: id.into(), x: x as f32, y: y as f32, mag: mag.into() });
        row_kept += 1;
    }
    ModelRc::new(VecModel::from(out))
}

/// Find the catalog row (in the *kept* index space — matching what
/// `build_mark_model` and `build_catalog_model` produce) closest to the given
/// FITS pixel within `tol_px` pixels. Returns (kept_index, distance²).
fn nearest_catalog_index(cat: &Catalog, fx: f64, fy: f64, tol_px: f64) -> Option<usize> {
    let tol2 = tol_px * tol_px;
    let mut best: Option<(usize, f64)> = None;
    for (i, (x, y)) in cat.xy_iter().enumerate() {
        let dx = x - fx;
        let dy = y - fy;
        let d2 = dx * dx + dy * dy;
        if d2 <= tol2 && best.map_or(true, |(_, b2)| d2 < b2) {
            best = Some((i, d2));
        }
    }
    best.map(|(i, _)| i)
}

/// Push current state-derived visuals (image, colorbar, markers, info badges) into the window.
fn refresh_view(window: &MainWindow, st: &State) {
    let frame_label = if st.frames.is_empty() {
        "0/0".to_string()
    } else {
        format!("{}/{}", st.active + 1, st.frames.len())
    };
    window.set_info_frame(frame_label.into());

    let Some(f) = st.active_frame() else {
        // no frames at all — push neutral defaults
        window.set_info_filename("—".into());
        window.set_info_object("—".into());
        window.set_active_stretch("linear".into());
        window.set_active_limits("zscale".into());
        window.set_active_cmap("grey".into());
        window.set_colorbar_strip(make_colorbar_strip(Colormap::Grey));
        window.set_markers(ModelRc::new(VecModel::from(Vec::<Mark>::new())));
        window.set_catalog_rows(ModelRc::new(VecModel::from(Vec::<CatRow>::new())));
        window.set_catalog_selected(-1);
        window.set_fits_width(0);
        window.set_fits_height(0);
        return;
    };

    window.set_active_stretch(f.stretch_label().into());
    window.set_active_limits(f.limits_label().into());
    window.set_active_cmap(f.cmap.name().into());
    window.set_colorbar_strip(make_colorbar_strip(f.cmap));

    if st.tile_mode {
        // Tile mode: replace the active-frame image with a composite of every
        // loaded frame; suppress per-frame overlays (markers, line/grid).
        let cw = (window.get_canvas_w() as f32 as i32).max(64) as usize;
        let ch = (window.get_canvas_h() as f32 as i32).max(64) as usize;
        window.set_fits_image(render_tile_view(st, cw, ch));
        window.set_fits_width(cw as i32);
        window.set_fits_height(ch as i32);
        window.set_view_zoom(1.0);
        window.set_view_pan_x(0.0);
        window.set_view_pan_y(0.0);
        window.set_markers(ModelRc::new(VecModel::from(Vec::<Mark>::new())));
        window.set_text_marks(ModelRc::new(VecModel::from(Vec::<TextMark>::new())));
        window.set_catalog_rows(build_catalog_model(f));
        window.set_catalog_selected(-1);
        window.set_info_filename(format!("[tile] {} frames", st.frames.len()).into());
        window.set_line_visible(false);
        window.set_grid_visible(false);
        push_crosshair_to_window(window, st);
        window.set_lock_zoom(st.lock_zoom);
        window.set_lock_pan(st.lock_pan);
        window.set_lock_cmap(st.lock_cmap);
        window.set_lock_scale(st.lock_scale);
        return;
    }

    window.set_markers(build_mark_model(f));
    window.set_text_marks(build_text_marks(f));
    window.set_catalog_rows(build_catalog_model(f));
    window.set_catalog_selected(f.selected_catalog.map(|i| i as i32).unwrap_or(-1));
    window.set_fits_image(render_image(f));
    window.set_fits_width(f.fits.width as i32);
    window.set_fits_height(f.fits.height as i32);
    window.set_info_filename(f.name.clone().into());

    // refresh line / vector / polygon / compass overlay
    match render_line_overlay(f) {
        Some(img) => { window.set_line_image(img); window.set_line_visible(true); }
        None => window.set_line_visible(false),
    }

    // refresh the WCS grid (cheap to bake at frame dims) when toggled on
    if window.get_grid_visible() {
        if let Some(img) = render_grid_overlay(f) {
            window.set_grid_image(img);
        } else {
            window.set_grid_visible(false);
        }
    }
    push_crosshair_to_window(window, st);

    // sync the lock toggles so the menu's check state is accurate
    window.set_lock_zoom(st.lock_zoom);
    window.set_lock_pan(st.lock_pan);
    window.set_lock_cmap(st.lock_cmap);
    window.set_lock_scale(st.lock_scale);
}

/// 128-bin histogram of finite samples in [lo, hi], rendered as a log-scale
/// bar plot on a dark background. Output is `w × h` RGBA suitable for
/// `Image::from_rgba8`.
fn render_histogram_image(f: &Frame, w: usize, h: usize) -> Image {
    let limits = f.limits();
    let (lo, hi) = (limits.low as f64, limits.high as f64);
    let span = (hi - lo).max(f64::EPSILON);

    const BINS: usize = 128;
    let mut bins = [0u64; BINS];
    for &v in &f.fits.data {
        if !v.is_finite() { continue; }
        let t = (v as f64 - lo) / span;
        if !(0.0..=1.0).contains(&t) { continue; }
        let mut b = (t * BINS as f64) as usize;
        if b >= BINS { b = BINS - 1; }
        bins[b] += 1;
    }
    let max_count = *bins.iter().max().unwrap_or(&1) as f64;
    let log_max = (max_count + 1.0).ln().max(1.0);

    // background `#0f1319ff`, frame `#2a3242ff`, bars accent teal `#4ec9b0ff`
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    {
        let bytes = buf.make_mut_bytes();
        for px in bytes.chunks_exact_mut(4) {
            px[0] = 0x0f; px[1] = 0x13; px[2] = 0x19; px[3] = 0xff;
        }
        // 1-px frame
        let mut put = |x: usize, y: usize, c: [u8; 4]| {
            if x < w && y < h {
                let i = (y * w + x) * 4;
                bytes[i..i+4].copy_from_slice(&c);
            }
        };
        let frame = [0x2a, 0x32, 0x42, 0xff];
        for x in 0..w { put(x, 0, frame); put(x, h-1, frame); }
        for y in 0..h { put(0, y, frame); put(w-1, y, frame); }

        // bars: each bin -> a column slab of width `w / BINS`
        let bar_w = (w as f64 / BINS as f64).max(1.0);
        let plot_top    = 8usize;
        let plot_bottom = h.saturating_sub(8);
        let plot_h      = plot_bottom.saturating_sub(plot_top).max(1);
        let bar = [0x4e, 0xc9, 0xb0, 0xff];
        for (bi, &count) in bins.iter().enumerate() {
            if count == 0 { continue; }
            let frac = (count as f64 + 1.0).ln() / log_max;
            let bar_h = (frac * plot_h as f64).round() as usize;
            let x0 = (bi as f64 * bar_w).floor() as usize;
            let x1 = (((bi + 1) as f64) * bar_w).floor() as usize;
            let y0 = plot_bottom.saturating_sub(bar_h);
            for y in y0..plot_bottom {
                for x in x0..x1.min(w.saturating_sub(1)) {
                    put(x, y, bar);
                }
            }
        }
    }
    Image::from_rgba8(buf)
}

/// Sample pixel values along a straight line in FITS-pixel space at 1-pixel
/// steps (nearest-neighbor). Returns (samples, length_in_pixels).
fn sample_line(f: &Frame, x0: f64, y0: f64, x1: f64, y1: f64) -> (Vec<f32>, f64) {
    let (w, h) = (f.fits.width, f.fits.height);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    let n = (len.ceil() as usize).max(2);
    let mut out: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        // FITS coords are 1-based; storage indices are 0-based.
        let fx = x0 + dx * t;
        let fy = y0 + dy * t;
        let ix = (fx - 1.0).round() as isize;
        // FITS y is bottom-up → flip to storage y (top-down).
        let iy_disp = (h as isize) - 1 - ((fy - 1.0).round() as isize);
        let v = if ix >= 0 && (ix as usize) < w && iy_disp >= 0 && (iy_disp as usize) < h {
            f.fits.data[(iy_disp as usize) * w + (ix as usize)]
        } else {
            f32::NAN
        };
        out.push(v);
    }
    (out, len)
}

/// Render a 1D line plot of `samples` (pixel value vs distance index) into a
/// w×h RGBA image. Caption embeds basic stats for the user.
fn render_line_plot(samples: &[f32], w: usize, h: usize) -> Image {
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    let bytes = buf.make_mut_bytes();
    for px in bytes.chunks_exact_mut(4) {
        px[0] = 0x0f; px[1] = 0x13; px[2] = 0x19; px[3] = 0xff;
    }
    let put = |bytes: &mut [u8], x: usize, y: usize, c: [u8; 4]| {
        if x < w && y < h {
            let i = (y * w + x) * 4;
            bytes[i..i+4].copy_from_slice(&c);
        }
    };
    let frame = [0x2a, 0x32, 0x42, 0xff];
    for x in 0..w { put(bytes, x, 0, frame); put(bytes, x, h-1, frame); }
    for y in 0..h { put(bytes, 0, y, frame); put(bytes, w-1, y, frame); }

    let finite: Vec<f32> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || samples.len() < 2 { return Image::from_rgba8(buf); }

    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in &finite {
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    let span = (hi - lo).max(f32::EPSILON);

    let pad_x = 6usize;
    let pad_y = 8usize;
    let plot_w = w.saturating_sub(pad_x * 2).max(1);
    let plot_h = h.saturating_sub(pad_y * 2).max(1);
    let n = samples.len();
    let line = [0x4e, 0xc9, 0xb0, 0xff];

    // Bresenham polyline
    let mut prev: Option<(i32, i32)> = None;
    let draw_seg = |bytes: &mut [u8], (mut x, mut y): (i32, i32), (x1, y1): (i32, i32), c: [u8; 4]| {
        let dx =  (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx = if x < x1 { 1 } else { -1 };
        let sy = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x >= 0 && y >= 0 { put(bytes, x as usize, y as usize, c); }
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    };
    for (i, &v) in samples.iter().enumerate() {
        if !v.is_finite() { prev = None; continue; }
        let xf = pad_x as f64 + (i as f64) * (plot_w as f64) / ((n - 1) as f64);
        let yf = pad_y as f64 + (1.0 - ((v - lo) / span) as f64) * (plot_h as f64);
        let cur = (xf.round() as i32, yf.round() as i32);
        if let Some(p) = prev { draw_seg(bytes, p, cur, line); }
        prev = Some(cur);
    }

    Image::from_rgba8(buf)
}

/// 5-level contour overlay drawn over the original image (display-flipped to
/// match `render_rgba_flipped`). Uses cheap sign-change detection on right/down
/// neighbors instead of full marching squares — one edge pixel per crossing.
fn render_contour_overlay(f: &Frame) -> Image {
    let limits = f.limits();
    let (lo, hi) = (limits.low as f64, limits.high as f64);
    let span = hi - lo;
    let levels: [f64; 5] = [
        lo + span * 1.0 / 6.0,
        lo + span * 2.0 / 6.0,
        lo + span * 3.0 / 6.0,
        lo + span * 4.0 / 6.0,
        lo + span * 5.0 / 6.0,
    ];
    // distinct contour colors (RGBA u8) — keep alpha < 255 so the image still shows through faintly
    let palette: [[u8; 4]; 5] = [
        [0x38, 0x98, 0xec, 0xe0],
        [0x4e, 0xc9, 0xb0, 0xe0],
        [0xff, 0xc1, 0x07, 0xe0],
        [0xd9, 0x77, 0x57, 0xe0],
        [0xff, 0x6b, 0x9d, 0xe0],
    ];

    let w = f.fits.width;
    let h = f.fits.height;
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    let bytes = buf.make_mut_bytes();
    // start fully transparent
    for px in bytes.chunks_exact_mut(4) { px.copy_from_slice(&[0, 0, 0, 0]); }

    let put = |bytes: &mut [u8], x: usize, y_disp: usize, c: [u8; 4]| {
        if x >= w || y_disp >= h { return; }
        let i = (y_disp * w + x) * 4;
        bytes[i..i+4].copy_from_slice(&c);
    };

    let data = &f.fits.data;
    for y in 0..h {
        let y_disp = h - 1 - y; // flip to match render_rgba_flipped
        for x in 0..w {
            let v = data[y * w + x] as f64;
            if !v.is_finite() { continue; }
            // right neighbor
            let r = if x + 1 < w { Some(data[y * w + x + 1] as f64) } else { None };
            // down neighbor (in FITS-y; that's still data[y+1] in our storage)
            let d = if y + 1 < h { Some(data[(y + 1) * w + x] as f64) } else { None };
            for (li, &lev) in levels.iter().enumerate() {
                let crosses = |a: f64, b: f64| (a - lev) * (b - lev) < 0.0;
                let mut hit = false;
                if let Some(rv) = r { if rv.is_finite() && crosses(v, rv) { hit = true; } }
                if !hit {
                    if let Some(dv) = d { if dv.is_finite() && crosses(v, dv) { hit = true; } }
                }
                if hit {
                    put(bytes, x, y_disp, palette[li]);
                    break;
                }
            }
        }
    }
    if f.orientation != Orientation::Identity {
        ds9_image::apply_orientation_rgba(buf.make_mut_bytes(), w, h, f.orientation);
    }
    Image::from_rgba8(buf)
}

/// Render an RA/Dec grid as an image-resolution RGBA overlay. Picks a "nice"
/// step on each axis so the visible field hits ~6-8 grid lines. Returns None
/// if the frame has no WCS or world↔pixel projection fails everywhere.
fn render_grid_overlay(f: &Frame) -> Option<Image> {
    let wcs = f.fits.wcs.as_ref()?;
    let w = f.fits.width;
    let h = f.fits.height;
    if w == 0 || h == 0 { return None; }

    // Sample the four edges to bracket the visible RA/Dec range.
    let mut ras: Vec<f64> = Vec::new();
    let mut decs: Vec<f64> = Vec::new();
    let n_edge = 16;
    for i in 0..=n_edge {
        let t = i as f64 / n_edge as f64;
        let xs = 1.0 + t * (w as f64 - 1.0);
        let ys = 1.0 + t * (h as f64 - 1.0);
        for &(x, y) in &[
            (xs, 1.0), (xs, h as f64),
            (1.0, ys), (w as f64, ys),
        ] {
            let (ra, dec) = wcs.pix_to_world(x, y);
            ras.push(ra);
            decs.push(dec);
        }
    }

    let dec_min = decs.iter().cloned().fold(f64::INFINITY, f64::min);
    let dec_max = decs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !dec_min.is_finite() || !dec_max.is_finite() { return None; }

    // RA is periodic — handle 0/360 wrap by working in (sin, cos) space and
    // recovering a contiguous span if the points straddle 0°.
    let mut ra_min = ras.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut ra_max = ras.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if ra_max - ra_min > 180.0 {
        // wrap: shift any RA < 180 by +360 so the span stays small
        let shifted: Vec<f64> = ras.iter().map(|&r| if r < 180.0 { r + 360.0 } else { r }).collect();
        ra_min = shifted.iter().cloned().fold(f64::INFINITY, f64::min);
        ra_max = shifted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

    let nice_step = |span: f64| -> f64 {
        // ~6-8 lines across the span. Candidates from coarse degrees down to 1".
        const STEPS: &[f64] = &[
            30.0, 15.0, 10.0, 5.0, 2.0, 1.0,
            0.5, 30.0/60.0, 15.0/60.0, 10.0/60.0, 5.0/60.0, 2.0/60.0, 1.0/60.0,
            30.0/3600.0, 15.0/3600.0, 10.0/3600.0, 5.0/3600.0, 2.0/3600.0, 1.0/3600.0,
        ];
        for &s in STEPS {
            if span / s <= 8.0 && span / s >= 2.0 { return s; }
        }
        // very tight or very wide — pick the smallest meaningful one
        STEPS[STEPS.len() - 1]
    };
    let ra_step  = nice_step(ra_max - ra_min);
    let dec_step = nice_step(dec_max - dec_min);

    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    let bytes = buf.make_mut_bytes();
    for px in bytes.chunks_exact_mut(4) { px.copy_from_slice(&[0, 0, 0, 0]); }
    let line_color = [0x4e, 0xc9, 0xb0, 0x90]; // teal, semi-transparent

    let put = |bytes: &mut [u8], x: i32, y: i32, c: [u8; 4]| {
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 4;
            bytes[i..i+4].copy_from_slice(&c);
        }
    };

    // Bresenham — clipped at the put-bounds.
    let draw_line = |bytes: &mut [u8], x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 4]| {
        let dx =  (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        // sanity cap so a runaway projection can't lock us up
        let mut steps = 0;
        let max_steps = (w + h) as i32 * 4;
        loop {
            put(bytes, x, y, c);
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
            steps += 1;
            if steps > max_steps { break; }
        }
    };

    let project = |ra: f64, dec: f64| -> Option<(i32, i32)> {
        let (px, py) = wcs.world_to_pix(ra, dec)?;
        // FITS y → display y (top-down). Marker code uses `h - cy`, so:
        let dx = (px - 1.0).round() as i32;
        let dy = (h as f64 - py).round() as i32;
        Some((dx, dy))
    };

    // Lines of constant Dec, sampled in RA.
    let dec_first = (dec_min / dec_step).floor() * dec_step;
    let n_dec = ((dec_max - dec_first) / dec_step).ceil() as i32 + 1;
    for k in 0..n_dec {
        let dec = dec_first + k as f64 * dec_step;
        if dec < dec_min - 0.5 * dec_step || dec > dec_max + 0.5 * dec_step { continue; }
        let mut prev: Option<(i32, i32)> = None;
        let n_pts = 200;
        for j in 0..=n_pts {
            let t = j as f64 / n_pts as f64;
            let ra = ra_min + t * (ra_max - ra_min);
            // unwrap shifted RA back into [0, 360)
            let ra_norm = ra.rem_euclid(360.0);
            match project(ra_norm, dec) {
                Some(p) => {
                    if let Some(p0) = prev { draw_line(bytes, p0.0, p0.1, p.0, p.1, line_color); }
                    prev = Some(p);
                }
                None => prev = None,
            }
        }
    }
    // Lines of constant RA, sampled in Dec.
    let ra_first = (ra_min / ra_step).floor() * ra_step;
    let n_ra = ((ra_max - ra_first) / ra_step).ceil() as i32 + 1;
    for k in 0..n_ra {
        let ra = ra_first + k as f64 * ra_step;
        if ra < ra_min - 0.5 * ra_step || ra > ra_max + 0.5 * ra_step { continue; }
        let ra_norm = ra.rem_euclid(360.0);
        let mut prev: Option<(i32, i32)> = None;
        let n_pts = 200;
        for j in 0..=n_pts {
            let t = j as f64 / n_pts as f64;
            let dec = dec_min + t * (dec_max - dec_min);
            match project(ra_norm, dec) {
                Some(p) => {
                    if let Some(p0) = prev { draw_line(bytes, p0.0, p0.1, p.0, p.1, line_color); }
                    prev = Some(p);
                }
                None => prev = None,
            }
        }
    }

    if f.orientation != Orientation::Identity {
        ds9_image::apply_orientation_rgba(buf.make_mut_bytes(), w, h, f.orientation);
    }
    Some(Image::from_rgba8(buf))
}

/// Compute the active frame's display-space (0-based, y-down) crosshair coords
/// from the session crosshair, projecting through the frame's WCS when both
/// the world point and the frame's WCS are available. Returns None if the
/// crosshair does not project into this frame.
fn project_crosshair(st: &State) -> Option<(f32, f32)> {
    let ch = st.crosshair.as_ref()?;
    let f = st.active_frame()?;

    // Prefer world-space lock when both ends have a WCS.
    if let (Some((ra, dec)), Some(wcs)) = (ch.world, f.fits.wcs.as_ref()) {
        if let Some((px, py)) = wcs.world_to_pix(ra, dec) {
            let (dx, dy) = fits_to_display_oriented(px, py, f);
            return Some((dx, dy));
        }
    }
    // Fall back to raw pixel coords if the crosshair was placed in *this* frame.
    let (idx, fx, fy) = ch.pixel;
    if idx == st.active {
        let (dx, dy) = fits_to_display_oriented(fx, fy, f);
        return Some((dx, dy));
    }
    None
}

fn push_crosshair_to_window(window: &MainWindow, st: &State) {
    match project_crosshair(st) {
        Some((dx, dy)) => {
            window.set_crosshair_x(dx);
            window.set_crosshair_y(dy);
            window.set_crosshair_visible(true);
        }
        None => window.set_crosshair_visible(false),
    }
}

/// Place the session crosshair from a *display-space* (0-based, y-down) point
/// in the active frame. Captures world coords if the frame has a WCS so other
/// frames with a WCS can mirror the same sky position.
fn set_crosshair_at_display(window: &MainWindow, st: &mut State, dx: f64, dy: f64) {
    let active = st.active;
    let (fx_fits, fy_fits, world) = {
        let Some(f) = st.active_frame() else { return };
        let (fx, fy) = display_to_fits(dx, dy, f);
        let world = f.fits.wcs.as_ref().map(|w| w.pix_to_world(fx, fy));
        (fx, fy, world)
    };
    st.crosshair = Some(Crosshair { world, pixel: (active, fx_fits, fy_fits) });
    push_crosshair_to_window(window, st);
    let label = match world {
        Some((ra, dec)) => format!("crosshair: ({fx_fits:.1}, {fy_fits:.1})  α={ra:.6}°  δ={dec:.6}°"),
        None => format!("crosshair: ({fx_fits:.1}, {fy_fits:.1})  [no WCS]"),
    };
    window.set_status_text(label.into());
}

/// min / max / mean / median / std over finite samples.
fn image_stats(img: &FitsImage) -> Option<(f64, f64, f64, f64, f64)> {
    let mut s: Vec<f32> = img.data.iter().copied().filter(|v| v.is_finite()).collect();
    if s.is_empty() { return None; }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let min = s[0] as f64;
    let max = s[n - 1] as f64;
    let median = if n % 2 == 1 {
        s[n / 2] as f64
    } else {
        (s[n / 2 - 1] as f64 + s[n / 2] as f64) * 0.5
    };
    let mean = s.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = s.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n as f64;
    Some((min, max, mean, median, var.sqrt()))
}

/// Save the slint window's pan/zoom into the currently active frame so a frame
/// switch round-trips losslessly.
fn save_view_into_active(window: &MainWindow, st: &mut State) {
    if let Some(f) = st.active_frame_mut() {
        f.view_zoom  = window.get_view_zoom();
        f.view_pan_x = window.get_view_pan_x();
        f.view_pan_y = window.get_view_pan_y();
    }
}

/// Push the active frame's pan/zoom + cursor + WCS into the slint window.
fn push_view_to_window(window: &MainWindow, st: &State) {
    if let Some(f) = st.active_frame() {
        window.set_view_zoom(f.view_zoom);
        window.set_view_pan_x(f.view_pan_x);
        window.set_view_pan_y(f.view_pan_y);
        window.set_cursor_image_x((f.fits.width  / 2) as f32);
        window.set_cursor_image_y((f.fits.height / 2) as f32);
        if let Some(wcs) = &f.fits.wcs {
            let cx = f.fits.width  as f64 / 2.0;
            let cy = f.fits.height as f64 / 2.0;
            let (ra, dec) = wcs.pix_to_world(cx, cy);
            window.set_info_wcs(format!(
                "{} {}", wcs.radesys.to_lowercase(), ds9_fits::format_sexagesimal(ra, dec)
            ).into());
        } else {
            window.set_info_wcs("wcs: ——".into());
        }
    }
}

/// Append a new frame for the given path. Active frame is set to the new one.
fn load_into(window: &MainWindow, st: &mut State, path: &Path) {
    match ds9_fits::load(path) {
        Ok(img) => {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("(fits)")
                .to_string();
            let (w, h, mn, mx) = (img.width, img.height, img.min, img.max);
            // persist the outgoing frame's view so we don't clobber it on switch
            save_view_into_active(window, st);
            let mut fr = Frame::new(img, name);
            fr.source_path = Some(path.to_path_buf());
            apply_prefs_to_frame(&st.prefs, &mut fr);
            st.frames.push(fr);
            st.active = st.frames.len() - 1;
            window.set_status_text(
                format!("loaded {w} × {h}    range {mn:.4} … {mx:.4}    [frame {}]", st.active + 1).into(),
            );
            push_view_to_window(window, st);
            refresh_view(window, st);
        }
        Err(e) => {
            window.set_status_text(format!("error: {e}").into());
        }
    }
}

/// Switch to a different frame index (saving/restoring view state).
fn switch_frame(window: &MainWindow, st: &mut State, target: usize) {
    if target >= st.frames.len() || target == st.active { return; }
    save_view_into_active(window, st);
    st.active = target;
    push_view_to_window(window, st);
    refresh_view(window, st);
    window.set_status_text(
        format!("frame {}/{} — {}",
            st.active + 1, st.frames.len(),
            st.active_frame().map(|f| f.name.as_str()).unwrap_or("—"),
        ).into(),
    );
}

// ---------------------------------------------------------------- regions --

fn region_new_sample(st: &mut State) {
    let Some(f) = st.active_frame_mut() else { return };
    let cx = f.fits.width  as f64 / 2.0;
    let cy = f.fits.height as f64 / 2.0;
    let r  = (f.fits.width.min(f.fits.height) as f64 * 0.05).max(4.0);
    f.markers.push(Marker::circle(cx, cy, r));
}

fn region_load(window: &MainWindow, st: &mut State) {
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Load DS9 region file")
        .add_filter("Region", &["reg"])
        .add_filter("All", &["*"])
        .pick_file();
    let Some(p) = chosen else { return };
    let Some(f) = st.active_frame_mut() else {
        window.set_status_text("no active frame — open a FITS first".into());
        return;
    };
    match std::fs::read_to_string(&p) {
        Ok(text) => {
            let parsed = match f.fits.wcs.as_ref() {
                Some(w) => ds9_marker::parse_reg_with_wcs(&text, w),
                None    => ds9_marker::parse_reg(&text),
            };
            match parsed {
                Ok(ms) => {
                    let n = ms.len();
                    let wcs_note = if f.fits.wcs.is_some() { "" } else { "  [no WCS — sky regions assumed image]" };
                    f.markers = ms;
                    f.selected_marker = None;
                    window.set_status_text(format!(
                        "loaded {n} regions from {}{wcs_note}", p.display()
                    ).into());
                    refresh_view(window, st);
                }
                Err(e) => window.set_status_text(format!("region parse error: {e}").into()),
            }
        }
        Err(e) => window.set_status_text(format!("region read error: {e}").into()),
    }
}

fn catalog_load(window: &MainWindow, st: &mut State) {
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Load source catalog")
        .add_filter("Catalog", &["cat", "tsv", "txt"])
        .add_filter("All", &["*"])
        .pick_file();
    let Some(p) = chosen else { return };
    let Some(f) = st.active_frame_mut() else {
        window.set_status_text("no active frame — open a FITS first".into());
        return;
    };
    match Catalog::from_path(&p) {
        Ok(cat) => {
            let n = cat.len();
            let xy = cat.xy_columns();
            let xy_msg = match xy {
                Some((xi, yi)) => format!(
                    " (X={}, Y={})",
                    cat.columns.get(xi).map(String::as_str).unwrap_or("?"),
                    cat.columns.get(yi).map(String::as_str).unwrap_or("?"),
                ),
                None => " (no X/Y_IMAGE columns — points won't plot)".to_string(),
            };
            window.set_status_text(format!(
                "loaded {n} catalog rows from {}{xy_msg}", p.display()
            ).into());
            f.catalog = Some(cat);
            f.selected_catalog = None;
            refresh_view(window, st);
        }
        Err(e) => window.set_status_text(format!("catalog read error: {e}").into()),
    }
}

/// Try to find a usable SExtractor binary on $PATH. Different distributions
/// install it under different names — `sex` (classic), `source-extractor`
/// (Debian/Ubuntu since `sex` collided with… you know what), or `sextractor`.
fn find_sextractor() -> Option<String> {
    for name in ["source-extractor", "sextractor", "sex"] {
        match std::process::Command::new(name)
            .arg("-v")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(_) => return Some(name.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            // exists but argument failed — still usable
            Err(_) => return Some(name.to_string()),
        }
    }
    None
}

/// Spawn an external SExtractor on the active frame's source FITS, parse the
/// resulting `ASCII_HEAD` catalog, and load it into the frame.
///
/// Defaults: `DETECT_THRESH=1.5`, `DETECT_MINAREA=5`, `BACK_SIZE=64`,
/// `CATALOG_TYPE=ASCII_HEAD`. Override by setting `SEXTRACTOR_OPTS` in the
/// environment — its tokens are appended to the command line so any
/// SExtractor `-KEY VALUE` pair works (e.g. `SEXTRACTOR_OPTS="-DETECT_THRESH 3.0"`).
fn run_sextractor(window: &MainWindow, st: &mut State) {
    let (idx, fits_path, name) = {
        let Some(f) = st.active_frame() else {
            window.set_status_text("sextractor: no active frame".into()); return;
        };
        let Some(p) = f.source_path.clone() else {
            window.set_status_text(
                "sextractor: active frame has no on-disk path (RGB / synthetic)".into()
            );
            return;
        };
        (st.active, p, f.name.clone())
    };

    let bin = match find_sextractor() {
        Some(b) => b,
        None => {
            window.set_status_text(
                "sextractor: binary not found on PATH (tried source-extractor, sextractor, sex)".into()
            );
            return;
        }
    };

    let tmp = std::env::temp_dir()
        .join(format!("ds9-rust-sex-{}-{}", std::process::id(), idx));
    if let Err(e) = std::fs::create_dir_all(&tmp) {
        window.set_status_text(format!("sextractor: tmpdir: {e}").into()); return;
    }
    let cfg = tmp.join("default.sex");
    let par = tmp.join("default.param");
    let cat = tmp.join("out.cat");

    let cfg_text = "\
# ds9-rust default SExtractor config
DETECT_TYPE      CCD
DETECT_MINAREA   5
DETECT_THRESH    1.5
ANALYSIS_THRESH  1.5
FILTER           N
DEBLEND_NTHRESH  32
DEBLEND_MINCONT  0.005
CLEAN            Y
CLEAN_PARAM      1.0
PHOT_APERTURES   5
SATUR_LEVEL      50000.0
MAG_ZEROPOINT    0.0
GAIN             0.0
PIXEL_SCALE      0.0
SEEING_FWHM      1.2
BACK_SIZE        64
BACK_FILTERSIZE  3
BACKPHOTO_TYPE   GLOBAL
";
    let par_text = "\
NUMBER
X_IMAGE
Y_IMAGE
MAG_AUTO
FLUX_AUTO
A_IMAGE
B_IMAGE
THETA_IMAGE
FLAGS
";
    if let Err(e) = std::fs::write(&cfg, cfg_text) {
        window.set_status_text(format!("sextractor: write cfg: {e}").into()); return;
    }
    if let Err(e) = std::fs::write(&par, par_text) {
        window.set_status_text(format!("sextractor: write par: {e}").into()); return;
    }

    let mut cmd = std::process::Command::new(&bin);
    cmd.current_dir(&tmp)
        .arg(&fits_path)
        .arg("-c").arg(&cfg)
        .arg("-CATALOG_NAME").arg(&cat)
        .arg("-CATALOG_TYPE").arg("ASCII_HEAD")
        .arg("-PARAMETERS_NAME").arg(&par);

    if let Ok(opts) = std::env::var("SEXTRACTOR_OPTS") {
        for token in opts.split_whitespace() { cmd.arg(token); }
    }

    window.set_status_text(format!("sextractor: running on {name}…").into());
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => { window.set_status_text(format!("sextractor: spawn: {e}").into()); return; }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let first = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
        window.set_status_text(format!("sextractor failed: {first}").into());
        return;
    }

    match Catalog::from_path(&cat) {
        Ok(c) => {
            let n = c.len();
            if let Some(f) = st.frames.get_mut(idx) {
                f.catalog = Some(c);
                f.selected_catalog = None;
            }
            refresh_view(window, st);
            window.set_status_text(format!(
                "sextractor: detected {n} sources from {name} ({})", bin
            ).into());
        }
        Err(e) => window.set_status_text(format!("sextractor: read catalog: {e}").into()),
    }
}

// -------------------------------------------------------- SAMP (outbound) --

/// Locate the per-user SAMP hub lockfile. Standard locations are
/// `$SAMP_HUB` (URL-style hint), `~/.samp`, or `$XDG_RUNTIME_DIR/samp/...`.
/// We only support the classic plain-file case here.
fn samp_lockfile_path() -> Option<PathBuf> {
    if let Some(env) = env::var_os("SAMP_HUB") {
        let s = env.to_string_lossy().to_string();
        if let Some(rest) = s.strip_prefix("std-lockurl:file://") {
            return Some(PathBuf::from(rest));
        }
        if !s.is_empty() && !s.contains("://") {
            return Some(PathBuf::from(s));
        }
    }
    let home = env::var_os("HOME").map(PathBuf::from)?;
    Some(home.join(".samp"))
}

/// Parse the SAMP lockfile (`key=value` per line, `#` comments). Returns
/// (xmlrpc_url, secret) — both required to register.
fn samp_read_lockfile() -> Result<(String, String), String> {
    let p = samp_lockfile_path().ok_or_else(|| "samp: no $HOME".to_string())?;
    let body = std::fs::read_to_string(&p)
        .map_err(|e| format!("samp: read {}: {e}", p.display()))?;
    let mut url = None;
    let mut secret = None;
    for line in body.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        let Some((k, v)) = t.split_once('=') else { continue };
        match k.trim() {
            "samp.hub.xmlrpc.url" => url = Some(v.trim().to_string()),
            "samp.secret"         => secret = Some(v.trim().to_string()),
            _ => {}
        }
    }
    match (url, secret) {
        (Some(u), Some(s)) => Ok((u, s)),
        _ => Err("samp: lockfile missing xmlrpc url or secret (no hub running?)".into()),
    }
}

/// Escape a string for XML PCDATA / attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
        .replace('"', "&quot;").replace('\'', "&apos;")
}

/// Tiny XML-RPC value serialiser. Supports just the types SAMP uses:
/// string, struct (BTreeMap-like Vec<(K,V)>), and array.
enum XV<'a> {
    Str(&'a str),
    Map(Vec<(&'a str, XV<'a>)>),
}

fn xv_to_xml(v: &XV) -> String {
    match v {
        XV::Str(s) => format!("<value><string>{}</string></value>", xml_escape(s)),
        XV::Map(items) => {
            let mut out = String::from("<value><struct>");
            for (k, val) in items {
                out.push_str("<member><name>");
                out.push_str(&xml_escape(k));
                out.push_str("</name>");
                out.push_str(&xv_to_xml(val));
                out.push_str("</member>");
            }
            out.push_str("</struct></value>");
            out
        }
    }
}

fn xmlrpc_call(url: &str, method: &str, params: &[XV]) -> Result<String, String> {
    let mut body = String::from("<?xml version=\"1.0\"?><methodCall><methodName>");
    body.push_str(&xml_escape(method));
    body.push_str("</methodName><params>");
    for p in params {
        body.push_str("<param>");
        body.push_str(&xv_to_xml(p));
        body.push_str("</param>");
    }
    body.push_str("</params></methodCall>");
    http_agent().post(url)
        .set("Content-Type", "text/xml")
        .send_string(&body)
        .map_err(|e| format!("xml-rpc: {e}"))?
        .into_string()
        .map_err(|e| format!("xml-rpc: read body: {e}"))
}

/// Pull the value of struct member `key` out of an XML-RPC `<struct>`
/// response. Naive substring search — fine for SAMP's flat hub-id /
/// private-key responses.
fn xmlrpc_struct_str(body: &str, key: &str) -> Option<String> {
    let needle = format!("<name>{key}</name>");
    let i = body.find(&needle)?;
    let rest = &body[i + needle.len()..];
    let v_start = rest.find("<string>")? + "<string>".len();
    let v_end = rest[v_start..].find("</string>")?;
    Some(rest[v_start..v_start + v_end].to_string())
}

fn samp_register(url: &str, secret: &str) -> Result<String, String> {
    let body = xmlrpc_call(url, "samp.hub.register", &[XV::Str(secret)])?;
    if body.contains("<fault>") {
        return Err(format!("samp: register fault: {body}"));
    }
    xmlrpc_struct_str(&body, "samp.private-key")
        .ok_or_else(|| format!("samp: register response missing samp.private-key: {body}"))
}

fn samp_unregister(url: &str, priv_key: &str) {
    // best-effort
    let _ = xmlrpc_call(url, "samp.hub.unregister", &[XV::Str(priv_key)]);
}

fn samp_declare_metadata(url: &str, priv_key: &str) -> Result<(), String> {
    let meta = XV::Map(vec![
        ("samp.name",         XV::Str("ds9-rust")),
        ("samp.description.text", XV::Str("ds9-rust — slint port of SAOImage DS9")),
        ("ds9-rust.version",  XV::Str("0.1")),
    ]);
    xmlrpc_call(url, "samp.hub.declareMetadata", &[XV::Str(priv_key), meta])?;
    Ok(())
}

/// Fire a notify-all of the given mtype + params struct. SAMP's hub fans it
/// out to every subscribed client; we do not wait for individual responses.
fn samp_notify_all(url: &str, priv_key: &str, mtype: &str, params: Vec<(&str, XV)>) -> Result<(), String> {
    let msg = XV::Map(vec![
        ("samp.mtype",  XV::Str(mtype)),
        ("samp.params", XV::Map(params)),
    ]);
    let body = xmlrpc_call(url, "samp.hub.notifyAll", &[XV::Str(priv_key), msg])?;
    if body.contains("<fault>") {
        return Err(format!("samp: notifyAll fault: {body}"));
    }
    Ok(())
}

/// Convert a local filesystem path to a `file://` URL. Doesn't try to be
/// clever about Windows; this binary only builds on linux/macOS anyway.
fn path_to_file_url(p: &Path) -> String {
    let abs = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let s = abs.to_string_lossy().replace('\\', "/");
    let mut enc = String::from("file://");
    for ch in s.chars() {
        match ch {
            '/' | 'A'..='Z' | 'a'..='z' | '0'..='9' |
            '-' | '.' | '_' | '~' | ':' => enc.push(ch),
            _ => {
                let mut b = [0u8; 4];
                for byte in ch.encode_utf8(&mut b).bytes() {
                    enc.push_str(&format!("%{:02X}", byte));
                }
            }
        }
    }
    enc
}

/// Send `image.load.fits` for the active frame's source FITS file.
fn samp_send_active_image(window: &MainWindow, st: &State) {
    let Some(f) = st.active_frame() else {
        window.set_status_text("samp: no active frame".into()); return;
    };
    let Some(path) = f.source_path.as_ref() else {
        window.set_status_text("samp: active frame is synthetic (no path)".into()); return;
    };
    let (url, secret) = match samp_read_lockfile() {
        Ok(x) => x, Err(e) => { window.set_status_text(e.into()); return; }
    };
    let key = match samp_register(&url, &secret) {
        Ok(k) => k, Err(e) => { window.set_status_text(e.into()); return; }
    };
    let _ = samp_declare_metadata(&url, &key);

    let furl = path_to_file_url(path);
    let name = f.name.clone();
    let res = samp_notify_all(&url, &key, "image.load.fits", vec![
        ("url",  XV::Str(&furl)),
        ("name", XV::Str(&name)),
    ]);
    samp_unregister(&url, &key);

    match res {
        Ok(()) => window.set_status_text(
            format!("samp: image.load.fits sent ({})", path.display()).into()
        ),
        Err(e) => window.set_status_text(e.into()),
    }
}

/// Send `table.load.votable` for a VOTable file the user picks.
fn samp_send_votable(window: &MainWindow) {
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Send VOTable via SAMP (table.load.votable)")
        .add_filter("VOTable", &["xml", "vot", "votable"])
        .add_filter("All", &["*"])
        .pick_file();
    let Some(p) = chosen else { return };
    let (url, secret) = match samp_read_lockfile() {
        Ok(x) => x, Err(e) => { window.set_status_text(e.into()); return; }
    };
    let key = match samp_register(&url, &secret) {
        Ok(k) => k, Err(e) => { window.set_status_text(e.into()); return; }
    };
    let _ = samp_declare_metadata(&url, &key);
    let furl = path_to_file_url(&p);
    let name = p.file_name().map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "table".into());
    let res = samp_notify_all(&url, &key, "table.load.votable", vec![
        ("url",  XV::Str(&furl)),
        ("name", XV::Str(&name)),
    ]);
    samp_unregister(&url, &key);
    match res {
        Ok(()) => window.set_status_text(
            format!("samp: table.load.votable sent ({})", p.display()).into()
        ),
        Err(e) => window.set_status_text(e.into()),
    }
}

// ----------------------------------------------- online catalog queries --

const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(HTTP_TIMEOUT)
        .user_agent("ds9-rust/0.1 (+https://github.com/anthropics/claude-code)")
        .build()
}

/// Resolve an object name → (RA, Dec) in degrees via CDS Sesame (which itself
/// queries SIMBAD/NED/VizieR in turn). The plain-text endpoint returns lines
/// like `%J 202.4695833 +47.1951667 …` — we just grep for `%J`.
fn sesame_resolve(name: &str) -> Result<(f64, f64), String> {
    let url = format!(
        "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/A?{}",
        urlencode(name)
    );
    let body = http_agent().get(&url).call()
        .map_err(|e| format!("sesame: {e}"))?
        .into_string()
        .map_err(|e| format!("sesame: read body: {e}"))?;
    for line in body.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("%J") else { continue };
        let mut it = rest.split_whitespace();
        let ra: f64 = it.next().and_then(|s| s.parse().ok())
            .ok_or_else(|| "sesame: malformed %J (RA)".to_string())?;
        let dec: f64 = it.next().and_then(|s| s.parse().ok())
            .ok_or_else(|| "sesame: malformed %J (Dec)".to_string())?;
        return Ok((ra, dec));
    }
    Err(format!("sesame: no %J line in response (name '{name}' unresolved)"))
}

/// Cone-search a VizieR catalog at (ra, dec) within `radius_deg`. Returns a
/// VOTable XML body which the existing `Catalog::from_votable` parser handles.
fn vizier_cone(ra: f64, dec: f64, radius_deg: f64, src: &str) -> Result<String, String> {
    let url = format!(
        "https://vizier.cds.unistra.fr/viz-bin/votable?-source={}&-c={}+{:+.6}&-c.rd={}&-out.max=500",
        urlencode(src), ra, dec, radius_deg
    );
    http_agent().get(&url).call()
        .map_err(|e| format!("vizier: {e}"))?
        .into_string()
        .map_err(|e| format!("vizier: read body: {e}"))
}

/// NED cone search at (ra, dec) → ASCII tab-separated table. NED's "tabular"
/// output starts with a few preamble lines, then a `No.\tObject Name…` header
/// row followed by data rows.
fn ned_cone(ra: f64, dec: f64, radius_deg: f64) -> Result<String, String> {
    // NED's classic objsearch endpoint: in_csys=Equatorial, in_equinox=J2000.0
    let url = format!(
        "https://ned.ipac.caltech.edu/cgi-bin/objsearch?\
         search_type=Near+Position+Search&\
         in_csys=Equatorial&in_equinox=J2000.0&\
         lon={:.6}d&lat={:+.6}d&radius={:.4}&\
         of=ascii_bar&list_limit=500",
        ra, dec, radius_deg * 60.0  // NED expects radius in arcmin
    );
    let body = http_agent().get(&url).call()
        .map_err(|e| format!("ned: {e}"))?
        .into_string()
        .map_err(|e| format!("ned: read body: {e}"))?;
    // NED's `ascii_bar` is `|`-separated with a header row beginning "No.|".
    // Convert to TSV for our existing parser.
    let mut header_seen = false;
    let mut out = String::new();
    for line in body.lines() {
        let t = line.trim_start();
        if !header_seen {
            if t.starts_with("No.|") || t.starts_with("No|") {
                header_seen = true;
                let row: String = t.split('|').map(|s| s.trim()).collect::<Vec<_>>().join("\t");
                out.push_str(&row);
                out.push('\n');
            }
            continue;
        }
        if t.is_empty() || !t.contains('|') { continue }
        let row: String = t.split('|').map(|s| s.trim()).collect::<Vec<_>>().join("\t");
        out.push_str(&row);
        out.push('\n');
    }
    if out.is_empty() {
        return Err("ned: no tabular results in response".into());
    }
    Ok(out)
}

/// Minimal RFC-3986-ish URL encoding for query-string values (no external
/// crate). Encodes everything outside the unreserved set.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' |
            b'-' | b'.' | b'_' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Returns the world coordinates of the active frame's image centre, if it
/// has a WCS. Otherwise `None`.
fn active_wcs_center(st: &State) -> Option<(f64, f64)> {
    let f = st.active_frame()?;
    let w = f.fits.wcs.as_ref()?;
    let cx = (f.fits.width  as f64) * 0.5 + 0.5;
    let cy = (f.fits.height as f64) * 0.5 + 0.5;
    Some(w.pix_to_world(cx, cy))
}

fn netcat_resolve(window: &MainWindow, st: &mut State) {
    let name = window.get_netcat_name().as_str().trim().to_string();
    if name.is_empty() {
        window.set_netcat_status("resolve: enter an object name first".into());
        return;
    }
    window.set_netcat_status(format!("resolving '{name}' via Sesame…").into());
    match sesame_resolve(&name) {
        Ok((ra, dec)) => {
            // If the active frame has a WCS, recenter the view on (ra, dec)
            // and drop a session crosshair.
            let placed = {
                let info: Option<(f64, f64, f32, f32)> = st.active_frame().and_then(|f| {
                    let wcs = f.fits.wcs.as_ref()?;
                    let (px, py) = wcs.world_to_pix(ra, dec)?;
                    let (dx, dy) = fits_to_display_oriented(px, py, f);
                    Some((px, py, dx, dy))
                });
                if let Some((px, py, dx, dy)) = info {
                    st.crosshair = Some(Crosshair {
                        world: Some((ra, dec)),
                        pixel: (st.active, px, py),
                    });
                    let cw = window.get_canvas_w() as f64;
                    let ch = window.get_canvas_h() as f64;
                    let z  = window.get_view_zoom() as f64;
                    window.set_view_pan_x((cw * 0.5 - dx as f64 * z) as f32);
                    window.set_view_pan_y((ch * 0.5 - dy as f64 * z) as f32);
                    true
                } else { false }
            };
            push_crosshair_to_window(window, st);
            refresh_view(window, st);
            let extra = if placed { "  (crosshair set)" } else { "  (no WCS to centre)" };
            window.set_netcat_status(format!(
                "{name} → RA {ra:.6}  Dec {dec:+.6}{extra}"
            ).into());
            window.set_status_text(format!(
                "resolved {name}: ({ra:.5}, {dec:+.5}){extra}"
            ).into());
        }
        Err(e) => {
            window.set_netcat_status(e.clone().into());
            window.set_status_text(e.into());
        }
    }
}

fn parse_radius(window: &MainWindow) -> Result<f64, String> {
    let s = window.get_netcat_radius().as_str().trim().to_string();
    s.parse::<f64>().ok()
        .filter(|r| *r > 0.0 && *r < 30.0)
        .ok_or_else(|| format!("invalid radius '{s}' (deg, 0 < r < 30)"))
}

fn netcat_vizier(window: &MainWindow, st: &mut State) {
    let Some((ra, dec)) = active_wcs_center(st) else {
        window.set_netcat_status(
            "vizier: active frame needs a WCS (so we know where to point)".into()
        );
        return;
    };
    let radius = match parse_radius(window) { Ok(r) => r, Err(e) => {
        window.set_netcat_status(format!("vizier: {e}").into()); return;
    }};
    let src = window.get_netcat_vizid().as_str().trim().to_string();
    if src.is_empty() {
        window.set_netcat_status("vizier: enter a catalog ID (e.g. I/345/gaia2)".into());
        return;
    }
    window.set_netcat_status(format!(
        "vizier: querying {src} @ ({ra:.4}, {dec:+.4}) r={radius} deg…"
    ).into());
    match vizier_cone(ra, dec, radius, &src) {
        Ok(body) => {
            let cat = ds9_catalog::Catalog::from_votable(&body);
            let n = cat.len();
            if let Some(f) = st.active_frame_mut() {
                f.catalog = Some(cat);
                f.selected_catalog = None;
            }
            refresh_view(window, st);
            window.set_netcat_status(format!("vizier: {n} rows from {src}").into());
            window.set_status_text(format!("vizier {src}: {n} rows").into());
        }
        Err(e) => {
            window.set_netcat_status(e.clone().into());
            window.set_status_text(e.into());
        }
    }
}

fn netcat_ned(window: &MainWindow, st: &mut State) {
    let Some((ra, dec)) = active_wcs_center(st) else {
        window.set_netcat_status("ned: active frame needs a WCS".into()); return;
    };
    let radius = match parse_radius(window) { Ok(r) => r, Err(e) => {
        window.set_netcat_status(format!("ned: {e}").into()); return;
    }};
    window.set_netcat_status(format!(
        "ned: querying near ({ra:.4}, {dec:+.4}) r={radius} deg…"
    ).into());
    match ned_cone(ra, dec, radius) {
        Ok(tsv) => {
            let cat = ds9_catalog::Catalog::from_tsv(&tsv);
            let n = cat.len();
            if let Some(f) = st.active_frame_mut() {
                f.catalog = Some(cat);
                f.selected_catalog = None;
            }
            refresh_view(window, st);
            window.set_netcat_status(format!("ned: {n} rows").into());
            window.set_status_text(format!("ned: {n} rows").into());
        }
        Err(e) => {
            window.set_netcat_status(e.clone().into());
            window.set_status_text(e.into());
        }
    }
}

fn catalog_clear(window: &MainWindow, st: &mut State) {
    if let Some(f) = st.active_frame_mut() {
        f.catalog = None;
        f.selected_catalog = None;
    }
    window.set_status_text("catalog cleared".into());
    refresh_view(window, st);
}

fn region_save(window: &MainWindow, st: &State) {
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Save DS9 region file")
        .set_file_name("regions.reg")
        .add_filter("Region", &["reg"])
        .save_file();
    let Some(p) = chosen else { return };
    let Some(f) = st.active_frame() else {
        window.set_status_text("no active frame".into());
        return;
    };
    match ds9_marker::write_reg(&p, &f.markers) {
        Ok(()) => window.set_status_text(format!("wrote {} regions → {}", f.markers.len(), p.display()).into()),
        Err(e) => window.set_status_text(format!("region write error: {e}").into()),
    }
}

// ---------------------------------------------------------------- composite --

/// Render the first three frames as the R / G / B channels of a single image.
/// Returns `None` if their dimensions don't match.
fn build_rgb_composite(frames: &[Frame]) -> Option<(Image, usize, usize)> {
    let f0 = &frames[0];
    let (w, h) = (f0.fits.width, f0.fits.height);
    if frames.iter().any(|f| f.fits.width != w || f.fits.height != h) {
        return None;
    }
    let render = |f: &Frame| -> Vec<u8> {
        let lim = f.limits();
        ds9_image::render_rgba_flipped(&f.fits, lim, f.stretch, Colormap::Grey)
    };
    let r = render(&frames[0]);
    let g = render(&frames[1]);
    let b = render(&frames[2]);
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        let off = i * 4;
        out[off]     = r[off];           // red channel from frame 0 luminance
        out[off + 1] = g[off + 1].max(g[off]);
        out[off + 2] = b[off + 2].max(b[off]);
        out[off + 3] = 255;
    }
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
    buf.make_mut_bytes().copy_from_slice(&out);
    Some((Image::from_rgba8(buf), w, h))
}

// ---------------------------------------------------------------- mosaic --

/// Build a WCS mosaic from every frame in `frames` that carries a TAN WCS.
/// The first such frame's WCS (CRVAL + CD matrix) is reused verbatim as the
/// output projection so pixel scale, orientation, and projection centre stay
/// the same. The output canvas is sized to the bounding box of all input
/// frames' corners in that reference WCS, and CRPIX is shifted so output
/// pixel (1, 1) maps to the bbox start.
///
/// Sampling is nearest-neighbour; overlapping pixels are averaged. Returns
/// `None` if fewer than two frames have a WCS, or if the resulting canvas
/// would exceed `MAX_MOSAIC_PX` pixels.
fn build_mosaic(frames: &[Frame]) -> Result<FitsImage, String> {
    const MAX_MOSAIC_PX: usize = 64_000_000; // ~256 MiB at f32, sanity cap.

    let with_wcs: Vec<&Frame> = frames.iter().filter(|f| f.fits.wcs.is_some()).collect();
    if with_wcs.len() < 2 {
        return Err("mosaic: need at least 2 frames with WCS".into());
    }
    let ref_wcs = with_wcs[0].fits.wcs.as_ref().unwrap();

    // Project every frame's 4 corners (1-based) → world → ref-WCS pixel.
    // CRPIX of the *temporary* reference is its own CRPIX; we'll shift after
    // we know the bbox.
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for f in &with_wcs {
        let w = f.fits.width  as f64;
        let h = f.fits.height as f64;
        let corners = [(1.0, 1.0), (w, 1.0), (1.0, h), (w, h)];
        let fw = f.fits.wcs.as_ref().unwrap();
        for (x, y) in corners {
            let (ra, dec) = fw.pix_to_world(x, y);
            if let Some((rx, ry)) = ref_wcs.world_to_pix(ra, dec) {
                if rx.is_finite() && ry.is_finite() {
                    min_x = min_x.min(rx);
                    min_y = min_y.min(ry);
                    max_x = max_x.max(rx);
                    max_y = max_y.max(ry);
                }
            }
        }
    }
    if !(min_x.is_finite() && max_x.is_finite()) {
        return Err("mosaic: could not project any corner — check WCS".into());
    }
    let out_w = ((max_x - min_x).ceil() as usize).saturating_add(1).max(1);
    let out_h = ((max_y - min_y).ceil() as usize).saturating_add(1).max(1);
    let total = out_w.checked_mul(out_h).ok_or("mosaic: overflow")?;
    if total > MAX_MOSAIC_PX {
        return Err(format!(
            "mosaic: output {out_w}×{out_h} = {total} px exceeds cap ({MAX_MOSAIC_PX})"
        ));
    }

    // Output WCS = ref WCS with CRPIX shifted so that output pixel (1,1) maps
    // to ref-WCS pixel (min_x, min_y).
    let mut out_wcs = ref_wcs.clone();
    out_wcs.crpix1 = ref_wcs.crpix1 - min_x + 1.0;
    out_wcs.crpix2 = ref_wcs.crpix2 - min_y + 1.0;

    let mut sum   = vec![0.0_f64; total];
    let mut count = vec![0_u32;   total];

    for f in &with_wcs {
        let fw = f.fits.wcs.as_ref().unwrap();
        let fw_w = f.fits.width  as i64;
        let fw_h = f.fits.height as i64;
        for oy in 0..out_h {
            for ox in 0..out_w {
                // 1-based output pixel
                let (ra, dec) = out_wcs.pix_to_world(ox as f64 + 1.0, oy as f64 + 1.0);
                let Some((fx, fy)) = fw.world_to_pix(ra, dec) else { continue };
                if !(fx.is_finite() && fy.is_finite()) { continue }
                // 1-based → 0-based nearest-neighbour
                let ix = fx.round() as i64 - 1;
                let iy = fy.round() as i64 - 1;
                if ix < 0 || iy < 0 || ix >= fw_w || iy >= fw_h { continue }
                let v = f.fits.data[(iy as usize) * (fw_w as usize) + ix as usize];
                if !v.is_finite() { continue }
                let oi = oy * out_w + ox;
                sum[oi]   += v as f64;
                count[oi] += 1;
            }
        }
    }

    let mut data = vec![f32::NAN; total];
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for i in 0..total {
        if count[i] > 0 {
            let v = (sum[i] / count[i] as f64) as f32;
            data[i] = v;
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
    }
    if !mn.is_finite() { mn = 0.0; mx = 1.0; }

    Ok(FitsImage {
        width:  out_w,
        height: out_h,
        data,
        min: mn,
        max: mx,
        wcs: Some(out_wcs),
    })
}

// ---------------------------------------------------------------- export --

/// Save the active frame's rendered RGBA as a PNG.
fn save_image_png(window: &MainWindow, st: &State) {
    let Some(f) = st.active_frame() else {
        window.set_status_text("save image: no active frame".into()); return;
    };
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Save image as PNG")
        .set_file_name(format!("{}.png",
            f.name.trim_end_matches(".fits").trim_end_matches(".fit")))
        .add_filter("PNG", &["png"])
        .save_file();
    let Some(p) = chosen else { return };
    let rgba = render_rgba_for_frame(f);
    let w = f.fits.width as u32;
    let h = f.fits.height as u32;
    let file = match std::fs::File::create(&p) {
        Ok(f) => f,
        Err(e) => { window.set_status_text(format!("save image: {e}").into()); return; }
    };
    let bw = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(bw, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    match enc.write_header().and_then(|mut wh| wh.write_image_data(&rgba)) {
        Ok(()) => window.set_status_text(format!("wrote {} ({}×{})", p.display(), w, h).into()),
        Err(e) => window.set_status_text(format!("save image: {e}").into()),
    }
}

/// Save the active frame's data as a minimal FITS (BITPIX=-32, NAXIS=2). The
/// header preserves CRPIX/CRVAL/CD if present so the saved file is still WCS-
/// usable in DS9 / ds9-rust.
fn save_image_fits(window: &MainWindow, st: &State) {
    let Some(f) = st.active_frame() else {
        window.set_status_text("save fits: no active frame".into()); return;
    };
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Save image as FITS")
        .set_file_name("image.fits")
        .add_filter("FITS", &["fits", "fit"])
        .save_file();
    let Some(p) = chosen else { return };
    match write_basic_fits(&p, &f.fits) {
        Ok(()) => window.set_status_text(format!("wrote {}", p.display()).into()),
        Err(e) => window.set_status_text(format!("save fits: {e}").into()),
    }
}

fn write_basic_fits(path: &Path, img: &FitsImage) -> std::io::Result<()> {
    use std::io::Write;
    let mut hdr: Vec<String> = Vec::new();
    let card = |k: &str, v: &str| format!("{:<8}= {:<70}", k, v);
    hdr.push(card("SIMPLE",  "T"));
    hdr.push(card("BITPIX",  "-32"));
    hdr.push(card("NAXIS",   "2"));
    hdr.push(card("NAXIS1",  &img.width.to_string()));
    hdr.push(card("NAXIS2",  &img.height.to_string()));
    hdr.push(card("BSCALE",  "1.0"));
    hdr.push(card("BZERO",   "0.0"));
    if let Some(w) = &img.wcs {
        hdr.push(card("CTYPE1",  &format!("'{:<8}'", w.ctype1)));
        hdr.push(card("CTYPE2",  &format!("'{:<8}'", w.ctype2)));
        hdr.push(card("CRPIX1",  &format!("{:.10E}", w.crpix1)));
        hdr.push(card("CRPIX2",  &format!("{:.10E}", w.crpix2)));
        hdr.push(card("CRVAL1",  &format!("{:.10E}", w.crval1)));
        hdr.push(card("CRVAL2",  &format!("{:.10E}", w.crval2)));
        hdr.push(card("CD1_1",   &format!("{:.10E}", w.cd11)));
        hdr.push(card("CD1_2",   &format!("{:.10E}", w.cd12)));
        hdr.push(card("CD2_1",   &format!("{:.10E}", w.cd21)));
        hdr.push(card("CD2_2",   &format!("{:.10E}", w.cd22)));
        hdr.push(card("RADESYS", &format!("'{:<8}'", w.radesys)));
    }
    hdr.push("END".to_string() + &" ".repeat(77));

    let mut bytes: Vec<u8> = Vec::new();
    for c in &hdr {
        let mut s = c.as_bytes().to_vec();
        s.resize(80, b' ');
        bytes.extend_from_slice(&s);
    }
    while bytes.len() % 2880 != 0 { bytes.push(b' '); }
    // FITS data is big-endian, written top-to-bottom in *FITS* orientation
    // (row 0 = bottom). Our in-memory data has row 0 = top after `pix_to_world`
    // conventions we maintain — but since FITS storage is row-major bottom-up
    // and we already feed the renderer with `(h - 1 - y)`, the in-memory order
    // matches FITS-on-disk. Just stream `f32 BE`.
    for &v in &img.data {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    let pad = (2880 - bytes.len() % 2880) % 2880;
    bytes.extend(std::iter::repeat(0u8).take(pad));
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

// ---------------------------------------------------------------- locks --

/// Apply each currently-on frame-lock from the active frame to every other
/// frame. Called when a Lock toggle flips on, and after any menu-driven change
/// that should propagate (cmap, scale, …). Continuous pan/zoom is handled
/// separately via the slint view-changed callback.
fn broadcast_locks(st: &mut State) {
    let Some(active) = st.active_frame() else { return };
    let zoom  = active.view_zoom;
    let pan_x = active.view_pan_x;
    let pan_y = active.view_pan_y;
    let cmap  = active.cmap;
    let stretch = active.stretch;
    let limits  = active.limits_mode;
    let active_idx = st.active;
    let (lz, lp, lc, ls) = (st.lock_zoom, st.lock_pan, st.lock_cmap, st.lock_scale);
    for (i, fr) in st.frames.iter_mut().enumerate() {
        if i == active_idx { continue; }
        if lz { fr.view_zoom  = zoom; }
        if lp { fr.view_pan_x = pan_x; fr.view_pan_y = pan_y; }
        if lc { fr.cmap = cmap; fr.custom_lut = None; }
        if ls { fr.stretch = stretch; fr.limits_mode = limits; }
    }
}

/// Push the active frame's colormap to every other frame (used when
/// `lock_cmap` is on and the user picks a new color).
fn broadcast_cmap(st: &mut State) {
    let Some(active) = st.active_frame() else { return };
    let cmap = active.cmap;
    let active_idx = st.active;
    for (i, fr) in st.frames.iter_mut().enumerate() {
        if i == active_idx { continue; }
        fr.cmap = cmap;
        fr.custom_lut = None;
    }
}

// ---------------------------------------------------------------- export --

/// Write a minimal uncompressed RGB TIFF (8-bit, top-down). No compression,
/// no extra tags beyond the bare minimum the spec requires for a viewer to
/// open the file.
fn write_tiff_rgb(path: &Path, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    use std::io::Write;
    // Convert RGBA → RGB
    let mut rgb: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    // Header: little-endian "II", magic 42, IFD offset 8.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"II");
    bytes.extend_from_slice(&42u16.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    // IFD: 12 entries × 12 bytes = 144 bytes; header is 8; IFD starts at 8;
    // strip data starts after IFD + next-offset (4 bytes) + extras.
    // We need offsets for: BitsPerSample (3 SHORTs => 6 bytes value), and
    // StripOffsets when only one strip → fits in 4-byte field.
    let bps_offset: u32   = 8 + 2 + (12 * 12) + 4;  // after IFD + nextIFD
    let strip_offset: u32 = bps_offset + 6;
    let strip_bytes: u32  = (w * h * 3) as u32;

    let entries: [(u16, u16, u32, u32); 12] = [
        // tag, type, count, value/offset
        (0x00FE, 4, 1, 0),                 // NewSubfileType: 0
        (0x0100, 4, 1, w),                 // ImageWidth
        (0x0101, 4, 1, h),                 // ImageLength
        (0x0102, 3, 3, bps_offset),        // BitsPerSample (8,8,8)
        (0x0103, 3, 1, 1),                 // Compression: none
        (0x0106, 3, 1, 2),                 // PhotometricInterpretation: RGB
        (0x0111, 4, 1, strip_offset),      // StripOffsets
        (0x0115, 3, 1, 3),                 // SamplesPerPixel
        (0x0116, 4, 1, h),                 // RowsPerStrip
        (0x0117, 4, 1, strip_bytes),       // StripByteCounts
        (0x011C, 3, 1, 1),                 // PlanarConfiguration: chunky
        (0x0153, 3, 3, bps_offset),        // SampleFormat (1,1,1) — re-uses BPS slot
    ];

    bytes.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, typ, count, val) in entries {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&typ.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        // For SHORT count 3 we wrote an offset (bps_offset); for SHORT count 1
        // the value goes in the low 2 bytes — but since we wrote a u32 directly
        // it works out either way (LE).
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());     // next-IFD = 0

    // BitsPerSample / SampleFormat shared slot: 3 SHORTs = 6 bytes
    bytes.extend_from_slice(&8u16.to_le_bytes());
    bytes.extend_from_slice(&8u16.to_le_bytes());
    bytes.extend_from_slice(&8u16.to_le_bytes());

    // Pixel data
    bytes.extend_from_slice(&rgb);

    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)
}

/// Write the rendered image as an EPSF — minimal PostScript that wraps an
/// 8-bit RGB image stream. No fonts / titles / margins; the viewer figures
/// out scaling from the bounding box.
fn write_postscript_rgb(path: &Path, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    use std::fmt::Write as _;
    use std::io::Write;
    // Page is letter, 72 dpi. Fit the image into a 540×720 box centred.
    let target_w = 540.0_f64.min(w as f64 * 2.0);
    let scale = target_w / w as f64;
    let target_h = h as f64 * scale;
    let xl = (612.0 - target_w) / 2.0;
    let yl = (792.0 - target_h) / 2.0;

    let mut s = String::new();
    let _ = writeln!(s, "%!PS-Adobe-3.0 EPSF-3.0");
    let _ = writeln!(s, "%%BoundingBox: {xl:.0} {yl:.0} {xr:.0} {yr:.0}",
        xr = xl + target_w, yr = yl + target_h);
    let _ = writeln!(s, "%%Pages: 1");
    let _ = writeln!(s, "%%EndComments");
    let _ = writeln!(s, "/picstr {} string def", w * 3);
    let _ = writeln!(s, "{xl} {yl} translate");
    let _ = writeln!(s, "{target_w:.2} {target_h:.2} scale");
    let _ = writeln!(s, "{w} {h} 8");
    // Image matrix: flip Y so row 0 is on top.
    let _ = writeln!(s, "[ {w} 0 0 -{h} 0 {h} ]");
    let _ = writeln!(s, "{{ currentfile picstr readhexstring pop }}");
    let _ = writeln!(s, "false 3 colorimage");
    let mut f = std::fs::File::create(path)?;
    f.write_all(s.as_bytes())?;
    // hex stream — 60 chars per line keeps the PS readable
    let mut col = 0;
    for px in rgba.chunks_exact(4) {
        for &b in &px[..3] {
            f.write_all(&[hex_nib(b >> 4), hex_nib(b & 0x0f)])?;
            col += 2;
            if col >= 60 { f.write_all(b"\n")?; col = 0; }
        }
    }
    if col != 0 { f.write_all(b"\n")?; }
    f.write_all(b"\n%%EOF\n")?;
    Ok(())
}

fn hex_nib(n: u8) -> u8 {
    if n < 10 { b'0' + n } else { b'a' + (n - 10) }
}

// -------------------------------------------------- region property editor --

/// Encode a marker color (RGBA u8) as `"#rrggbb"`. Alpha is dropped — we
/// don't expose it in the editor.
fn color_to_hex(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Parse `"#rrggbb"` (or `"rrggbb"`) into an RGB triple, alpha defaults to
/// the marker's existing alpha. Returns None on a malformed string.
fn parse_hex_rgb(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 { return None; }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some([r, g, b])
}

fn fmt_num(v: f64) -> slint::SharedString { format!("{:.2}", v).into() }

/// Push the active frame's selected marker into the slint property editor
/// fields. Sets the kind label, x/y/size/angle/color/text strings, and an
/// informational status row. Empty kind means "no selection".
fn populate_region_props(window: &MainWindow, st: &State) {
    let Some(f) = st.active_frame() else {
        window.set_region_prop_kind("—".into());
        window.set_region_prop_status("(no active frame)".into());
        return;
    };
    let Some(idx) = f.selected_marker else {
        window.set_region_prop_kind("(no selection)".into());
        window.set_region_prop_status(
            "click a region in edit mode to select, then reopen Properties…".into()
        );
        return;
    };
    let Some(m) = f.markers.get(idx) else { return };
    let (kind, x, y, s1, s2, ang, text) = match &m.shape {
        MShape::Circle { center, r } => (
            "Circle", center.x, center.y, *r, 0.0, 0.0, String::new(),
        ),
        MShape::Box { center, w, h, theta_deg } => (
            "Box", center.x, center.y, *w, *h, *theta_deg, String::new(),
        ),
        MShape::Ellipse { center, a, b, theta_deg } => (
            "Ellipse", center.x, center.y, *a, *b, *theta_deg, String::new(),
        ),
        MShape::Annulus { center, r_inner, r_outer } => (
            "Annulus", center.x, center.y, *r_outer, *r_inner, 0.0, String::new(),
        ),
        MShape::Point { center } => (
            "Point", center.x, center.y, 0.0, 0.0, 0.0, String::new(),
        ),
        MShape::Line { from, to } => (
            "Line", from.x, from.y, to.x, to.y, 0.0, String::new(),
        ),
        MShape::Compass { center, len } => (
            "Compass", center.x, center.y, *len, 0.0, 0.0, String::new(),
        ),
        MShape::Text { center, body } => (
            "Text", center.x, center.y, 0.0, 0.0, 0.0, body.clone(),
        ),
        MShape::Polygon { points } => {
            window.set_region_prop_kind(format!("Polygon ({} pts, read-only)", points.len()).into());
            window.set_region_prop_x(points.first().map(|p| fmt_num(p.x)).unwrap_or_default());
            window.set_region_prop_y(points.first().map(|p| fmt_num(p.y)).unwrap_or_default());
            window.set_region_prop_size1("".into());
            window.set_region_prop_size2("".into());
            window.set_region_prop_angle("".into());
            window.set_region_prop_color(color_to_hex(m.color).into());
            window.set_region_prop_text("".into());
            window.set_region_prop_status(
                "polygons can only have their color edited from this panel".into()
            );
            return;
        }
    };
    window.set_region_prop_kind(format!("#{} {}", idx + 1, kind).into());
    window.set_region_prop_x(fmt_num(x));
    window.set_region_prop_y(fmt_num(y));
    window.set_region_prop_size1(if s1 == 0.0 && !matches!(m.shape, MShape::Line { .. }) { "".into() } else { fmt_num(s1) });
    window.set_region_prop_size2(if s2 == 0.0 && !matches!(m.shape, MShape::Line { .. }) { "".into() } else { fmt_num(s2) });
    window.set_region_prop_angle(if ang == 0.0 && !matches!(m.shape, MShape::Box { .. } | MShape::Ellipse { .. }) { "".into() } else { fmt_num(ang) });
    window.set_region_prop_color(color_to_hex(m.color).into());
    window.set_region_prop_text(text.into());
    window.set_region_prop_status(
        format!("editing region #{} ({}). press Enter or Apply to commit.", idx + 1, kind).into()
    );
}

/// Read the slint editor fields and write them back into the active frame's
/// selected marker. Logs the outcome to the panel's status row.
fn apply_region_props(window: &MainWindow, st: &mut State) {
    let Some(idx) = st.active_frame().and_then(|f| f.selected_marker) else {
        window.set_region_prop_status("apply: no selection".into()); return;
    };
    let parse = |s: slint::SharedString| -> Option<f64> {
        let s = s.as_str().trim();
        if s.is_empty() { None } else { s.parse::<f64>().ok() }
    };
    let x   = parse(window.get_region_prop_x());
    let y   = parse(window.get_region_prop_y());
    let s1  = parse(window.get_region_prop_size1());
    let s2  = parse(window.get_region_prop_size2());
    let ang = parse(window.get_region_prop_angle());
    let color = parse_hex_rgb(window.get_region_prop_color().as_str());
    let body  = window.get_region_prop_text().as_str().to_string();

    let mut applied = Vec::<&'static str>::new();
    let Some(f) = st.active_frame_mut() else { return };
    let Some(m) = f.markers.get_mut(idx) else { return };
    if let Some([r, g, b]) = color {
        m.color = [r, g, b, m.color[3]];
        applied.push("color");
    }
    use ds9_marker::PixelPos;
    match &mut m.shape {
        MShape::Circle { center, r } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if let Some(v) = s1 { *r = v; applied.push("r"); }
        }
        MShape::Box { center, w, h, theta_deg } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if let Some(v) = s1 { *w = v; applied.push("w"); }
            if let Some(v) = s2 { *h = v; applied.push("h"); }
            if let Some(v) = ang { *theta_deg = v; applied.push("θ"); }
        }
        MShape::Ellipse { center, a, b, theta_deg } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if let Some(v) = s1 { *a = v; applied.push("a"); }
            if let Some(v) = s2 { *b = v; applied.push("b"); }
            if let Some(v) = ang { *theta_deg = v; applied.push("θ"); }
        }
        MShape::Annulus { center, r_inner, r_outer } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if let Some(v) = s1 { *r_outer = v; applied.push("r_outer"); }
            if let Some(v) = s2 { *r_inner = v; applied.push("r_inner"); }
        }
        MShape::Point { center } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
        }
        MShape::Line { from, to } => {
            if let Some(v) = x { from.x = v; applied.push("from.x"); }
            if let Some(v) = y { from.y = v; applied.push("from.y"); }
            if let Some(v) = s1 { to.x = v; applied.push("to.x"); }
            if let Some(v) = s2 { to.y = v; applied.push("to.y"); }
        }
        MShape::Compass { center, len } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if let Some(v) = s1 { *len = v; applied.push("len"); }
        }
        MShape::Text { center, body: existing } => {
            if let Some(v) = x { center.x = v; applied.push("x"); }
            if let Some(v) = y { center.y = v; applied.push("y"); }
            if !body.is_empty() && body != *existing {
                *existing = body.clone();
                applied.push("text");
            }
        }
        MShape::Polygon { .. } => {
            // color already applied above; geometry is read-only here
            let _ = (x, y, s1, s2, ang);
            let _ = PixelPos { x: 0.0, y: 0.0 };
        }
    }
    let msg = if applied.is_empty() {
        format!("apply: nothing changed")
    } else {
        format!("applied: {}", applied.join(", "))
    };
    window.set_region_prop_status(msg.into());
    refresh_view(window, st);
}

// -------------------------------------------------------- region analysis --

/// Iterate (x_fits, y_fits, value) over every finite pixel inside a circular
/// aperture of radius `r` centered at (cx, cy) in FITS coords. Skips
/// out-of-bounds and NaN samples.
fn aperture_iter<'a>(
    f: &'a Frame, cx: f64, cy: f64, r: f64,
) -> impl Iterator<Item = (f64, f64, f32)> + 'a {
    let (w, h) = (f.fits.width, f.fits.height);
    let r2 = r * r;
    let x_lo = ((cx - r).floor() as isize).max(1);
    let x_hi = ((cx + r).ceil()  as isize).min(w as isize);
    let y_lo = ((cy - r).floor() as isize).max(1);
    let y_hi = ((cy + r).ceil()  as isize).min(h as isize);
    (y_lo..=y_hi).flat_map(move |yf| {
        (x_lo..=x_hi).filter_map(move |xf| {
            let dx = xf as f64 - cx;
            let dy = yf as f64 - cy;
            if dx * dx + dy * dy > r2 { return None; }
            // FITS y is bottom-up; storage is top-down.
            let ix = (xf - 1) as usize;
            let iy = h - yf as usize;
            let v = f.fits.data[iy * w + ix];
            if !v.is_finite() { return None; }
            Some((xf as f64, yf as f64, v))
        })
    })
}

/// Intensity-weighted centroid of pixels in `aperture` whose value is above
/// the local background (= aperture median). Returns (cx, cy) in FITS coords,
/// or None if there are too few above-background pixels.
fn compute_centroid(f: &Frame, cx: f64, cy: f64, r: f64) -> Option<(f64, f64)> {
    let mut samples: Vec<(f64, f64, f32)> = aperture_iter(f, cx, cy, r).collect();
    if samples.len() < 4 { return None; }
    samples.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let bg = samples[samples.len() / 2].2;
    let mut sw = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut n = 0_usize;
    for (x, y, v) in &samples {
        let w_ = (*v - bg).max(0.0) as f64;
        if w_ <= 0.0 { continue; }
        sw += w_;
        sx += w_ * x;
        sy += w_ * y;
        n += 1;
    }
    if n < 3 || sw <= 0.0 { return None; }
    Some((sx / sw, sy / sw))
}

/// Mean pixel value per radial bin in [0, r_max], with `n_bins` equal-width
/// bins. Empty bins return NaN. Returns the bin centers so plot code can use
/// them as labels.
fn compute_radial_profile(
    f: &Frame, cx: f64, cy: f64, r_max: f64, n_bins: usize,
) -> Vec<f32> {
    let mut sum = vec![0.0_f64; n_bins];
    let mut cnt = vec![0_u64;   n_bins];
    let r_max = r_max.max(1.0);
    for (x, y, v) in aperture_iter(f, cx, cy, r_max) {
        let dx = x - cx;
        let dy = y - cy;
        let r = (dx * dx + dy * dy).sqrt();
        let b = ((r / r_max) * n_bins as f64).floor() as usize;
        let b = b.min(n_bins - 1);
        sum[b] += v as f64;
        cnt[b] += 1;
    }
    sum.iter().zip(cnt.iter()).map(|(s, &c)| {
        if c == 0 { f32::NAN } else { (s / c as f64) as f32 }
    }).collect()
}

/// Project the selected marker (or the only marker) into a (cx, cy, r_eff)
/// triple suitable for centroid / radial-profile aperture math. Returns None
/// if the shape doesn't have a natural circular extent (Polygon / Line / …).
fn marker_aperture(m: &Marker) -> Option<(f64, f64, f64)> {
    match &m.shape {
        MShape::Circle  { center, r }            => Some((center.x, center.y, *r)),
        MShape::Annulus { center, r_outer, .. }  => Some((center.x, center.y, *r_outer)),
        MShape::Ellipse { center, a, b, .. }     => Some((center.x, center.y, a.max(*b))),
        MShape::Box     { center, w, h, .. }     => Some((center.x, center.y, (w.max(*h)) * 0.5)),
        _ => None,
    }
}

// ---------------------------------------------------------------- HDU --

/// True for HDU kinds whose data we can render as an image.
fn is_loadable_hdu(kind: &str) -> bool {
    matches!(kind, "PRIMARY" | "IMAGE" | "TILE-COMPRESSED")
}

/// Enumerate the HDUs of the active frame's source file and push them into the
/// `hdu-rows` slint model. Returns the source path on success, or an error
/// message on failure (no source file, no HDUs, parse error).
fn populate_hdu_panel(window: &MainWindow, st: &State) -> Result<(), String> {
    let active = st.active_frame().ok_or("HDU navigator: no active frame")?;
    let path = active.source_path.clone()
        .ok_or("HDU navigator: active frame is synthetic (no source file)")?;
    let cur_idx = active.hdu_idx;
    let hdus = ds9_fits::enumerate_hdus(&path)
        .map_err(|e| format!("HDU navigator: {e}"))?;
    if hdus.is_empty() { return Err("HDU navigator: file has no HDUs".into()); }

    let rows: Vec<HduRow> = hdus.iter().map(|h| {
        let dims = match h.dims {
            Some((w, h_)) => format!(" {w}×{h_}"),
            None => String::new(),
        };
        let label = format!("[{:>2}] {:<16} {}{dims}", h.idx, h.kind, h.name);
        HduRow {
            idx: h.idx as i32,
            label: label.into(),
            loadable: is_loadable_hdu(h.kind),
            active: h.idx == cur_idx,
        }
    }).collect();

    window.set_hdu_rows(ModelRc::new(VecModel::from(rows)));
    let label = path.file_name().and_then(|s| s.to_str()).unwrap_or("(fits)");
    window.set_hdu_source_name(format!("({})", label).into());
    Ok(())
}

/// Replace the active frame's image with `path`'s `target_idx` HDU. Updates
/// the panel's row state and refreshes the view.
fn load_hdu_into_active(window: &MainWindow, st: &mut State, target_idx: usize) {
    let Some(path) = st.active_frame().and_then(|f| f.source_path.clone()) else {
        window.set_status_text("HDU load: no source path".into()); return;
    };
    let info = match ds9_fits::enumerate_hdus(&path) {
        Ok(v) => v,
        Err(e) => { window.set_status_text(format!("HDU load: {e}").into()); return; }
    };
    let Some(meta) = info.into_iter().find(|h| h.idx == target_idx) else {
        window.set_status_text(format!("HDU load: no HDU [{target_idx}]").into()); return;
    };
    if !is_loadable_hdu(meta.kind) {
        window.set_status_text(format!("HDU [{target_idx}] is {} (not loadable)", meta.kind).into());
        return;
    }
    match ds9_fits::load_hdu(&path, target_idx) {
        Ok(img) => {
            let (w, h) = (img.width, img.height);
            save_view_into_active(window, st);
            if let Some(f) = st.active_frame_mut() {
                let new_name = format!("{}[{}]",
                    path.file_name().and_then(|s| s.to_str()).unwrap_or("(fits)"),
                    meta.name);
                f.fits = img;
                f.name = new_name;
                f.hdu_idx = target_idx;
                f.view_zoom = fit_zoom(w, h);
                f.view_pan_x = 0.0;
                f.view_pan_y = 0.0;
            }
            push_view_to_window(window, st);
            refresh_view(window, st);
            // refresh the panel so the active row marker moves
            if window.get_hdu_panel_visible() {
                let _ = populate_hdu_panel(window, st);
            }
            window.set_status_text(format!(
                "HDU [{target_idx}] {} {} ({w}×{h})", meta.kind, meta.name,
            ).into());
        }
        Err(e) => window.set_status_text(format!("HDU load: {e}").into()),
    }
}

/// Find the next loadable HDU after the active frame's current HDU and load
/// it (wrapping). No-op if the file has no other loadable HDUs.
fn advance_hdu(window: &MainWindow, st: &mut State) {
    let Some(active) = st.active_frame() else {
        window.set_status_text("HDU: no active frame".into()); return;
    };
    let Some(path) = active.source_path.clone() else {
        window.set_status_text("HDU: active frame is synthetic".into()); return;
    };
    let cur_idx = active.hdu_idx;
    let hdus = match ds9_fits::enumerate_hdus(&path) {
        Ok(v) => v,
        Err(e) => { window.set_status_text(format!("HDU: {e}").into()); return; }
    };
    let total = hdus.len();
    if total == 0 { window.set_status_text("HDU: file has no HDUs".into()); return; }
    let mut chosen: Option<usize> = None;
    for k in 1..=total {
        let i = (cur_idx + k) % total;
        if is_loadable_hdu(hdus[i].kind) { chosen = Some(i); break; }
    }
    let Some(target) = chosen else {
        window.set_status_text("HDU: no other loadable HDUs".into()); return;
    };
    load_hdu_into_active(window, st, target);
}

// -------------------------------------------------------------- prefs --

/// Parse the smooth-mini-language used by `Prefs` and the Preferences panel:
/// `off | gaussian:SIGMA | boxcar:N | median:N`. Whitespace tolerated. On a
/// parse error the default (off) is returned.
fn parse_smooth_spec(s: &str) -> SmoothKind {
    let s = s.trim().to_ascii_lowercase();
    if s.is_empty() || s == "off" || s == "none" {
        return SmoothKind::Gaussian { sigma: 0.0 };
    }
    let (kind, val) = match s.split_once(':') {
        Some((k, v)) => (k.trim(), v.trim()),
        None => return SmoothKind::Gaussian { sigma: 0.0 },
    };
    match kind {
        "gaussian" | "gauss" | "g" => {
            let sigma: f32 = val.parse().unwrap_or(0.0);
            SmoothKind::Gaussian { sigma: sigma.max(0.0) }
        }
        "boxcar" | "box" | "b" => {
            let n: u32 = val.parse().unwrap_or(0);
            let n = n.max(1);
            // boxcar/median expect odd window
            let n = if n % 2 == 0 { n + 1 } else { n };
            SmoothKind::Boxcar { n }
        }
        "median" | "med" | "m" => {
            let n: u32 = val.parse().unwrap_or(0);
            let n = n.max(1);
            let n = if n % 2 == 0 { n + 1 } else { n };
            SmoothKind::Median { n }
        }
        _ => SmoothKind::Gaussian { sigma: 0.0 },
    }
}

fn parse_limits_spec(s: &str) -> Option<LimitsMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "zscale" => Some(LimitsMode::Zscale),
        "minmax" => Some(LimitsMode::MinMax),
        _        => None,
    }
}

fn parse_stretch_spec(s: &str) -> Option<Stretch> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "linear"  => Stretch::Linear,
        "log"     => Stretch::Log,
        "sqrt"    => Stretch::Sqrt,
        "squared" => Stretch::Squared,
        "asinh"   => Stretch::Asinh,
        "sinh"    => Stretch::Sinh,
        _         => return None,
    })
}

/// Apply the session prefs as the *initial* state of a freshly-created frame.
/// Called from `load_into` (and other Frame creation sites) right after
/// `Frame::new` so that user-saved defaults stick to every loaded image.
fn apply_prefs_to_frame(prefs: &Prefs, fr: &mut Frame) {
    if let Some(c) = Colormap::from_name(&prefs.cmap) { fr.cmap = c; }
    if let Some(s) = parse_stretch_spec(&prefs.stretch) { fr.stretch = s; }
    if let Some(lm) = parse_limits_spec(&prefs.limits) { fr.limits_mode = lm; }
    let sk = parse_smooth_spec(&prefs.smooth);
    fr.smooth_kind = sk;
    if let SmoothKind::Gaussian { sigma } = sk { fr.smooth_sigma = sigma; }
    fr.bin_factor = prefs.bin.max(1);
}

fn populate_prefs_panel(window: &MainWindow, st: &State) {
    let p = &st.prefs;
    window.set_prefs_cmap(p.cmap.clone().into());
    window.set_prefs_stretch(p.stretch.clone().into());
    window.set_prefs_limits(p.limits.clone().into());
    window.set_prefs_smooth(p.smooth.clone().into());
    window.set_prefs_bin(p.bin.to_string().into());
    window.set_prefs_blink(p.blink_ms.to_string().into());
    window.set_prefs_status("edit fields, then Apply (this session) or Save (persist).".into());
}

/// Read the panel fields → update `State.prefs` → push them onto the active
/// frame and refresh.
fn apply_prefs_from_panel(window: &MainWindow, st: &mut State) {
    let cmap    = window.get_prefs_cmap().as_str().trim().to_string();
    let stretch = window.get_prefs_stretch().as_str().trim().to_string();
    let limits  = window.get_prefs_limits().as_str().trim().to_string();
    let smooth  = window.get_prefs_smooth().as_str().trim().to_string();
    let bin     = window.get_prefs_bin().as_str().trim().parse::<u32>().unwrap_or(1).max(1);
    let blink   = window.get_prefs_blink().as_str().trim().parse::<u32>().unwrap_or(500).max(50);

    let mut warnings = Vec::<&'static str>::new();
    if Colormap::from_name(&cmap).is_none()      { warnings.push("cmap"); }
    if parse_stretch_spec(&stretch).is_none()    { warnings.push("stretch"); }
    if parse_limits_spec(&limits).is_none()      { warnings.push("limits"); }

    st.prefs = Prefs { cmap, stretch, limits, smooth, bin, blink_ms: blink };
    let p = st.prefs.clone();
    if let Some(f) = st.active_frame_mut() {
        apply_prefs_to_frame(&p, f);
    }
    refresh_view(window, st);

    let msg = if warnings.is_empty() {
        "prefs applied to active frame and session defaults.".to_string()
    } else {
        format!("prefs applied; ignored unknown: {}.", warnings.join(", "))
    };
    window.set_prefs_status(msg.into());
    window.set_status_text("prefs applied".into());
}

fn prefs_path() -> Option<PathBuf> {
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("ds9-rust").join("prefs.toml"))
}

fn save_prefs_to_disk(prefs: &Prefs) -> std::io::Result<PathBuf> {
    let p = prefs_path().ok_or_else(|| std::io::Error::new(
        std::io::ErrorKind::NotFound, "no $HOME / $XDG_CONFIG_HOME"))?;
    if let Some(parent) = p.parent() { std::fs::create_dir_all(parent)?; }
    let body = format!(
        "# ds9-rust preferences\n\
         cmap     = \"{}\"\n\
         stretch  = \"{}\"\n\
         limits   = \"{}\"\n\
         smooth   = \"{}\"\n\
         bin      = {}\n\
         blink_ms = {}\n",
        prefs.cmap, prefs.stretch, prefs.limits, prefs.smooth, prefs.bin, prefs.blink_ms,
    );
    std::fs::write(&p, body)?;
    Ok(p)
}

/// Best-effort load of `prefs.toml`. We hand-parse `key = value` lines with
/// optional double-quoted strings; missing/invalid keys fall back to the
/// `Default::default()` value. This avoids pulling in a real TOML crate.
fn load_prefs_from_disk() -> Prefs {
    let mut p = Prefs::default();
    let Some(path) = prefs_path() else { return p };
    let Ok(text) = std::fs::read_to_string(&path) else { return p };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let Some((k, v)) = line.split_once('=') else { continue };
        let k = k.trim();
        let v = v.trim().trim_matches('"').to_string();
        match k {
            "cmap"     => p.cmap = v,
            "stretch"  => p.stretch = v,
            "limits"   => p.limits = v,
            "smooth"   => p.smooth = v,
            "bin"      => if let Ok(n) = v.parse::<u32>() { p.bin = n.max(1); },
            "blink_ms" => if let Ok(n) = v.parse::<u32>() { p.blink_ms = n.max(50); },
            _ => {}
        }
    }
    p
}

// ---------------------------------------------------------------- menus --

fn handle_menu(window: &MainWindow, st: &mut State, menu: &str, item: &str) {
    match (menu, item) {
        // File
        ("File", "Open…") => window.invoke_request_open_file(),
        ("File", "Save Image…") => save_image_png(window, st),
        ("File", "Save FITS…")  => save_image_fits(window, st),
        ("File", "Save TIFF…") => {
            let Some(f) = st.active_frame() else {
                window.set_status_text("save tiff: no active frame".into()); return;
            };
            let chosen: Option<PathBuf> = rfd::FileDialog::new()
                .set_title("Save image as TIFF")
                .set_file_name(format!("{}.tif",
                    f.name.trim_end_matches(".fits").trim_end_matches(".fit")))
                .add_filter("TIFF", &["tif", "tiff"])
                .save_file();
            let Some(p) = chosen else { return };
            let rgba = render_rgba_for_frame(f);
            match write_tiff_rgb(&p, &rgba, f.fits.width as u32, f.fits.height as u32) {
                Ok(()) => window.set_status_text(format!("wrote {}", p.display()).into()),
                Err(e) => window.set_status_text(format!("save tiff: {e}").into()),
            }
        }
        ("File", "Save JPEG…") => {
            window.set_status_text(
                "JPEG export not built in — use 'Save Image…' (PNG) or 'Save TIFF…' for now".into()
            );
        }
        ("File", "Save PostScript…") => {
            let Some(f) = st.active_frame() else {
                window.set_status_text("save ps: no active frame".into()); return;
            };
            let chosen: Option<PathBuf> = rfd::FileDialog::new()
                .set_title("Save image as PostScript")
                .set_file_name(format!("{}.eps",
                    f.name.trim_end_matches(".fits").trim_end_matches(".fit")))
                .add_filter("PostScript", &["ps", "eps"])
                .save_file();
            let Some(p) = chosen else { return };
            let rgba = render_rgba_for_frame(f);
            match write_postscript_rgb(&p, &rgba, f.fits.width as u32, f.fits.height as u32) {
                Ok(()) => window.set_status_text(format!("wrote {}", p.display()).into()),
                Err(e) => window.set_status_text(format!("save ps: {e}").into()),
            }
        }
        ("File", "HDU Navigator…") => {
            // Toggle the panel; populate when opening.
            if window.get_hdu_panel_visible() {
                window.set_hdu_panel_visible(false);
                window.set_status_text("HDU navigator: hidden".into());
            } else {
                match populate_hdu_panel(window, st) {
                    Ok(()) => {
                        window.set_hdu_panel_visible(true);
                        window.set_status_text("HDU navigator: shown".into());
                    }
                    Err(msg) => window.set_status_text(msg.into()),
                }
            }
        }
        ("File", "Next HDU") => { advance_hdu(window, st); }
        ("File", "Print…") => {
            // Spawn a print pipeline by saving a PNG to a temp path and
            // shelling out to `lpr`. If lpr isn't on PATH, fall back to
            // notifying the user where the PNG ended up.
            let Some(f) = st.active_frame() else {
                window.set_status_text("print: no active frame".into()); return;
            };
            let tmp = std::env::temp_dir().join(format!("ds9-rust-print-{}.png", std::process::id()));
            let rgba = render_rgba_for_frame(f);
            let (w_, h_) = (f.fits.width as u32, f.fits.height as u32);
            let res = (|| -> std::io::Result<()> {
                let file = std::fs::File::create(&tmp)?;
                let bw = std::io::BufWriter::new(file);
                let mut enc = png::Encoder::new(bw, w_, h_);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let mut wh = enc.write_header().map_err(|e| std::io::Error::other(e.to_string()))?;
                wh.write_image_data(&rgba).map_err(|e| std::io::Error::other(e.to_string()))
            })();
            match res {
                Ok(()) => {
                    let printed = std::process::Command::new("lpr").arg(&tmp).status();
                    let msg = match printed {
                        Ok(s) if s.success() => format!("sent to lpr  ({})", tmp.display()),
                        _ => format!("saved {} (lpr unavailable)", tmp.display()),
                    };
                    window.set_status_text(msg.into());
                }
                Err(e) => window.set_status_text(format!("print: {e}").into()),
            }
        }
        ("File", "SAMP Send Image")    => samp_send_active_image(window, st),
        ("File", "SAMP Send VOTable…") => samp_send_votable(window),
        ("File", "Quit")  => { let _ = slint::quit_event_loop(); }

        // View — sidebar panel toggles and overlays
        ("View", "Panner") => {
            let on = !window.get_panner_visible();
            window.set_panner_visible(on);
            window.set_status_text(if on { "panner: shown".into() } else { "panner: hidden".into() });
        }
        ("View", "Magnifier") => {
            let on = !window.get_magnifier_visible();
            window.set_magnifier_visible(on);
            window.set_status_text(if on { "magnifier: shown".into() } else { "magnifier: hidden".into() });
        }
        ("View", "Coordinate Grid") => {
            let Some(f) = st.active_frame() else {
                window.set_status_text("grid: no active frame".into()); return;
            };
            if f.fits.wcs.is_none() {
                window.set_status_text("grid: active frame has no WCS".into()); return;
            }
            let on = !window.get_grid_visible();
            if on {
                match render_grid_overlay(f) {
                    Some(img) => {
                        window.set_grid_image(img);
                        window.set_grid_visible(true);
                        window.set_status_text("grid: shown".into());
                    }
                    None => window.set_status_text("grid: WCS projection failed".into()),
                }
            } else {
                window.set_grid_visible(false);
                window.set_status_text("grid: hidden".into());
            }
        }
        ("View", "Crosshair") => {
            // toggle: if a crosshair exists, clear it; otherwise drop one at
            // the cursor's current position.
            if st.crosshair.is_some() {
                st.crosshair = None;
                push_crosshair_to_window(window, st);
                window.set_status_text("crosshair: cleared".into());
            } else {
                let cx = window.get_cursor_image_x() as f64;
                let cy = window.get_cursor_image_y() as f64;
                set_crosshair_at_display(window, st, cx, cy);
            }
        }

        // Frame
        ("Frame", "New Frame") => window.invoke_request_open_file(),
        ("Frame", "Delete Frame") => {
            if st.frames.is_empty() {
                window.set_status_text("no frames to delete".into());
            } else {
                let removed = st.active;
                st.frames.remove(removed);
                if st.frames.is_empty() {
                    st.active = 0;
                } else if st.active >= st.frames.len() {
                    st.active = st.frames.len() - 1;
                }
                push_view_to_window(window, st);
                refresh_view(window, st);
                window.set_status_text(format!(
                    "deleted frame {} ({} remaining)", removed + 1, st.frames.len()
                ).into());
            }
        }
        ("Frame", "Next") => {
            if !st.frames.is_empty() {
                let n = st.frames.len();
                switch_frame(window, st, (st.active + 1) % n);
            }
        }
        ("Frame", "Previous") => {
            if !st.frames.is_empty() {
                let n = st.frames.len();
                switch_frame(window, st, (st.active + n - 1) % n);
            }
        }
        ("Frame", "Match…") => {
            let Some(active) = st.active_frame() else {
                window.set_status_text("match: no active frame".into()); return;
            };
            let zoom = active.view_zoom;
            let pan_x = active.view_pan_x;
            let pan_y = active.view_pan_y;
            let active_idx = st.active;
            let n = st.frames.len();
            for (i, fr) in st.frames.iter_mut().enumerate() {
                if i == active_idx { continue; }
                fr.view_zoom = zoom;
                fr.view_pan_x = pan_x;
                fr.view_pan_y = pan_y;
            }
            window.set_status_text(format!("matched view of {} frames to active", n).into());
        }
        ("Frame", "Blink") => {
            let next = !window.get_blink_active();
            window.set_blink_active(next);
            window.set_status_text(
                if next { "blink: on (cycling every 500 ms)".into() }
                else    { "blink: off".into() }
            );
        }
        ("Frame", "HDU Movie") => {
            let next = !window.get_hdu_movie_active();
            window.set_hdu_movie_active(next);
            window.set_status_text(
                if next { "HDU movie: on (cycling every 800 ms)".into() }
                else    { "HDU movie: off".into() }
            );
        }
        ("Frame", "Rotate 180°") | ("Frame", "Flip Horizontal") | ("Frame", "Flip Vertical")
        | ("Frame", "Reset Orientation") => {
            let new_o = match item {
                "Rotate 180°"       => Orientation::Rot180,
                "Flip Horizontal"   => Orientation::FlipH,
                "Flip Vertical"     => Orientation::FlipV,
                _                   => Orientation::Identity,
            };
            if let Some(f) = st.active_frame_mut() {
                f.orientation = new_o;
            }
            refresh_view(window, st);
            window.set_status_text(format!("orientation: {}", new_o.name()).into());
        }
        ("Frame", "Lock Zoom") | ("Frame", "Lock Pan")
        | ("Frame", "Lock Color") | ("Frame", "Lock Scale") => {
            let (slot, label) = match item {
                "Lock Zoom"  => (&mut st.lock_zoom, "lock zoom"),
                "Lock Pan"   => (&mut st.lock_pan,  "lock pan"),
                "Lock Color" => (&mut st.lock_cmap, "lock color"),
                "Lock Scale" => (&mut st.lock_scale, "lock scale"),
                _ => return,
            };
            *slot = !*slot;
            let on = *slot;
            window.set_status_text(format!("{label}: {}", if on { "on" } else { "off" }).into());
            // sync the slint mirror
            if item == "Lock Zoom"  { window.set_lock_zoom(on);  }
            if item == "Lock Pan"   { window.set_lock_pan(on);   }
            if item == "Lock Color" { window.set_lock_cmap(on);  }
            if item == "Lock Scale" { window.set_lock_scale(on); }
            // when turning on, apply the active frame's state to every other frame
            if on {
                broadcast_locks(st);
            }
        }
        ("Frame", "RGB Composite") => {
            if st.frames.len() < 3 {
                window.set_status_text("RGB: need at least 3 frames loaded".into());
            } else {
                match build_rgb_composite(&st.frames[0..3]) {
                    Some((img, w, h)) => {
                        window.set_fits_image(img);
                        window.set_fits_width(w as i32);
                        window.set_fits_height(h as i32);
                        window.set_status_text(format!(
                            "RGB composite from frames 1-3 ({w}×{h})"
                        ).into());
                    }
                    None => window.set_status_text(
                        "RGB: frames 1-3 must share the same dimensions".into()
                    ),
                }
            }
        }
        ("Frame", "Tile Frames") => {
            if st.frames.len() < 2 && !st.tile_mode {
                window.set_status_text("tile: load 2+ frames first".into());
            } else {
                st.tile_mode = !st.tile_mode;
                if !st.tile_mode {
                    // restore the active frame's saved view
                    push_view_to_window(window, st);
                }
                refresh_view(window, st);
                window.set_status_text(
                    if st.tile_mode {
                        format!("tile: showing {} frames (click a tile to select)", st.frames.len()).into()
                    } else {
                        "tile: off".into()
                    }
                );
            }
        }
        ("Frame", "Mosaic WCS") => {
            match build_mosaic(&st.frames) {
                Ok(img) => {
                    let (w, h, n_in) = (img.width, img.height,
                        st.frames.iter().filter(|f| f.fits.wcs.is_some()).count());
                    save_view_into_active(window, st);
                    let mut fr = Frame::new(img, format!("mosaic ({n_in} frames)"));
                    apply_prefs_to_frame(&st.prefs, &mut fr);
                    st.frames.push(fr);
                    st.active = st.frames.len() - 1;
                    push_view_to_window(window, st);
                    refresh_view(window, st);
                    window.set_status_text(format!(
                        "mosaic: {n_in} frames → {w}×{h}    [frame {}]", st.active + 1
                    ).into());
                }
                Err(e) => window.set_status_text(e.into()),
            }
        }

        // Scale — stretch / limits live on the active frame
        ("Scale", "linear")  => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Linear;  } refresh_view(window, st); }
        ("Scale", "log")     => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Log;     } refresh_view(window, st); }
        ("Scale", "sqrt")    => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Sqrt;    } refresh_view(window, st); }
        ("Scale", "squared") => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Squared; } refresh_view(window, st); }
        ("Scale", "asinh")   => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Asinh;   } refresh_view(window, st); }
        ("Scale", "sinh")    => { if let Some(f) = st.active_frame_mut() { f.stretch = Stretch::Sinh;    } refresh_view(window, st); }
        ("Scale", "minmax")  => { if let Some(f) = st.active_frame_mut() { f.limits_mode = LimitsMode::MinMax; } refresh_view(window, st); }
        ("Scale", "zscale")  => { if let Some(f) = st.active_frame_mut() { f.limits_mode = LimitsMode::Zscale; } refresh_view(window, st); }

        // Bin
        ("Bin", "Average") | ("Bin", "Sum") | ("Bin", "Sub-sample") => {
            let mode = match item {
                "Average"   => BinMode::Average,
                "Sum"       => BinMode::Sum,
                _           => BinMode::Subsample,
            };
            if let Some(f) = st.active_frame_mut() { f.bin_mode = mode; }
            refresh_view(window, st);
            window.set_status_text(format!("bin mode: {}", mode.label()).into());
        }
        ("Bin", n) => {
            if let Ok(factor) = n.parse::<u32>() {
                if let Some(f) = st.active_frame_mut() { f.bin_factor = factor.max(1); }
                refresh_view(window, st);
                window.set_status_text(format!("bin: {n}×{n}").into());
            }
        }

        // Color
        ("Color", "Load Custom…") => {
            let chosen: Option<PathBuf> = rfd::FileDialog::new()
                .set_title("Load custom colormap (256-stop RGB text)")
                .add_filter("Colormap", &["lut", "cmap", "txt"])
                .add_filter("All", &["*"])
                .pick_file();
            let Some(p) = chosen else { return };
            match ds9_image::load_user_lut(&p) {
                Ok(lut) => {
                    if let Some(f) = st.active_frame_mut() {
                        f.custom_lut = Some(Box::new(lut));
                    }
                    refresh_view(window, st);
                    window.set_status_text(format!("custom colormap loaded from {}", p.display()).into());
                }
                Err(e) => window.set_status_text(format!("custom cmap: {e}").into()),
            }
        }
        ("Color", "Clear Custom") => {
            if let Some(f) = st.active_frame_mut() { f.custom_lut = None; }
            refresh_view(window, st);
            window.set_status_text("custom colormap cleared".into());
        }
        ("Color", name) => {
            if let Some(c) = Colormap::from_name(name) {
                if let Some(f) = st.active_frame_mut() { f.cmap = c; f.custom_lut = None; }
                if st.lock_cmap { broadcast_cmap(st); }
                refresh_view(window, st);
            }
        }

        // Region
        ("Region", "New")    => { region_new_sample(st); refresh_view(window, st); }
        ("Region", "Load…")  => { region_load(window, st); }
        ("Region", "Save…")  => { region_save(window, st); }
        ("Region", "Delete Selected") => {
            let removed = st.active_frame_mut().and_then(|f| {
                f.selected_marker.take().and_then(|i| {
                    if i < f.markers.len() { f.markers.remove(i); Some((i, f.markers.len())) }
                    else { None }
                })
            });
            if let Some((i, n)) = removed {
                window.set_status_text(format!("deleted region {} (now {n} regions)", i + 1).into());
                refresh_view(window, st);
            } else {
                window.set_status_text("no region selected".into());
            }
        }
        ("Region", "Delete All") => {
            if let Some(f) = st.active_frame_mut() {
                let n = f.markers.len();
                f.markers.clear();
                f.selected_marker = None;
                window.set_status_text(format!("cleared {n} regions").into());
                refresh_view(window, st);
            }
        }
        ("Region", "Projection…") => {
            // Pick a Line marker — selected one if it's a Line, otherwise the
            // first Line in the marker list.
            let line: Option<((f64, f64), (f64, f64))> = st.active_frame().and_then(|f| {
                let pick = f.selected_marker
                    .and_then(|i| f.markers.get(i))
                    .filter(|m| matches!(m.shape, MShape::Line { .. }))
                    .or_else(|| f.markers.iter().find(|m| matches!(m.shape, MShape::Line { .. })));
                pick.and_then(|m| match &m.shape {
                    MShape::Line { from, to } => Some(((from.x, from.y), (to.x, to.y))),
                    _ => None,
                })
            });
            let Some((p0, p1)) = line else {
                window.set_status_text(
                    "projection: load or draw a Line region first (use Region ▸ Load…)".into()
                );
                return;
            };
            let Some(f) = st.active_frame() else { return };
            let (samples, len) = sample_line(f, p0.0, p0.1, p1.0, p1.1);
            window.set_projection_image(render_line_plot(&samples, 760, 420));
            let finite = samples.iter().filter(|v| v.is_finite()).count();
            window.set_projection_caption(format!(
                "PROJECTION  len={:.1} px  n={}/{}", len, finite, samples.len()
            ).into());
            let on = !window.get_projection_visible();
            window.set_projection_visible(on);
            window.set_status_text(
                if on { format!("projection: {} samples along {:.1} px", samples.len(), len).into() }
                else  { "projection: hidden".into() }
            );
        }
        ("Region", "Properties…") => {
            if window.get_region_prop_visible() {
                window.set_region_prop_visible(false);
                window.set_status_text("region properties: hidden".into());
            } else {
                populate_region_props(window, st);
                window.set_region_prop_visible(true);
                window.set_status_text("region properties: shown".into());
            }
        }
        ("Region", "Centroid") => {
            // Find a marker with a circular extent (selected first, else first
            // such marker), compute centroid, snap its center to it.
            let aperture: Option<(usize, f64, f64, f64)> = st.active_frame().and_then(|f| {
                let pick_idx = f.selected_marker
                    .filter(|&i| f.markers.get(i).and_then(marker_aperture).is_some())
                    .or_else(|| f.markers.iter().position(|m| marker_aperture(m).is_some()));
                pick_idx.and_then(|i| {
                    marker_aperture(&f.markers[i]).map(|(cx, cy, r)| (i, cx, cy, r))
                })
            });
            let Some((idx, cx, cy, r)) = aperture else {
                window.set_status_text(
                    "centroid: select a Circle / Box / Ellipse / Annulus region first".into()
                );
                return;
            };
            let Some(f) = st.active_frame() else { return };
            let Some((nx, ny)) = compute_centroid(f, cx, cy, r) else {
                window.set_status_text("centroid: not enough flux above local background".into());
                return;
            };
            let (dx, dy) = (nx - cx, ny - cy);
            if let Some(f) = st.active_frame_mut() {
                if let Some(m) = f.markers.get_mut(idx) {
                    translate_marker(m, dx, dy);
                    f.selected_marker = Some(idx);
                }
            }
            refresh_view(window, st);
            window.set_status_text(format!(
                "centroid: ({:.2}, {:.2}) → ({:.2}, {:.2}) Δ=({:+.2}, {:+.2})",
                cx, cy, nx, ny, dx, dy
            ).into());
        }
        ("Region", "Radial Profile…") => {
            let aperture: Option<(f64, f64, f64)> = st.active_frame().and_then(|f| {
                f.selected_marker
                    .and_then(|i| f.markers.get(i))
                    .and_then(marker_aperture)
                    .or_else(|| f.markers.iter().find_map(marker_aperture))
            });
            let Some((cx, cy, r)) = aperture else {
                window.set_status_text(
                    "radial profile: select a Circle / Ellipse / Annulus region first".into()
                );
                return;
            };
            let Some(f) = st.active_frame() else { return };
            let prof = compute_radial_profile(f, cx, cy, r, 40);
            window.set_radial_image(render_line_plot(&prof, 760, 420));
            window.set_radial_caption(format!(
                "RADIAL PROFILE  ({:.1}, {:.1})  r≤{:.1}  {} bins",
                cx, cy, r, prof.len()
            ).into());
            let on = !window.get_radial_visible();
            window.set_radial_visible(on);
            window.set_status_text(
                if on { format!("radial profile: 40 bins, r≤{:.1} px", r).into() }
                else  { "radial profile: hidden".into() }
            );
        }
        ("Region", "Info")   => {
            let msg = match st.active_frame() {
                Some(f) => {
                    let sel = match f.selected_marker {
                        Some(i) => format!("  (selected #{})", i + 1),
                        None => String::new(),
                    };
                    format!("regions: {}{sel}", f.markers.len())
                }
                None => "no active frame".to_string(),
            };
            window.set_status_text(msg.into());
        }

        // Catalog
        ("Catalog", "Load…") => { catalog_load(window, st); }
        ("Catalog", "Clear") => { catalog_clear(window, st); }
        ("Catalog", "Run SExtractor…") => { run_sextractor(window, st); }
        ("Catalog", "Online Query…") => {
            window.set_netcat_visible(!window.get_netcat_visible());
            window.set_status_text(
                if window.get_netcat_visible() { "online catalog query: shown".into() }
                else                            { "online catalog query: hidden".into() }
            );
        }
        ("Catalog", "Info")  => {
            let msg = match st.active_frame().and_then(|f| f.catalog.as_ref()) {
                Some(c) => format!(
                    "catalog: {} rows, {} cols  →  X/Y = {:?}",
                    c.len(), c.columns.len(),
                    c.xy_columns().map(|(xi, yi)| (
                        c.columns[xi].as_str(), c.columns[yi].as_str()
                    ))
                ),
                None => "no catalog loaded".to_string(),
            };
            window.set_status_text(msg.into());
        }

        // Analysis — smoothing
        ("Analysis", "Smooth (cycle)") => {
            // Cycle the active kernel's strength: Gaussian σ ∈ {2,4,8,off};
            // Boxcar / Median window n ∈ {3,5,9,off}.
            if let Some(f) = st.active_frame_mut() {
                f.smooth_kind = match f.smooth_kind {
                    SmoothKind::Gaussian { sigma } => {
                        let next = match sigma {
                            s if s <= 0.0 => 2.0,
                            s if s < 3.0  => 4.0,
                            s if s < 6.0  => 8.0,
                            _             => 0.0,
                        };
                        SmoothKind::Gaussian { sigma: next }
                    }
                    SmoothKind::Boxcar { n } => {
                        let next = match n { 0 | 1 => 3, 2..=3 => 5, 4..=5 => 9, _ => 1 };
                        SmoothKind::Boxcar { n: next }
                    }
                    SmoothKind::Median { n } => {
                        let next = match n { 0 | 1 => 3, 2..=3 => 5, 4..=5 => 9, _ => 1 };
                        SmoothKind::Median { n: next }
                    }
                };
                // mirror to the legacy sigma field so the IPC `smooth ?` query stays accurate
                if let SmoothKind::Gaussian { sigma } = f.smooth_kind { f.smooth_sigma = sigma; }
                window.set_status_text(format!("smooth: {}", f.smooth_kind.label()).into());
                refresh_view(window, st);
            }
        }
        ("Analysis", "Smooth Off") => {
            if let Some(f) = st.active_frame_mut() {
                f.smooth_sigma = 0.0;
                f.smooth_kind = SmoothKind::Gaussian { sigma: 0.0 };
            }
            window.set_status_text("smooth: off".into());
            refresh_view(window, st);
        }
        ("Analysis", "Smooth Gaussian") => {
            if let Some(f) = st.active_frame_mut() {
                f.smooth_kind = SmoothKind::Gaussian { sigma: 2.0 };
                f.smooth_sigma = 2.0;
            }
            window.set_status_text("smooth: gaussian σ=2".into());
            refresh_view(window, st);
        }
        ("Analysis", "Smooth Boxcar") => {
            if let Some(f) = st.active_frame_mut() {
                f.smooth_kind = SmoothKind::Boxcar { n: 3 };
            }
            window.set_status_text("smooth: boxcar 3×3".into());
            refresh_view(window, st);
        }
        ("Analysis", "Smooth Median") => {
            if let Some(f) = st.active_frame_mut() {
                f.smooth_kind = SmoothKind::Median { n: 3 };
            }
            window.set_status_text("smooth: median 3×3".into());
            refresh_view(window, st);
        }

        // Analysis
        ("Analysis", "Histogram…") => {
            if let Some(f) = st.active_frame() {
                window.set_histogram_image(render_histogram_image(f, 760, 420));
                let on = !window.get_histogram_visible();
                window.set_histogram_visible(on);
                window.set_status_text(
                    if on { "histogram: shown".into() } else { "histogram: hidden".into() }
                );
            } else {
                window.set_status_text("histogram: no image loaded".into());
            }
        }
        ("Analysis", "Contour Levels…") => {
            if let Some(f) = st.active_frame() {
                window.set_contour_image(render_contour_overlay(f));
                let on = !window.get_contour_visible();
                window.set_contour_visible(on);
                window.set_status_text(
                    if on { "contours: shown (5 levels)".into() } else { "contours: hidden".into() }
                );
            } else {
                window.set_status_text("contour: no image loaded".into());
            }
        }
        ("Analysis", "Statistics…") => {
            if let Some(f) = st.active_frame() {
                let img = &f.fits;
                if let Some((min, max, mean, median, std)) = image_stats(img) {
                    window.set_status_text(format!(
                        "stats: n={}  min={:.4}  max={:.4}  mean={:.4}  median={:.4}  σ={:.4}",
                        img.data.len(), min, max, mean, median, std,
                    ).into());
                } else {
                    window.set_status_text("stats: no finite samples".into());
                }
            } else {
                window.set_status_text("stats: no image loaded".into());
            }
        }
        ("Analysis", "Pixel Table…") => {
            let cx = window.get_cursor_image_x() as i32;
            let cy = window.get_cursor_image_y() as i32;
            if let Some(f) = st.active_frame() {
                let img = &f.fits;
                let mut buf = String::new();
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let ix = cx + dx;
                        let iy = cy + dy;
                        if ix >= 0 && iy >= 0 && (ix as usize) < img.width && (iy as usize) < img.height {
                            let v = img.data[iy as usize * img.width + ix as usize];
                            buf.push_str(&format!("{v:>9.3} "));
                        } else {
                            buf.push_str("    ——   ");
                        }
                    }
                    buf.push_str(" │ ");
                }
                window.set_status_text(format!("5×5 around ({cx},{cy}):  {buf}").into());
            } else {
                window.set_status_text("pixel table: no image loaded".into());
            }
        }

        // Zoom
        ("Zoom", "Zoom In") => {
            window.set_view_zoom((window.get_view_zoom() * 1.5).clamp(0.02, 64.0));
        }
        ("Zoom", "Zoom Out") => {
            window.set_view_zoom((window.get_view_zoom() / 1.5).clamp(0.02, 64.0));
        }
        ("Zoom", "Fit") => {
            if let Some(f) = st.active_frame() {
                window.set_view_zoom(fit_zoom(f.fits.width, f.fits.height));
                window.set_view_pan_x(0.0);
                window.set_view_pan_y(0.0);
            }
        }
        ("Zoom", "Reset") => {
            window.set_view_zoom(1.0);
            window.set_view_pan_x(0.0);
            window.set_view_pan_y(0.0);
        }
        ("Zoom", n) if n.ends_with('×') => {
            if let Ok(z) = n.trim_end_matches('×').parse::<f32>() {
                window.set_view_zoom(z.clamp(0.02, 64.0));
            }
        }

        // Edit
        ("Edit", "Crop to Selected") => {
            let Some(f) = st.active_frame() else {
                window.set_status_text("crop: no active frame".into()); return;
            };
            // Prefer the selected marker; otherwise use the bounding box of all
            // markers that have one. If still nothing, bail.
            let bbox: Option<(f64, f64, f64, f64)> = f.selected_marker
                .and_then(|i| f.markers.get(i))
                .map(|m| marker_display_bbox(m, f))
                .or_else(|| {
                    if f.markers.is_empty() { return None; }
                    let mut acc: Option<(f64, f64, f64, f64)> = None;
                    for m in &f.markers {
                        let b = marker_display_bbox(m, f);
                        acc = Some(match acc {
                            None => b,
                            Some(a) => (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
                        });
                    }
                    acc
                });
            let Some((mnx, mny, mxx, mxy)) = bbox else {
                window.set_status_text(
                    "crop: select a region first (or load some — Region ▸ Load…)".into()
                );
                return;
            };
            let bw = (mxx - mnx).max(1.0);
            let bh = (mxy - mny).max(1.0);
            // canvas-w / canvas-h are slint `length` props (logical px) → f32.
            let cw = window.get_canvas_w() as f64;
            let ch = window.get_canvas_h() as f64;
            let zoom = ((cw / bw).min(ch / bh) as f32).clamp(0.02, 64.0);
            let cx = 0.5 * (mnx + mxx);
            let cy = 0.5 * (mny + mxy);
            window.set_view_zoom(zoom);
            window.set_view_pan_x((cw * 0.5 - cx * zoom as f64) as f32);
            window.set_view_pan_y((ch * 0.5 - cy * zoom as f64) as f32);
            window.set_status_text(format!(
                "crop: {:.0}×{:.0} px → zoom {:.2}×", bw, bh, zoom
            ).into());
        }
        ("Edit", "Reset Crop") => { handle_menu(window, st, "Zoom", "Reset"); }
        ("Edit", "Preferences…") => {
            if window.get_prefs_visible() {
                window.set_prefs_visible(false);
                window.set_status_text("preferences: hidden".into());
            } else {
                populate_prefs_panel(window, st);
                window.set_prefs_visible(true);
                window.set_status_text("preferences: shown".into());
            }
        }

        ("Help", "About ds9-rust") => {
            window.set_status_text(
                "ds9-rust 0.1 — slint port of SAOImage DS9 (Smithsonian Astrophysical Observatory)"
                    .into(),
            );
        }

        _ => {
            window.set_status_text(format!("{menu} ▸ {item} — not implemented yet").into());
        }
    }
}

// ---------------------------------------------------------------- IPC --

/// Dispatch a single line of the IPC text protocol. Runs on the UI thread.
/// Returns a response string (status / error / value) for the client.
fn dispatch_ipc(window: &MainWindow, st: &mut State, line: &str) -> String {
    let line = line.trim();
    if line.is_empty() { return String::new(); }
    // We re-tokenise into 4 for verbs like `pan to X Y` and `scale limits A B`.
    let toks: Vec<&str> = line.splitn(4, ' ').collect();
    match toks.as_slice() {
        ["quit"] | ["exit"] => { let _ = slint::quit_event_loop(); "ok".into() }
        ["xpaaccess"]   => "yes".into(),
        ["version"]     => "ds9-rust 0.1".into(),
        ["mode", m]     => {
            window.set_active_mode(slint::SharedString::from(*m));
            "ok".into()
        }
        ["frame", "mosaic"]    => { handle_menu(window, st, "Frame", "Mosaic WCS"); "ok".into() }
        ["frame", "tile"]      => { handle_menu(window, st, "Frame", "Tile Frames"); "ok".into() }
        ["frame", "new"]       => { handle_menu(window, st, "Frame", "New Frame"); "ok".into() }
        ["frame", "delete"]    => { handle_menu(window, st, "Frame", "Delete Frame"); "ok".into() }
        ["frame", "clear"]     => {
            // XPA-style "clear active frame": drop markers and catalog,
            // leave image. (DS9's `clear` actually wipes the image; we
            // preserve it because the image is what the user just opened.)
            if let Some(f) = st.active_frame_mut() {
                f.markers.clear();
                f.selected_marker = None;
                f.catalog = None;
                f.selected_catalog = None;
            }
            refresh_view(window, st);
            "ok".into()
        }
        ["samp", "image"]      => { handle_menu(window, st, "File", "SAMP Send Image"); "ok".into() }
        ["samp", "votable"]    => { handle_menu(window, st, "File", "SAMP Send VOTable…"); "ok".into() }
        ["frame", "next"]      => { handle_menu(window, st, "Frame", "Next"); "ok".into() }
        ["frame", "previous"]  => { handle_menu(window, st, "Frame", "Previous"); "ok".into() }
        ["frame", n] => {
            if let Ok(idx) = n.parse::<usize>() {
                if idx >= 1 { switch_frame(window, st, idx - 1); }
            }
            "ok".into()
        }
        ["pan", "to", x, y] => {
            // pan so that (x_fits, y_fits) appears under the canvas centre.
            let (fx, fy) = match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(a), Ok(b)) => (a, b),
                _ => return "err pan to X Y".into(),
            };
            let Some(f) = st.active_frame() else { return "err no frame".into() };
            let (dx, dy) = fits_to_display_oriented(fx, fy, f);
            let cw = window.get_canvas_w() as f64;
            let ch = window.get_canvas_h() as f64;
            let z  = window.get_view_zoom() as f64;
            window.set_view_pan_x((cw * 0.5 - dx as f64 * z) as f32);
            window.set_view_pan_y((ch * 0.5 - dy as f64 * z) as f32);
            "ok".into()
        }
        ["crosshair"] => {
            // toggle off
            st.crosshair = None;
            push_crosshair_to_window(window, st);
            "ok".into()
        }
        ["crosshair", "clear"] => {
            st.crosshair = None;
            push_crosshair_to_window(window, st);
            "ok".into()
        }
        ["crosshair", x, y] => {
            let (fx, fy) = match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(a), Ok(b)) => (a, b),
                _ => return "err crosshair X Y".into(),
            };
            let Some(f) = st.active_frame() else { return "err no frame".into() };
            let (dx, dy) = fits_to_display_oriented(fx, fy, f);
            set_crosshair_at_display(window, st, dx as f64, dy as f64);
            "ok".into()
        }
        ["scale", "limits", a, b] => {
            match (a.parse::<f32>(), b.parse::<f32>()) {
                (Ok(low), Ok(high)) if high > low => {
                    if let Some(f) = st.active_frame_mut() {
                        f.limits_mode = LimitsMode::User { low, high };
                    }
                    refresh_view(window, st);
                    "ok".into()
                }
                _ => "err scale limits LO HI (HI > LO)".into(),
            }
        }
        ["scale", s]            => { handle_menu(window, st, "Scale", s); "ok".into() }
        ["cmap", "invert"]      => {
            if let Some(f) = st.active_frame_mut() {
                if let Some(lut) = &f.custom_lut {
                    let mut new_lut: Box<[[u8; 3]; 256]> = lut.clone();
                    new_lut.reverse();
                    f.custom_lut = Some(new_lut);
                } else {
                    let mut lut: Box<[[u8; 3]; 256]> = Box::new([[0u8; 3]; 256]);
                    let strip = f.cmap.rgba_strip();
                    for i in 0..256 {
                        let off = (255 - i) * 4;
                        lut[i] = [strip[off], strip[off + 1], strip[off + 2]];
                    }
                    f.custom_lut = Some(lut);
                }
            }
            refresh_view(window, st);
            "ok".into()
        }
        ["cmap", c]             => { handle_menu(window, st, "Color", c); "ok".into() }
        ["regions", "delete", "all"] => {
            handle_menu(window, st, "Region", "Delete All"); "ok".into()
        }
        ["regions", "delete", n] => {
            match n.parse::<usize>() {
                Ok(idx) if idx >= 1 => {
                    if let Some(f) = st.active_frame_mut() {
                        let i = idx - 1;
                        if i < f.markers.len() {
                            f.markers.remove(i);
                            if f.selected_marker == Some(i) { f.selected_marker = None; }
                            refresh_view(window, st);
                            return "ok".into();
                        }
                    }
                    "err: index out of range".into()
                }
                _ => "err: regions delete all|N (1-based)".into(),
            }
        }
        ["regions", "load", path] => { region_load_path(window, st, Path::new(path)); "ok".into() }
        ["regions", "save", path] => {
            if let Some(f) = st.active_frame() {
                match ds9_marker::write_reg(path, &f.markers) {
                    Ok(()) => format!("ok {} regions", f.markers.len()),
                    Err(e) => format!("err {e}"),
                }
            } else { "err no active frame".into() }
        }
        ["bin", n]              => { handle_menu(window, st, "Bin", n); "ok".into() }
        ["zoom", "in"]          => { handle_menu(window, st, "Zoom", "Zoom In"); "ok".into() }
        ["zoom", "out"]         => { handle_menu(window, st, "Zoom", "Zoom Out"); "ok".into() }
        ["zoom", "fit"]         => { handle_menu(window, st, "Zoom", "Fit"); "ok".into() }
        ["zoom", n]             => {
            if let Ok(z) = n.parse::<f32>() { window.set_view_zoom(z.clamp(0.02, 64.0)); }
            "ok".into()
        }
        ["region", "load", path]  => { region_load_path(window, st, Path::new(path)); "ok".into() }
        ["region", "save", path]  => {
            if let Some(f) = st.active_frame() {
                match ds9_marker::write_reg(path, &f.markers) {
                    Ok(()) => format!("ok {} regions", f.markers.len()),
                    Err(e) => format!("err {e}"),
                }
            } else { "err no active frame".into() }
        }
        ["file", "open", path]    => { load_into(window, st, Path::new(path)); "ok".into() }
        ["save", "png", path]     => {
            let Some(f) = st.active_frame() else { return "err no frame".into() };
            let rgba = render_rgba_for_frame(f);
            let (w_, h_) = (f.fits.width as u32, f.fits.height as u32);
            let res = (|| -> std::io::Result<()> {
                let file = std::fs::File::create(path)?;
                let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w_, h_);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let mut wh = enc.write_header().map_err(|e| std::io::Error::other(e.to_string()))?;
                wh.write_image_data(&rgba).map_err(|e| std::io::Error::other(e.to_string()))
            })();
            match res { Ok(()) => "ok".into(), Err(e) => format!("err {e}") }
        }
        ["save", "fits", path]    => {
            let Some(f) = st.active_frame() else { return "err no frame".into() };
            match write_basic_fits(Path::new(path), &f.fits) {
                Ok(()) => "ok".into(), Err(e) => format!("err {e}"),
            }
        }
        ["sextractor"] => {
            run_sextractor(window, st);
            "ok".into()
        }
        ["crop"]       => { handle_menu(window, st, "Edit", "Crop to Selected"); "ok".into() }
        ["crop", "reset"] => { handle_menu(window, st, "Edit", "Reset Crop"); "ok".into() }
        ["projection"] => { handle_menu(window, st, "Region", "Projection…"); "ok".into() }
        ["centroid"]   => { handle_menu(window, st, "Region", "Centroid"); "ok".into() }
        ["radial"]     => { handle_menu(window, st, "Region", "Radial Profile…"); "ok".into() }
        ["hdu", "next"]   => { advance_hdu(window, st); "ok".into() }
        ["hdu", "list"]   => {
            let Some(p) = st.active_frame().and_then(|f| f.source_path.clone())
                else { return "err no source path".into() };
            match ds9_fits::enumerate_hdus(&p) {
                Ok(v) => v.iter().map(|h| format!("{} {} {}", h.idx, h.kind, h.name))
                    .collect::<Vec<_>>().join("\n"),
                Err(e) => format!("err {e}"),
            }
        }
        ["hdu", n] => {
            match n.parse::<usize>() {
                Ok(i) => { load_hdu_into_active(window, st, i); "ok".into() }
                Err(_) => "err: hdu N | hdu next | hdu list".into(),
            }
        }
        ["movie", "on"]   => { window.set_hdu_movie_active(true);  "ok".into() }
        ["movie", "off"]  => { window.set_hdu_movie_active(false); "ok".into() }
        ["value"] => {
            let cx = window.get_cursor_image_x() as i32;
            let cy = window.get_cursor_image_y() as i32;
            if let Some(f) = st.active_frame() {
                let img = &f.fits;
                if cx >= 0 && cy >= 0 && (cx as usize) < img.width && (cy as usize) < img.height {
                    let v = img.data[cy as usize * img.width + cx as usize];
                    format!("ok {v}")
                } else { "err out of bounds".into() }
            } else { "err no frame".into() }
        }
        ["help"] => "commands: quit|exit | xpaaccess | version | mode M | \
            frame next|previous|N|new|delete|clear|mosaic|tile | \
            pan to X Y | crosshair [X Y|clear] | \
            scale S | scale limits LO HI | \
            cmap C | cmap invert | bin N | \
            zoom in|out|fit|N | \
            region load|save P | regions load|save P | regions delete all|N | \
            file open P | save png|fits P | value | sextractor | \
            samp image|votable | \
            crop [reset] | projection | centroid | radial | \
            hdu next|list|N | movie on|off | help".into(),
        _ => format!("err unknown: {line}"),
    }
}

/// Helper used by both region_load (via dialog) and dispatch_ipc.
fn region_load_path(window: &MainWindow, st: &mut State, p: &Path) {
    let Some(f) = st.active_frame_mut() else {
        window.set_status_text("region load: no active frame".into()); return;
    };
    let text = match std::fs::read_to_string(p) {
        Ok(t) => t,
        Err(e) => { window.set_status_text(format!("region load: {e}").into()); return; }
    };
    let parsed = match f.fits.wcs.as_ref() {
        Some(w) => ds9_marker::parse_reg_with_wcs(&text, w),
        None    => ds9_marker::parse_reg(&text),
    };
    match parsed {
        Ok(ms) => {
            let n = ms.len();
            f.markers = ms; f.selected_marker = None;
            window.set_status_text(format!("loaded {n} regions from {}", p.display()).into());
            refresh_view(window, st);
        }
        Err(e) => window.set_status_text(format!("region parse error: {e}").into()),
    }
}

/// Drive the line-based IPC dispatch over a single duplex byte stream.
/// Reads commands until EOF, calls `dispatch_ipc` on the UI thread, writes
/// the response back. Used for both Unix-socket and TCP transports.
fn handle_ipc_stream<S>(weak: slint::Weak<MainWindow>, mut stream: S)
where S: std::io::Read + std::io::Write + Send + 'static + TryCloneStream<Cloned = S> {
    use std::io::{BufRead, BufReader};
    let read = match stream.try_clone_stream() { Ok(s) => s, Err(_) => return };
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    while reader.read_line(&mut line).map(|n| n > 0).unwrap_or(false) {
        let cmd = line.trim().to_string();
        line.clear();
        if cmd.is_empty() { continue; }
        let weak = weak.clone();
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let _ = slint::invoke_from_event_loop(move || {
            let resp = if let Some(w) = weak.upgrade() {
                STATE_FOR_IPC.with(|c| {
                    if let Some(st) = c.borrow().as_ref() {
                        dispatch_ipc(&w, &mut st.borrow_mut(), &cmd)
                    } else { "err state unavailable".into() }
                })
            } else { "err window gone".into() };
            let _ = tx.send(resp);
        });
        let resp = rx.recv().unwrap_or_else(|_| "err no response".into());
        let _ = writeln!(stream, "{resp}");
    }
}

/// Tiny abstraction for "give me a clone of this socket I can read from in a
/// separate buffered reader". Only impl'd for the two stream types we use.
trait TryCloneStream {
    type Cloned;
    fn try_clone_stream(&self) -> std::io::Result<Self::Cloned>;
}
impl TryCloneStream for std::os::unix::net::UnixStream {
    type Cloned = Self;
    fn try_clone_stream(&self) -> std::io::Result<Self> { self.try_clone() }
}
impl TryCloneStream for std::net::TcpStream {
    type Cloned = Self;
    fn try_clone_stream(&self) -> std::io::Result<Self> { self.try_clone() }
}

/// Spawn a thread that listens on a Unix-domain socket for line-based IPC
/// commands. Returns the path so callers can advertise it.
fn start_ipc_server(weak: slint::Weak<MainWindow>, state: Rc<RefCell<State>>) -> Option<PathBuf> {
    use std::os::unix::net::UnixListener;

    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
    let sock = dir.join(format!("ds9-rust-{user}.sock"));
    let _ = std::fs::remove_file(&sock);
    let listener = match UnixListener::bind(&sock) {
        Ok(l) => l,
        Err(e) => { eprintln!("ds9-rust IPC: bind {} failed: {e}", sock.display()); return None; }
    };

    let _state = state; // keep alive in closure
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let weak = weak.clone();
            std::thread::spawn(move || handle_ipc_stream(weak, stream));
        }
    });
    Some(sock)
}

/// Bind a localhost-only TCP listener for the same line-based IPC. Picks an
/// ephemeral port (or `DS9_RUST_TCP_PORT` if set). Returns the bound port.
fn start_tcp_ipc_server(weak: slint::Weak<MainWindow>, state: Rc<RefCell<State>>) -> Option<u16> {
    use std::net::TcpListener;
    let port: u16 = env::var("DS9_RUST_TCP_PORT").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let bind_addr = format!("127.0.0.1:{port}");
    let listener = match TcpListener::bind(&bind_addr) {
        Ok(l) => l,
        Err(e) => { eprintln!("ds9-rust IPC TCP: bind {bind_addr} failed: {e}"); return None; }
    };
    let port = listener.local_addr().ok()?.port();
    let _state = state;
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let weak = weak.clone();
            std::thread::spawn(move || handle_ipc_stream(weak, stream));
        }
    });
    Some(port)
}

/// Write the discovery file `~/.ds9-rust/xpa.info` listing how clients can
/// reach this instance (analogous to xpans). Best-effort; failures are
/// non-fatal.
fn write_discovery_file(uds: Option<&Path>, tcp_port: Option<u16>) {
    let Some(home) = env::var_os("HOME") else { return };
    let dir = PathBuf::from(home).join(".ds9-rust");
    let _ = std::fs::create_dir_all(&dir);
    let pid = std::process::id();
    let mut body = format!("pid={}\n", pid);
    if let Some(p) = uds { body.push_str(&format!("unix={}\n", p.display())); }
    if let Some(p) = tcp_port { body.push_str(&format!("tcp=127.0.0.1:{p}\n")); }
    let _ = std::fs::write(dir.join("xpa.info"), body);
}

thread_local! {
    /// Bridge so the IPC dispatch closure (which only captures Send things)
    /// can reach the shared State without smuggling a !Send Rc across threads.
    static STATE_FOR_IPC: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------- main --

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().collect();
    let win = MainWindow::new()?;
    let state = Rc::new(RefCell::new(State::new()));
    // Pull persisted preferences (if any) before loading the first file so
    // those defaults stick to every frame in this session.
    state.borrow_mut().prefs = load_prefs_from_disk();

    refresh_view(&win, &state.borrow());

    if let Some(p) = argv.get(1) {
        load_into(&win, &mut state.borrow_mut(), Path::new(p));
    }
    if let Some(p) = argv.get(2) {
        if let Ok(text) = std::fs::read_to_string(p) {
            if let Ok(ms) = ds9_marker::parse_reg(&text) {
                if let Some(f) = state.borrow_mut().active_frame_mut() {
                    f.markers = ms;
                }
                refresh_view(&win, &state.borrow());
                win.set_status_text(format!("loaded regions from {p}").into());
            }
        }
    }
    if let Some(p) = argv.get(3) {
        if let Ok(cat) = Catalog::from_path(p) {
            let n = cat.len();
            if let Some(f) = state.borrow_mut().active_frame_mut() {
                f.catalog = Some(cat);
            }
            refresh_view(&win, &state.borrow());
            win.set_status_text(format!("loaded catalog ({n} rows) from {p}").into());
        }
    }

    // ---- file open dialog ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_request_open_file(move || {
            let Some(w) = weak.upgrade() else { return };
            let chosen: Option<PathBuf> = rfd::FileDialog::new()
                .set_title("Open FITS")
                .add_filter("FITS", &["fits", "fit", "fts", "fz"])
                .add_filter("All", &["*"])
                .pick_file();
            if let Some(p) = chosen {
                load_into(&w, &mut state.borrow_mut(), &p);
            }
        });
    }

    // ---- canvas mouse → image-space coordinates + value + WCS readout ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_canvas_mouse_move(move |x, y| {
            let Some(w) = weak.upgrade() else { return };

            // ----- marker drag (edit mode only) -----
            {
                let mut s = state.borrow_mut();
                let active_fits = s.active_frame().map(|f| display_to_fits(x as f64, y as f64, f));
                if let (Some((fits_x, fits_y)), Some(idx), Some((px, py))) =
                    (active_fits, s.dragging_marker, s.last_drag_fits)
                {
                    if let Some(f) = s.active_frame_mut() {
                        if idx < f.markers.len() {
                            let dx = fits_x - px;
                            let dy = fits_y - py;
                            translate_marker(&mut f.markers[idx], dx, dy);
                            s.last_drag_fits = Some((fits_x, fits_y));
                            w.set_status_text(format!(
                                "drag region #{}  Δ=({:+.1}, {:+.1})", idx + 1, dx, dy,
                            ).into());
                            if let Some(f) = s.active_frame() {
                                w.set_markers(build_mark_model(f));
                            }
                            return;
                        }
                    }
                }
            }

            let st = state.borrow();
            let Some(f) = st.active_frame() else {
                w.set_info_coords("x: ——      y: ——".into());
                w.set_info_value("value: ——".into());
                w.set_info_wcs("wcs: ——".into());
                return;
            };
            let (fits_x, fits_y) = display_to_fits(x as f64, y as f64, f);
            w.set_info_coords(format!("x: {:>7.1}    y: {:>7.1}", fits_x, fits_y).into());

            let ux = (fits_x - 1.0).round() as i32;
            let uy = (fits_y - 1.0).round() as i32;
            let img = &f.fits;
            let v_text = if ux >= 0 && uy >= 0 && (ux as usize) < img.width && (uy as usize) < img.height {
                let v = img.data[uy as usize * img.width + ux as usize];
                format!("value: {v:>10.4}")
            } else { "value: ——".to_string() };
            let wcs_text = if let Some(wcs) = &img.wcs {
                let (ra, dec) = wcs.pix_to_world(fits_x, fits_y);
                format!("{} {}", wcs.radesys.to_lowercase(), ds9_fits::format_sexagesimal(ra, dec))
            } else { "wcs: ——".to_string() };
            w.set_info_value(v_text.into());
            w.set_info_wcs(wcs_text.into());
        });
    }

    // ---- click ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_canvas_clicked(move |x, y| {
            let Some(w) = weak.upgrade() else { return };
            // Tile mode: click a cell to select that frame and exit tile.
            {
                let mut s = state.borrow_mut();
                if s.tile_mode {
                    let cw = (w.get_canvas_w() as f32 as i32).max(64) as usize;
                    let ch = (w.get_canvas_h() as f32 as i32).max(64) as usize;
                    if let Some(idx) = tile_hit(&s, cw, ch, x as f64, y as f64) {
                        s.tile_mode = false;
                        let prev = s.active;
                        s.active = idx;
                        push_view_to_window(&w, &s);
                        refresh_view(&w, &s);
                        w.set_status_text(format!(
                            "tile: selected frame {} (was {})", idx + 1, prev + 1
                        ).into());
                    }
                    return;
                }
            }
            let mode = w.get_active_mode().to_string();
            // Crosshair mode: drop / move the session crosshair at the click.
            if mode == "crosshair" {
                let mut s = state.borrow_mut();
                if s.active_frame().is_some() {
                    set_crosshair_at_display(&w, &mut s, x as f64, y as f64);
                    return;
                }
            }
            // FITS coords from the click — orientation-aware
            let fits_xy = {
                let s = state.borrow();
                s.active_frame().map(|f| display_to_fits(x as f64, y as f64, f))
            };
            let Some((cx_fits, cy_fits)) = fits_xy else {
                w.set_status_text(format!("click @ image ({:.1}, {:.1})", x, y).into());
                return;
            };
            // Edit mode: drop a small circle at the click
            if mode == "edit" {
                let r = 6.0;
                let mut s = state.borrow_mut();
                if let Some(f) = s.active_frame_mut() {
                    f.markers.push(Marker::circle(cx_fits, cy_fits, r));
                    w.set_status_text(format!("region @ ({cx_fits:.1}, {cy_fits:.1})  r={r:.1}").into());
                    refresh_view(&w, &s);
                    return;
                }
            }
            // Otherwise, try to select a nearby catalog source — the OGFinder
            // workflow: click a star on the image, the table jumps to the row.
            let mut s = state.borrow_mut();
            let hit = s.active_frame()
                .and_then(|f| f.catalog.as_ref())
                .and_then(|cat| nearest_catalog_index(cat, cx_fits, cy_fits, 8.0));
            if let Some(idx) = hit {
                if let Some(f) = s.active_frame_mut() { f.selected_catalog = Some(idx); }
                w.set_status_text(format!(
                    "selected catalog row {} @ ({:.1}, {:.1})",
                    idx + 1, cx_fits, cy_fits
                ).into());
                refresh_view(&w, &s);
                return;
            }
            w.set_status_text(format!("click @ image ({:.1}, {:.1})", x, y).into());
        });
    }

    // ---- catalog row click → recenter + select ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_catalog_row_activated(move |idx| {
            let Some(w) = weak.upgrade() else { return };
            let idx = idx as usize;
            let mut s = state.borrow_mut();
            // Re-derive the (x, y) for that row from the catalog itself so we
            // do not drift if the slint model and Rust state ever disagree.
            let xy = s.active_frame()
                .and_then(|f| f.catalog.as_ref())
                .and_then(|c| c.xy_iter().nth(idx));
            if let Some((x, y)) = xy {
                if let Some(f) = s.active_frame_mut() { f.selected_catalog = Some(idx); }
                refresh_view(&w, &s);
                if let Some(f) = s.active_frame() {
                    let (display_x, display_y) = fits_to_display_oriented(x, y, f);
                    w.invoke_recenter_view_on(display_x, display_y);
                }
                w.set_status_text(
                    format!("row {} @ ({:.1}, {:.1})", idx + 1, x, y).into(),
                );
            }
        });
    }

    // ---- canvas pointer-down: select / start drag ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_canvas_pressed(move |x, y| {
            let Some(w) = weak.upgrade() else { return };
            let mut s = state.borrow_mut();
            let xy_hit = s.active_frame().map(|f| {
                let (fx, fy) = display_to_fits(x as f64, y as f64, f);
                let hit = hit_test_markers(&f.markers, fx, fy);
                (fx, fy, hit)
            });
            let Some((fx, fy, hit)) = xy_hit else {
                w.set_marker_drag_active(false); return;
            };
            match hit {
                Some(idx) => {
                    if let Some(f) = s.active_frame_mut() { f.selected_marker = Some(idx); }
                    let edit = w.get_active_mode().to_string() == "edit";
                    if edit {
                        s.dragging_marker = Some(idx);
                        s.last_drag_fits = Some((fx, fy));
                    }
                    w.set_marker_drag_active(true);
                    w.set_status_text(format!(
                        "{} region #{}", if edit { "drag" } else { "select" }, idx + 1
                    ).into());
                    refresh_view(&w, &s);
                }
                None => {
                    s.dragging_marker = None;
                    s.last_drag_fits = None;
                    w.set_marker_drag_active(false);
                }
            }
        });
    }

    // ---- canvas pointer-up after a marker press: clear drag state ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_canvas_released(move || {
            let Some(w) = weak.upgrade() else { return };
            let mut s = state.borrow_mut();
            s.dragging_marker = None;
            s.last_drag_fits = None;
            w.set_marker_drag_active(false);
        });
    }

    // ---- mode toggle ----
    {
        let weak = win.as_weak();
        win.on_set_mode(move |mode| {
            let Some(w) = weak.upgrade() else { return };
            w.set_active_mode(mode.clone());
            w.set_status_text(format!("mode: {mode}").into());
        });
    }

    // ---- menubar ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_menu_action(move |menu, item| {
            let Some(w) = weak.upgrade() else { return };
            handle_menu(&w, &mut state.borrow_mut(), &menu, &item);
        });
    }

    // ---- blink timer: while blink-active, advance Frame Next every 500ms ----
    let blink_timer = slint::Timer::default();
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        blink_timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(500),
            move || {
                let Some(w) = weak.upgrade() else { return };
                if !w.get_blink_active() { return; }
                let mut s = state.borrow_mut();
                if s.frames.len() < 2 { return; }
                let n = s.frames.len();
                let target = (s.active + 1) % n;
                switch_frame(&w, &mut s, target);
            },
        );
    }

    // ---- HDU movie timer: while hdu-movie-active, advance to next loadable
    //      HDU of the active frame's source file every 800 ms ----
    let hdu_movie_timer = slint::Timer::default();
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        hdu_movie_timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(800),
            move || {
                let Some(w) = weak.upgrade() else { return };
                if !w.get_hdu_movie_active() { return; }
                advance_hdu(&w, &mut state.borrow_mut());
            },
        );
    }

    // ---- HDU panel row click: load that HDU into the active frame ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_hdu_row_clicked(move |idx| {
            let Some(w) = weak.upgrade() else { return };
            if idx < 0 { return; }
            load_hdu_into_active(&w, &mut state.borrow_mut(), idx as usize);
        });
    }

    // ---- Region property editor — Apply + Revert buttons ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_region_prop_apply(move || {
            let Some(w) = weak.upgrade() else { return };
            apply_region_props(&w, &mut state.borrow_mut());
        });
    }
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_region_prop_revert(move || {
            let Some(w) = weak.upgrade() else { return };
            populate_region_props(&w, &state.borrow());
        });
    }

    // ---- Preferences — Apply / Save / Reset ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_prefs_apply(move || {
            let Some(w) = weak.upgrade() else { return };
            apply_prefs_from_panel(&w, &mut state.borrow_mut());
        });
    }
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_prefs_save(move || {
            let Some(w) = weak.upgrade() else { return };
            // Write the *currently shown* values, not the last-applied ones,
            // so users can save without first clicking Apply.
            apply_prefs_from_panel(&w, &mut state.borrow_mut());
            let st = state.borrow();
            match save_prefs_to_disk(&st.prefs) {
                Ok(p)  => w.set_prefs_status(format!("saved → {}", p.display()).into()),
                Err(e) => w.set_prefs_status(format!("save failed: {e}").into()),
            }
        });
    }
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_prefs_reset(move || {
            let Some(w) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                st.prefs = Prefs::default();
            }
            populate_prefs_panel(&w, &state.borrow());
            w.set_prefs_status("reset to factory defaults (not yet applied — click Apply).".into());
        });
    }

    // ---- Online catalog query — Resolve / VizieR / NED ----
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_netcat_resolve(move || {
            let Some(w) = weak.upgrade() else { return };
            netcat_resolve(&w, &mut state.borrow_mut());
        });
    }
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_netcat_vizier(move || {
            let Some(w) = weak.upgrade() else { return };
            netcat_vizier(&w, &mut state.borrow_mut());
        });
    }
    {
        let weak = win.as_weak();
        let state = Rc::clone(&state);
        win.on_netcat_ned(move || {
            let Some(w) = weak.upgrade() else { return };
            netcat_ned(&w, &mut state.borrow_mut());
        });
    }

    // ---- IPC servers (Unix-domain socket + localhost TCP) ----
    STATE_FOR_IPC.with(|c| { *c.borrow_mut() = Some(Rc::clone(&state)); });
    let uds = start_ipc_server(win.as_weak(), Rc::clone(&state));
    let tcp = start_tcp_ipc_server(win.as_weak(), Rc::clone(&state));
    if let Some(p) = &uds {
        eprintln!("ds9-rust IPC (unix) listening on {}", p.display());
    }
    if let Some(port) = tcp {
        eprintln!("ds9-rust IPC (tcp) listening on 127.0.0.1:{port}");
    }
    write_discovery_file(uds.as_deref(), tcp);
    let banner = match (&uds, tcp) {
        (Some(p), Some(port)) => format!("ready  (ipc: {} | tcp:{})", p.display(), port),
        (Some(p), None)       => format!("ready  (ipc: {})", p.display()),
        (None, Some(port))    => format!("ready  (tcp: 127.0.0.1:{port})"),
        (None, None)          => "ready  (no ipc)".into(),
    };
    win.set_status_text(banner.into());

    win.run()?;
    Ok(())
}
