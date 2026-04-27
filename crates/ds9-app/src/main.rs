use anyhow::Result;
use ds9_fits::FitsImage;
use ds9_image::{Colormap, Limits, Stretch};
use ds9_catalog::Catalog;
use ds9_marker::{Marker, Shape as MShape};
use slint::{ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::cell::RefCell;
use std::env;
use std::path::{Path, PathBuf};
use std::rc::Rc;

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
    /// Gaussian-smooth kernel σ in pixels (0 = off).
    smooth_sigma: f32,
    /// Block-bin factor (1 = off, 2/4/8/16/32 chunkify).
    bin_factor: u32,
}

impl Frame {
    fn new(fits: FitsImage, name: String) -> Self {
        let (w, h) = (fits.width, fits.height);
        Self {
            fits,
            name,
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
            bin_factor: 1,
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

struct State {
    frames: Vec<Frame>,
    /// Index into `frames`; only meaningful when `frames` is non-empty.
    active: usize,
    /// index into the active frame's markers (transient, not per-frame state)
    dragging_marker: Option<usize>,
    last_drag_fits: Option<(f64, f64)>,
}

impl State {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            active: 0,
            dragging_marker: None,
            last_drag_fits: None,
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

/// Apply per-frame bin/smooth filters before stretching, returning the raw
/// flipped RGBA bytes. Caller wraps these into an Image / PNG / etc.
fn render_rgba_for_frame(f: &Frame) -> Vec<u8> {
    let mut owned: Option<FitsImage> = None;
    if f.bin_factor > 1 {
        owned = Some(ds9_image::bin_average(&f.fits, f.bin_factor));
    }
    if f.smooth_sigma > 0.0 {
        let src = owned.as_ref().unwrap_or(&f.fits);
        owned = Some(ds9_image::smooth_gaussian(src, f.smooth_sigma));
    }
    let img: &FitsImage = owned.as_ref().unwrap_or(&f.fits);
    let limits = match f.limits_mode {
        LimitsMode::Zscale => Limits::zscale(img),
        LimitsMode::MinMax => Limits::minmax(img),
        LimitsMode::User { low, high } => Limits { low, high },
    };
    ds9_image::render_rgba_flipped(img, limits, f.stretch, f.cmap)
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
fn fits_to_display(cx: f64, cy: f64, h: usize) -> (f32, f32) {
    ((cx - 1.0) as f32, (h as f32 - cy as f32))
}

fn marker_color(m: &Marker) -> slint::Color {
    slint::Color::from_argb_u8(m.color[3], m.color[0], m.color[1], m.color[2])
}

fn build_mark_model(f: &Frame) -> ModelRc<Mark> {
    let h = f.fits.height;
    let cat_count = f.catalog.as_ref().map(|c| c.len()).unwrap_or(0).min(5000);
    let mut out: Vec<Mark> = Vec::with_capacity(f.markers.len() + cat_count);

    // catalog points first so user-drawn regions paint on top
    if let Some(cat) = &f.catalog {
        let amber = slint::Color::from_argb_u8(0xff, 0xff, 0xc1, 0x07);
        for (i, (x, y)) in cat.xy_iter().enumerate() {
            if i >= 5000 { break; }
            let (cx, cy) = fits_to_display(x, y, h);
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
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *r as f32, ry: *r as f32, color, selected: sel })
            }
            MShape::Box { center: c, w, h: bh, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 1, cx, cy, rx: (*w as f32) / 2.0, ry: (*bh as f32) / 2.0, color, selected: sel })
            }
            MShape::Ellipse { center: c, a, b, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *a as f32, ry: *b as f32, color, selected: sel })
            }
            MShape::Annulus { center: c, r_outer, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *r_outer as f32, ry: *r_outer as f32, color, selected: sel })
            }
            MShape::Point { center: c } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 1, cx, cy, rx: 2.0, ry: 2.0, color, selected: sel })
            }
            // line / polygon / compass / text not yet drawn
            _ => None,
        };
        if let Some(m) = mark { out.push(m); }
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
    window.set_markers(build_mark_model(f));
    window.set_catalog_rows(build_catalog_model(f));
    window.set_catalog_selected(f.selected_catalog.map(|i| i as i32).unwrap_or(-1));
    window.set_fits_image(render_image(f));
    window.set_fits_width(f.fits.width as i32);
    window.set_fits_height(f.fits.height as i32);
    window.set_info_filename(f.name.clone().into());
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
    Image::from_rgba8(buf)
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
            st.frames.push(Frame::new(img, name));
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

// ---------------------------------------------------------------- menus --

fn handle_menu(window: &MainWindow, st: &mut State, menu: &str, item: &str) {
    match (menu, item) {
        // File
        ("File", "Open…") => window.invoke_request_open_file(),
        ("File", "Save Image…") => save_image_png(window, st),
        ("File", "Save FITS…")  => save_image_fits(window, st),
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
        ("File", "Quit")  => { let _ = slint::quit_event_loop(); }

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
        ("Bin", n) => {
            if let Ok(factor) = n.parse::<u32>() {
                if let Some(f) = st.active_frame_mut() { f.bin_factor = factor.max(1); }
                refresh_view(window, st);
                window.set_status_text(format!("bin: {n}×{n}").into());
            }
        }

        // Color
        ("Color", name) => {
            if let Some(c) = Colormap::from_name(name) {
                if let Some(f) = st.active_frame_mut() { f.cmap = c; }
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
            if let Some(f) = st.active_frame_mut() {
                // 0 → 2 → 4 → 8 → 0
                f.smooth_sigma = match f.smooth_sigma {
                    s if s <= 0.0 => 2.0,
                    s if s < 3.0  => 4.0,
                    s if s < 6.0  => 8.0,
                    _             => 0.0,
                };
                let label = if f.smooth_sigma <= 0.0 {
                    "off".to_string()
                } else {
                    format!("σ = {:.0}px", f.smooth_sigma)
                };
                window.set_status_text(format!("smooth: {label}").into());
                refresh_view(window, st);
            }
        }
        ("Analysis", "Smooth Off") => {
            if let Some(f) = st.active_frame_mut() { f.smooth_sigma = 0.0; }
            window.set_status_text("smooth: off".into());
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
    let toks: Vec<&str> = line.splitn(3, ' ').collect();
    match toks.as_slice() {
        ["quit"] => { let _ = slint::quit_event_loop(); "ok".into() }
        ["frame", "next"]      => { handle_menu(window, st, "Frame", "Next"); "ok".into() }
        ["frame", "previous"]  => { handle_menu(window, st, "Frame", "Previous"); "ok".into() }
        ["frame", n] => {
            if let Ok(idx) = n.parse::<usize>() {
                if idx >= 1 { switch_frame(window, st, idx - 1); }
            }
            "ok".into()
        }
        ["scale", s]            => { handle_menu(window, st, "Scale", s); "ok".into() }
        ["cmap", c]             => { handle_menu(window, st, "Color", c); "ok".into() }
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
        ["help"] => "commands: quit | frame next|previous|N | scale S | cmap C | bin N | zoom in|out|fit|N | region load|save P | file open P | save png|fits P | value | help".into(),
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

/// Spawn a thread that listens on a Unix-domain socket for line-based IPC
/// commands. Returns the path so callers can advertise it.
fn start_ipc_server(weak: slint::Weak<MainWindow>, state: Rc<RefCell<State>>) -> Option<PathBuf> {
    use std::os::unix::net::UnixListener;
    use std::io::{BufRead, BufReader, Write};

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

    // Channel: IPC thread → UI thread
    let _state = state;  // keep alive in closure
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let weak = weak.clone();
            std::thread::spawn(move || {
                let read = match stream.try_clone() { Ok(s) => s, Err(_) => return };
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
            });
        }
    });
    Some(sock)
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
            let h = w.get_fits_height();
            let fy = if h > 0 { h as f32 - y } else { y };
            let fits_x = (x + 1.0) as f64;
            let fits_y = (fy + 1.0) as f64;

            // ----- marker drag (edit mode only) -----
            {
                let mut s = state.borrow_mut();
                if let (Some(idx), Some((px, py))) = (s.dragging_marker, s.last_drag_fits) {
                    if let Some(f) = s.active_frame_mut() {
                        if idx < f.markers.len() {
                            let dx = fits_x - px;
                            let dy = fits_y - py;
                            translate_marker(&mut f.markers[idx], dx, dy);
                            s.last_drag_fits = Some((fits_x, fits_y));
                            w.set_status_text(format!(
                                "drag region #{}  Δ=({:+.1}, {:+.1})", idx + 1, dx, dy,
                            ).into());
                            // re-emit just the marker model to avoid re-rendering the image
                            if let Some(f) = s.active_frame() {
                                w.set_markers(build_mark_model(f));
                            }
                            return;
                        }
                    }
                }
            }

            w.set_info_coords(format!("x: {:>7.1}    y: {:>7.1}", fits_x, fits_y).into());

            // pixel value + WCS lookup
            let ux = x as i32;
            let uy = (fy - 1.0) as i32;
            let st = state.borrow();
            let (v_text, wcs_text) = if let Some(f) = st.active_frame() {
                let img = &f.fits;
                let v = if ux >= 0 && uy >= 0 && (ux as usize) < img.width && (uy as usize) < img.height {
                    let v = img.data[uy as usize * img.width + ux as usize];
                    format!("value: {v:>10.4}")
                } else {
                    "value: ——".to_string()
                };
                let w_text = if let Some(wcs) = &img.wcs {
                    let (ra, dec) = wcs.pix_to_world(fits_x, fits_y);
                    format!("{} {}", wcs.radesys.to_lowercase(), ds9_fits::format_sexagesimal(ra, dec))
                } else {
                    "wcs: ——".to_string()
                };
                (v, w_text)
            } else {
                ("value: ——".to_string(), "wcs: ——".to_string())
            };
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
            let mode = w.get_active_mode().to_string();
            let h = w.get_fits_height();
            // In edit mode, a click drops a small circle at the click location.
            // Coords from slint are display-space (0-based, y-down); markers store FITS coords.
            if mode == "edit" && h > 0 {
                let cx_fits = (x + 1.0) as f64;
                let cy_fits = (h as f32 - y) as f64;
                let r = 6.0;
                let mut s = state.borrow_mut();
                if let Some(f) = s.active_frame_mut() {
                    f.markers.push(Marker::circle(cx_fits, cy_fits, r));
                    w.set_status_text(format!("region @ ({cx_fits:.1}, {cy_fits:.1})  r={r:.1}").into());
                    refresh_view(&w, &s);
                    return;
                }
            }
            // In any other mode, try to select a nearby catalog source — the
            // OGFinder workflow: click a star on the image, the table jumps to
            // the matching row.
            if h > 0 {
                let cx_fits = (x + 1.0) as f64;
                let cy_fits = (h as f32 - y) as f64;
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
                let h = w.get_fits_height();
                let display_x = (x - 1.0) as f32;
                let display_y = h as f32 - y as f32;
                w.invoke_recenter_view_on(display_x, display_y);
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
            let h = w.get_fits_height();
            if h <= 0 { w.set_marker_drag_active(false); return; }
            let fx = (x + 1.0) as f64;
            let fy = (h as f32 - y) as f64;
            let mut s = state.borrow_mut();
            let hit = s.active_frame().and_then(|f| hit_test_markers(&f.markers, fx, fy));
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

    // ---- IPC server (Unix-domain socket) ----
    STATE_FOR_IPC.with(|c| { *c.borrow_mut() = Some(Rc::clone(&state)); });
    if let Some(p) = start_ipc_server(win.as_weak(), Rc::clone(&state)) {
        eprintln!("ds9-rust IPC listening on {}", p.display());
        win.set_status_text(format!("ready  (ipc: {})", p.display()).into());
    }

    win.run()?;
    Ok(())
}
