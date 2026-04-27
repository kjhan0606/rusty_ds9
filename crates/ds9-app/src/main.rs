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

struct State {
    fits: Option<FitsImage>,
    stretch: Stretch,
    limits_mode: LimitsMode,
    cmap: Colormap,
    markers: Vec<Marker>,
    catalog: Option<Catalog>,
}

impl State {
    fn new() -> Self {
        Self {
            fits: None,
            stretch: Stretch::Linear,
            limits_mode: LimitsMode::Zscale,
            cmap: Colormap::Grey,
            markers: Vec::new(),
            catalog: None,
        }
    }

    fn limits(&self, img: &FitsImage) -> Limits {
        match self.limits_mode {
            LimitsMode::Zscale => Limits::zscale(img),
            LimitsMode::MinMax => Limits::minmax(img),
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

// ---------------------------------------------------------------- render --

fn render_image(img: &FitsImage, st: &State) -> Image {
    let limits = st.limits(img);
    let rgba = ds9_image::render_rgba_flipped(img, limits, st.stretch, st.cmap);
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(img.width as u32, img.height as u32);
    buf.make_mut_bytes().copy_from_slice(&rgba);
    Image::from_rgba8(buf)
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

fn build_mark_model(st: &State) -> ModelRc<Mark> {
    let h = st.fits.as_ref().map(|i| i.height).unwrap_or(0);
    let cat_count = st.catalog.as_ref().map(|c| c.len()).unwrap_or(0).min(5000);
    let mut out: Vec<Mark> = Vec::with_capacity(st.markers.len() + cat_count);

    // catalog points first so user-drawn regions paint on top
    if let Some(cat) = &st.catalog {
        let amber = slint::Color::from_argb_u8(0xff, 0xff, 0xc1, 0x07);
        for (i, (x, y)) in cat.xy_iter().enumerate() {
            if i >= 5000 { break; }
            let (cx, cy) = fits_to_display(x, y, h);
            out.push(Mark { kind: 0, cx, cy, rx: 4.0, ry: 4.0, color: amber });
        }
    }

    for m in &st.markers {
        let color = marker_color(m);
        let mark = match &m.shape {
            MShape::Circle { center: c, r } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *r as f32, ry: *r as f32, color })
            }
            MShape::Box { center: c, w, h: bh, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 1, cx, cy, rx: (*w as f32) / 2.0, ry: (*bh as f32) / 2.0, color })
            }
            MShape::Ellipse { center: c, a, b, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *a as f32, ry: *b as f32, color })
            }
            MShape::Annulus { center: c, r_outer, .. } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 0, cx, cy, rx: *r_outer as f32, ry: *r_outer as f32, color })
            }
            MShape::Point { center: c } => {
                let (cx, cy) = fits_to_display(c.x, c.y, h);
                Some(Mark { kind: 1, cx, cy, rx: 2.0, ry: 2.0, color })
            }
            // line / polygon / compass / text not yet drawn
            _ => None,
        };
        if let Some(m) = mark { out.push(m); }
    }
    ModelRc::new(VecModel::from(out))
}

/// Push current state-derived visuals (image, colorbar, markers, info badges) into the window.
fn refresh_view(window: &MainWindow, st: &State) {
    window.set_active_stretch(st.stretch_label().into());
    window.set_active_limits(st.limits_label().into());
    window.set_active_cmap(st.cmap.name().into());
    window.set_colorbar_strip(make_colorbar_strip(st.cmap));
    window.set_markers(build_mark_model(st));
    if let Some(img) = &st.fits {
        window.set_fits_image(render_image(img, st));
    }
}

fn load_into(window: &MainWindow, st: &mut State, path: &Path) {
    match ds9_fits::load(path) {
        Ok(img) => {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("(fits)")
                .to_string();
            window.set_fits_width(img.width as i32);
            window.set_fits_height(img.height as i32);
            window.set_info_filename(name.into());
            window.set_info_object("—".into());
            window.set_view_zoom(fit_zoom(img.width, img.height));
            window.set_view_pan_x(0.0);
            window.set_view_pan_y(0.0);
            // seed cursor to image centre so the magnifier has content even
            // before the user has moved the mouse onto the canvas
            window.set_cursor_image_x((img.width  / 2) as f32);
            window.set_cursor_image_y((img.height / 2) as f32);
            window.set_status_text(
                format!(
                    "loaded {} × {}    range {:.4} … {:.4}",
                    img.width, img.height, img.min, img.max
                )
                .into(),
            );
            // seed WCS readout with the image centre so the field is non-empty
            // before the user starts hovering.
            if let Some(wcs) = &img.wcs {
                let cx = img.width  as f64 / 2.0;
                let cy = img.height as f64 / 2.0;
                let (ra, dec) = wcs.pix_to_world(cx, cy);
                window.set_info_wcs(format!(
                    "{} {}", wcs.radesys.to_lowercase(), ds9_fits::format_sexagesimal(ra, dec)
                ).into());
            }
            st.fits = Some(img);
            refresh_view(window, st);
        }
        Err(e) => {
            window.set_status_text(format!("error: {e}").into());
        }
    }
}

// ---------------------------------------------------------------- regions --

fn region_new_sample(st: &mut State) {
    if let Some(img) = &st.fits {
        // sample circle at frame centre, radius ≈ 5% of min dim
        let cx = img.width  as f64 / 2.0;
        let cy = img.height as f64 / 2.0;
        let r  = (img.width.min(img.height) as f64 * 0.05).max(4.0);
        st.markers.push(Marker::circle(cx, cy, r));
    } else {
        st.markers.push(Marker::circle(100.0, 100.0, 10.0));
    }
}

fn region_load(window: &MainWindow, st: &mut State) {
    let chosen: Option<PathBuf> = rfd::FileDialog::new()
        .set_title("Load DS9 region file")
        .add_filter("Region", &["reg"])
        .add_filter("All", &["*"])
        .pick_file();
    let Some(p) = chosen else { return };
    match std::fs::read_to_string(&p) {
        Ok(text) => match ds9_marker::parse_reg(&text) {
            Ok(ms) => {
                let n = ms.len();
                st.markers = ms;
                window.set_status_text(format!("loaded {n} regions from {}", p.display()).into());
                refresh_view(window, st);
            }
            Err(e) => window.set_status_text(format!("region parse error: {e}").into()),
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
            st.catalog = Some(cat);
            refresh_view(window, st);
        }
        Err(e) => window.set_status_text(format!("catalog read error: {e}").into()),
    }
}

fn catalog_clear(window: &MainWindow, st: &mut State) {
    st.catalog = None;
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
    match ds9_marker::write_reg(&p, &st.markers) {
        Ok(()) => window.set_status_text(format!("wrote {} regions → {}", st.markers.len(), p.display()).into()),
        Err(e) => window.set_status_text(format!("region write error: {e}").into()),
    }
}

// ---------------------------------------------------------------- menus --

fn handle_menu(window: &MainWindow, st: &mut State, menu: &str, item: &str) {
    match (menu, item) {
        // File
        ("File", "Open…") => window.invoke_request_open_file(),
        ("File", "Quit")  => { let _ = slint::quit_event_loop(); }

        // Scale — stretch
        ("Scale", "linear")  => { st.stretch = Stretch::Linear;  refresh_view(window, st); }
        ("Scale", "log")     => { st.stretch = Stretch::Log;     refresh_view(window, st); }
        ("Scale", "sqrt")    => { st.stretch = Stretch::Sqrt;    refresh_view(window, st); }
        ("Scale", "squared") => { st.stretch = Stretch::Squared; refresh_view(window, st); }
        ("Scale", "asinh")   => { st.stretch = Stretch::Asinh;   refresh_view(window, st); }
        ("Scale", "sinh")    => { st.stretch = Stretch::Sinh;    refresh_view(window, st); }
        // Scale — limits
        ("Scale", "minmax")  => { st.limits_mode = LimitsMode::MinMax; refresh_view(window, st); }
        ("Scale", "zscale")  => { st.limits_mode = LimitsMode::Zscale; refresh_view(window, st); }

        // Color
        ("Color", name) => {
            if let Some(c) = Colormap::from_name(name) {
                st.cmap = c;
                refresh_view(window, st);
            }
        }

        // Region
        ("Region", "New")    => { region_new_sample(st); refresh_view(window, st); }
        ("Region", "Load…")  => { region_load(window, st); }
        ("Region", "Save…")  => { region_save(window, st); }
        ("Region", "Info")   => {
            window.set_status_text(format!("regions: {}", st.markers.len()).into());
        }

        // Catalog
        ("Catalog", "Load…") => { catalog_load(window, st); }
        ("Catalog", "Clear") => { catalog_clear(window, st); }
        ("Catalog", "Info")  => {
            let msg = match &st.catalog {
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

        // Zoom
        ("Zoom", "Zoom In") => {
            window.set_view_zoom((window.get_view_zoom() * 1.5).clamp(0.02, 64.0));
        }
        ("Zoom", "Zoom Out") => {
            window.set_view_zoom((window.get_view_zoom() / 1.5).clamp(0.02, 64.0));
        }
        ("Zoom", "Fit") => {
            if let Some(img) = &st.fits {
                window.set_view_zoom(fit_zoom(img.width, img.height));
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
                state.borrow_mut().markers = ms;
                refresh_view(&win, &state.borrow());
                win.set_status_text(format!("loaded regions from {p}").into());
            }
        }
    }
    if let Some(p) = argv.get(3) {
        if let Ok(cat) = Catalog::from_path(p) {
            let n = cat.len();
            state.borrow_mut().catalog = Some(cat);
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
            w.set_info_coords(format!("x: {:>7.1}    y: {:>7.1}", fits_x, fits_y).into());

            // pixel value + WCS lookup
            let ux = x as i32;
            let uy = (fy - 1.0) as i32;
            let st = state.borrow();
            let (v_text, wcs_text) = if let Some(img) = &st.fits {
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
            // In edit mode, a click drops a small circle at the click location.
            // Coords from slint are display-space (0-based, y-down); markers store FITS coords.
            if mode == "edit" {
                let h = w.get_fits_height();
                if h > 0 {
                    let cx_fits = (x + 1.0) as f64;
                    let cy_fits = (h as f32 - y) as f64;
                    let r = 6.0;
                    let mut s = state.borrow_mut();
                    s.markers.push(Marker::circle(cx_fits, cy_fits, r));
                    w.set_status_text(format!("region @ ({cx_fits:.1}, {cy_fits:.1})  r={r:.1}").into());
                    refresh_view(&w, &s);
                    return;
                }
            }
            w.set_status_text(format!("click @ image ({:.1}, {:.1})", x, y).into());
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

    win.run()?;
    Ok(())
}
