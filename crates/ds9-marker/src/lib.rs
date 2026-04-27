//! Region/marker model + minimal DS9 `.reg` file I/O.
//! Supports `image`, `physical`, `fk5`/`icrs`, and `galactic` coordinate
//! systems for the most common shapes (circle / box / ellipse / annulus /
//! point / line / polygon). Sky-coord regions need a WCS — pass one via
//! [`parse_reg_with_wcs`]. The plain [`parse_reg`] entry point still works
//! and treats sky coords as `image` (back-compat).

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use ds9_fits::Wcs;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelPos { pub x: f64, pub y: f64 }

#[derive(Debug, Clone)]
pub enum Shape {
    Circle  { center: PixelPos, r: f64 },
    Ellipse { center: PixelPos, a: f64, b: f64, theta_deg: f64 },
    Box     { center: PixelPos, w: f64, h: f64, theta_deg: f64 },
    Annulus { center: PixelPos, r_inner: f64, r_outer: f64 },
    Polygon { points: Vec<PixelPos> },
    Point   { center: PixelPos },
    Line    { from: PixelPos, to: PixelPos },
    Compass { center: PixelPos, len: f64 },
    Text    { center: PixelPos, body: String },
}

#[derive(Debug, Clone)]
pub struct Marker {
    pub shape: Shape,
    pub color: [u8; 4],
    pub width: f32,
    pub tags: Vec<String>,
    pub highlight: bool,
}

impl Default for Marker {
    fn default() -> Self {
        Self {
            shape: Shape::Point { center: PixelPos { x: 0.0, y: 0.0 } },
            color: [0x4e, 0xc9, 0xb0, 0xff],   // ds9-rust accent teal
            width: 1.5,
            tags: Vec::new(),
            highlight: false,
        }
    }
}

impl Marker {
    pub fn circle(cx: f64, cy: f64, r: f64) -> Self {
        Self { shape: Shape::Circle { center: PixelPos { x: cx, y: cy }, r }, ..Default::default() }
    }
    pub fn rbox(cx: f64, cy: f64, w: f64, h: f64, theta_deg: f64) -> Self {
        Self { shape: Shape::Box { center: PixelPos { x: cx, y: cy }, w, h, theta_deg }, ..Default::default() }
    }
}

// ----------------------------------------------------------------- I/O --

#[derive(Debug)]
pub enum RegError {
    Io(io::Error),
    Parse(String),
}
impl From<io::Error> for RegError {
    fn from(e: io::Error) -> Self { RegError::Io(e) }
}
impl std::fmt::Display for RegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegError::Io(e) => write!(f, "io: {e}"),
            RegError::Parse(s) => write!(f, "parse: {s}"),
        }
    }
}
impl std::error::Error for RegError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordSys {
    Image,     // 1-based pixel coords (FITS)
    Physical,  // treat as image
    Fk5,       // RA/Dec degrees, ICRS-equivalent for our precision
    Icrs,
    Galactic,  // l/b degrees → ICRS → pixel
}

/// Parse a DS9 `.reg` file with no WCS. Sky-coord regions are kept verbatim
/// as `image`, so they will plot in the wrong place — see
/// [`parse_reg_with_wcs`] for the WCS-aware version.
pub fn parse_reg(text: &str) -> Result<Vec<Marker>, RegError> {
    parse_reg_inner(text, None)
}

/// Parse a DS9 `.reg` file. Sky-coord regions (`fk5`, `icrs`, `galactic`)
/// are projected to image pixels using `wcs`. Lines that need the WCS but
/// for which the projection fails are silently dropped.
pub fn parse_reg_with_wcs(text: &str, wcs: &Wcs) -> Result<Vec<Marker>, RegError> {
    parse_reg_inner(text, Some(wcs))
}

fn parse_reg_inner(text: &str, wcs: Option<&Wcs>) -> Result<Vec<Marker>, RegError> {
    let mut out = Vec::new();
    let mut sys = CoordSys::Image;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let body = line.split('#').next().unwrap_or(line).trim();
        if body.is_empty() { continue; }
        let lc = body.to_lowercase();
        // bare coord-system tokens switch the parser's mode
        match lc.as_str() {
            "image"    | "physical"   => { sys = match lc.as_str() {
                "image" => CoordSys::Image, _ => CoordSys::Physical,
            }; continue; }
            "fk5" | "fk4" | "j2000" | "b1950" => { sys = CoordSys::Fk5;      continue; }
            "icrs"     => { sys = CoordSys::Icrs;     continue; }
            "galactic" => { sys = CoordSys::Galactic; continue; }
            _ => {}
        }
        if lc.starts_with("global") { continue; }

        // a shape line may itself be prefixed with a coord system, e.g.
        // `fk5; circle(...)`. Detect and apply for that one line only.
        let (line_sys, body_only) = strip_inline_sys(body, sys);
        if let Some(m) = parse_shape_line(body_only, line_sys, wcs) {
            out.push(m);
        }
    }
    Ok(out)
}

fn strip_inline_sys(body: &str, default: CoordSys) -> (CoordSys, &str) {
    if let Some(idx) = body.find(';') {
        let (head, rest) = body.split_at(idx);
        let rest = rest.trim_start_matches(';').trim();
        let s = head.trim().to_lowercase();
        let sys = match s.as_str() {
            "image"    => CoordSys::Image,
            "physical" => CoordSys::Physical,
            "fk5" | "fk4" | "j2000" | "b1950" => CoordSys::Fk5,
            "icrs"     => CoordSys::Icrs,
            "galactic" => CoordSys::Galactic,
            _ => return (default, body),
        };
        (sys, rest)
    } else {
        (default, body)
    }
}

fn parse_shape_line(body: &str, sys: CoordSys, wcs: Option<&Wcs>) -> Option<Marker> {
    let body = body.trim_start_matches('-').trim();
    let (kind, args_str) = body.split_once('(')?;
    let kind = kind.trim().to_lowercase();
    let args_str = args_str.trim_end().trim_end_matches(')');
    let raw_args: Vec<&str> = args_str.split(',').map(str::trim).collect();

    // For sky systems we expect the first two args to be sexagesimal/decimal
    // coords and the remainder to be sizes (with unit suffixes like `"` or `'`).
    let needs_wcs = matches!(sys, CoordSys::Fk5 | CoordSys::Icrs | CoordSys::Galactic);
    let (cx_pix, cy_pix, scale_pix_per_arcsec): (f64, f64, f64) = if needs_wcs {
        let wcs = wcs?;
        let lon = parse_lon(raw_args.first()?)?;   // RA / l in degrees
        let lat = parse_lat(raw_args.get(1)?)?;    // Dec / b in degrees
        let (ra, dec) = match sys {
            CoordSys::Galactic => ds9_fits::galactic_to_icrs(lon, lat),
            _ => (lon, lat),
        };
        let (px, py) = wcs.world_to_pix(ra, dec)?;
        // scale: 1 arcsec on the sky → ? pixels (use CD matrix scale at center)
        let det = (wcs.cd11 * wcs.cd22 - wcs.cd12 * wcs.cd21).abs();
        let pix_per_deg = if det > 0.0 { 1.0 / det.sqrt() } else { 1.0 };
        (px, py, pix_per_deg / 3600.0)
    } else {
        let x = parse_pix(raw_args.first()?)?;
        let y = parse_pix(raw_args.get(1)?)?;
        (x, y, 1.0)
    };

    // Convert remaining numeric args. For sky systems, sizes are in
    // arcsec (default) or arcmin (suffix `'`) or degrees (suffix `d`/none on
    // angles). For pixel systems they're plain pixels.
    let mut tail: Vec<f64> = Vec::with_capacity(raw_args.len().saturating_sub(2));
    for s in raw_args.iter().skip(2) {
        let v = if needs_wcs && !is_pure_angle(kind.as_str(), tail.len()) {
            parse_size_to_pix(s, scale_pix_per_arcsec)?
        } else {
            parse_coord(s)?
        };
        tail.push(v);
    }

    let center = PixelPos { x: cx_pix, y: cy_pix };

    let m = match (kind.as_str(), tail.as_slice()) {
        ("circle",  [r])              => Marker::circle(center.x, center.y, *r),
        ("box",     [w, h, th])       => Marker::rbox(center.x, center.y, *w, *h, *th),
        ("box",     [w, h])           => Marker::rbox(center.x, center.y, *w, *h, 0.0),
        ("ellipse", [a, b, th])       => Marker {
            shape: Shape::Ellipse { center, a: *a, b: *b, theta_deg: *th },
            ..Default::default()
        },
        ("ellipse", [a, b])           => Marker {
            shape: Shape::Ellipse { center, a: *a, b: *b, theta_deg: 0.0 },
            ..Default::default()
        },
        ("annulus", [ri, ro])         => Marker {
            shape: Shape::Annulus { center, r_inner: *ri, r_outer: *ro },
            ..Default::default()
        },
        ("point",   [])               => Marker {
            shape: Shape::Point { center },
            ..Default::default()
        },
        // For polyline-y shapes in sky systems we'd need to project each
        // vertex separately — easier to re-route to the polygon parser below.
        ("line",    _) | ("polygon", _) if needs_wcs => {
            let pts = project_vertex_list(raw_args.as_slice(), sys, wcs?)?;
            if kind == "line" && pts.len() == 2 {
                Marker { shape: Shape::Line { from: pts[0], to: pts[1] }, ..Default::default() }
            } else if kind == "polygon" && pts.len() >= 3 {
                Marker { shape: Shape::Polygon { points: pts }, ..Default::default() }
            } else {
                return None;
            }
        }
        ("line",    [x2, y2])         => Marker {
            shape: Shape::Line {
                from: center,
                to:   PixelPos { x: *x2, y: *y2 },
            },
            ..Default::default()
        },
        ("polygon", pts) if pts.len() >= 4 && pts.len() % 2 == 0 => {
            // pixel-system polygon: cx,cy were already eaten as the first vertex
            let mut all = vec![center];
            for c in pts.chunks_exact(2) {
                all.push(PixelPos { x: c[0], y: c[1] });
            }
            Marker { shape: Shape::Polygon { points: all }, ..Default::default() }
        }
        _ => return None,
    };
    Some(m)
}

/// True if the n-th tail arg of a shape is a plain angle in degrees (e.g. the
/// rotation of a box) rather than a sky size.
fn is_pure_angle(kind: &str, tail_idx: usize) -> bool {
    matches!((kind, tail_idx),
        ("box",     2)  // box(x, y, w, h, theta)
        | ("ellipse", 2)
    )
}

/// Project a list of (lon, lat, lon, lat, …) sky vertices to pixel space.
fn project_vertex_list(args: &[&str], sys: CoordSys, wcs: &Wcs) -> Option<Vec<PixelPos>> {
    if args.len() < 2 || args.len() % 2 != 0 { return None; }
    let mut out = Vec::with_capacity(args.len() / 2);
    for chunk in args.chunks_exact(2) {
        let lon = parse_lon(chunk[0])?;
        let lat = parse_lat(chunk[1])?;
        let (ra, dec) = match sys {
            CoordSys::Galactic => ds9_fits::galactic_to_icrs(lon, lat),
            _ => (lon, lat),
        };
        let (px, py) = wcs.world_to_pix(ra, dec)?;
        out.push(PixelPos { x: px, y: py });
    }
    Some(out)
}

/// RA / l: accept `12:34:56.7` (sexagesimal hours for RA, degrees for l) or
/// decimal degrees with optional `d`/`h` suffix. We disambiguate using a
/// `:` sign or an `h` suffix — DS9's heuristic is "if it has `h:`, it's hours".
fn parse_lon(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.contains(':') {
        // assume hours for RA-style with three colon-separated parts
        let parts: Vec<&str> = t.split(':').collect();
        let h: f64 = parts.first()?.parse().ok()?;
        let m: f64 = parts.get(1).and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let sec: f64 = parts.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let sign = if h.is_sign_negative() { -1.0 } else { 1.0 };
        Some(sign * (h.abs() + m / 60.0 + sec / 3600.0) * 15.0)
    } else if let Some(stripped) = t.strip_suffix('h') {
        stripped.parse::<f64>().ok().map(|v| v * 15.0)
    } else {
        t.trim_end_matches(|c: char| c == 'd' || c.is_whitespace()).parse().ok()
    }
}

/// Dec / b: `+12:34:56` sexagesimal degrees, or decimal degrees.
fn parse_lat(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.contains(':') {
        let parts: Vec<&str> = t.split(':').collect();
        let head = parts.first()?;
        let sign = if head.starts_with('-') { -1.0 } else { 1.0 };
        let d: f64 = head.trim_start_matches(['+', '-']).parse().ok()?;
        let m: f64 = parts.get(1).and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let sec: f64 = parts.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.0);
        Some(sign * (d + m / 60.0 + sec / 3600.0))
    } else {
        t.trim_end_matches(|c: char| c == 'd' || c.is_whitespace()).parse().ok()
    }
}

/// Pixel-system coordinate parse with permissive trailing units.
fn parse_pix(s: &str) -> Option<f64> { parse_coord(s) }

/// Sky-system size: arcseconds by default; `'` = arcmin; `d` = degrees.
fn parse_size_to_pix(s: &str, pix_per_arcsec: f64) -> Option<f64> {
    let t = s.trim();
    if let Some(stripped) = t.strip_suffix("'") {
        stripped.parse::<f64>().ok().map(|v| v * 60.0 * pix_per_arcsec)
    } else if let Some(stripped) = t.strip_suffix("d") {
        stripped.parse::<f64>().ok().map(|v| v * 3600.0 * pix_per_arcsec)
    } else if let Some(stripped) = t.strip_suffix("\"") {
        stripped.parse::<f64>().ok().map(|v| v * pix_per_arcsec)
    } else {
        // bare number — assume arcsec (DS9 default for sky systems)
        t.parse::<f64>().ok().map(|v| v * pix_per_arcsec)
    }
}

/// Permissive numeric parse: strips trailing units like `"`, `'`, `p`, `i`.
fn parse_coord(s: &str) -> Option<f64> {
    let t = s.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E');
    t.parse::<f64>().ok()
}

/// Write a `.reg` file. Uses the `image` coordinate system.
pub fn write_reg<P: AsRef<Path>>(path: P, markers: &[Marker]) -> Result<(), RegError> {
    let mut s = String::new();
    s.push_str("# Region file format: DS9 ds9-rust\n");
    s.push_str("global color=green width=1 select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n");
    s.push_str("image\n");
    for m in markers {
        s.push_str(&format_shape(&m.shape));
        s.push('\n');
    }
    let mut f = fs::File::create(path)?;
    f.write_all(s.as_bytes())?;
    Ok(())
}

fn format_shape(sh: &Shape) -> String {
    match sh {
        Shape::Circle  { center: c, r }              => format!("circle({:.4},{:.4},{:.4})", c.x, c.y, r),
        Shape::Ellipse { center: c, a, b, theta_deg } => format!("ellipse({:.4},{:.4},{:.4},{:.4},{:.4})", c.x, c.y, a, b, theta_deg),
        Shape::Box     { center: c, w, h, theta_deg } => format!("box({:.4},{:.4},{:.4},{:.4},{:.4})", c.x, c.y, w, h, theta_deg),
        Shape::Annulus { center: c, r_inner, r_outer } => format!("annulus({:.4},{:.4},{:.4},{:.4})", c.x, c.y, r_inner, r_outer),
        Shape::Polygon { points }                    => {
            let inner: Vec<String> = points.iter()
                .flat_map(|p| [format!("{:.4}", p.x), format!("{:.4}", p.y)])
                .collect();
            format!("polygon({})", inner.join(","))
        }
        Shape::Point   { center: c }                 => format!("point({:.4},{:.4})", c.x, c.y),
        Shape::Line    { from: a, to: b }            => format!("line({:.4},{:.4},{:.4},{:.4})", a.x, a.y, b.x, b.y),
        Shape::Compass { center: c, len }            => format!("# compass({:.4},{:.4},{:.4})", c.x, c.y, len),
        Shape::Text    { center: c, body }           => format!("# text({:.4},{:.4}) text={{{}}}", c.x, c.y, body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let txt = "# comment\nglobal color=red\nimage\ncircle(100,200,15)\nbox(50,60,10,20,30)\n";
        let m = parse_reg(txt).unwrap();
        assert_eq!(m.len(), 2);
        assert!(matches!(m[0].shape, Shape::Circle { .. }));
        assert!(matches!(m[1].shape, Shape::Box { .. }));
    }

    #[test]
    fn parse_fk5_with_wcs() {
        // small TAN WCS centered at (RA=10°, Dec=20°), 1″/pix CD matrix
        let wcs = ds9_fits::Wcs {
            ctype1: "RA---TAN".into(), ctype2: "DEC--TAN".into(),
            radesys: "ICRS".into(),
            crpix1: 100.0, crpix2: 100.0,
            crval1: 10.0,  crval2: 20.0,
            cd11: -1.0/3600.0, cd12: 0.0,
            cd21:  0.0,        cd22: 1.0/3600.0,
        };
        let txt = "fk5\ncircle(10.001, 20.0, 5\")\n";
        let m = parse_reg_with_wcs(txt, &wcs).unwrap();
        assert_eq!(m.len(), 1);
        if let Shape::Circle { center, r } = &m[0].shape {
            // 0.001° east of CRVAL1 ≈ 3.6″ ÷ 1″/pix east of CRPIX1 = ~3.6 pixels.
            // CD11 is negative (RA increases west), so center.x ≈ 100 - 3.4
            assert!((center.x - 96.6).abs() < 1.0,
                "center.x={} (expected ~96.6)", center.x);
            assert!((center.y - 100.0).abs() < 0.5);
            // r is 5″ in pixels (1″/pix) ≈ 5
            assert!((r - 5.0).abs() < 0.2, "r={r} (expected ~5)");
        } else {
            panic!("expected circle, got {:?}", m[0].shape);
        }
    }

    #[test]
    fn parse_sexagesimal_dec() {
        // Same WCS as above
        let wcs = ds9_fits::Wcs {
            ctype1: "RA---TAN".into(), ctype2: "DEC--TAN".into(),
            radesys: "ICRS".into(),
            crpix1: 100.0, crpix2: 100.0,
            crval1: 10.0,  crval2: 20.0,
            cd11: -1.0/3600.0, cd12: 0.0,
            cd21:  0.0,        cd22: 1.0/3600.0,
        };
        // RA = 00:40:00.000h = 10°, Dec = +20:00:00 → exact center
        let txt = "fk5\ncircle(00:40:00.000, +20:00:00, 1\")\n";
        let m = parse_reg_with_wcs(txt, &wcs).unwrap();
        assert_eq!(m.len(), 1);
        if let Shape::Circle { center, .. } = &m[0].shape {
            assert!((center.x - 100.0).abs() < 0.1);
            assert!((center.y - 100.0).abs() < 0.1);
        }
    }

    #[test]
    fn roundtrip() {
        let ms = vec![
            Marker::circle(10.0, 20.0, 5.0),
            Marker::rbox(30.0, 40.0, 4.0, 6.0, 12.0),
        ];
        let dir = std::env::temp_dir();
        let p = dir.join("ds9_rust_test.reg");
        write_reg(&p, &ms).unwrap();
        let back = parse_reg(&fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(back.len(), 2);
    }
}
