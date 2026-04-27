//! Region/marker model + minimal DS9 `.reg` file I/O.
//! Supports the `image` coordinate system for the most common shapes:
//! circle / box / ellipse / point / polygon. Other coordinate systems
//! (fk5, galactic, …) are parsed positionally but treated as `image`
//! until WCS is wired in.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

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

/// Parse a DS9 `.reg` file. Returns markers in image-space (1-based, FITS
/// convention). Lines we don't recognize are silently skipped — DS9's grammar
/// is sprawling and we want graceful loading even when a file uses features
/// we don't implement yet.
pub fn parse_reg(text: &str) -> Result<Vec<Marker>, RegError> {
    let mut out = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        // strip trailing properties after "#"
        let body = line.split('#').next().unwrap_or(line).trim();
        if body.is_empty() { continue; }
        // skip lone coord-system tokens
        let lc = body.to_lowercase();
        if matches!(lc.as_str(),
            "image" | "fk5" | "fk4" | "icrs" | "galactic" | "ecliptic" |
            "physical" | "amplifier" | "detector"
        ) { continue; }
        if lc.starts_with("global") { continue; }

        if let Some(m) = parse_shape_line(body) {
            out.push(m);
        }
    }
    Ok(out)
}

fn parse_shape_line(body: &str) -> Option<Marker> {
    // body looks like:  circle(100, 200, 15)  or  -box(50,50,10,10,0)
    let body = body.trim_start_matches('-').trim();
    let (kind, args_str) = body.split_once('(')?;
    let kind = kind.trim().to_lowercase();
    let args_str = args_str.trim_end().trim_end_matches(')');
    let args: Vec<f64> = args_str.split(',')
        .map(|s| parse_coord(s.trim()))
        .collect::<Option<Vec<_>>>()?;
    let m = match (kind.as_str(), args.as_slice()) {
        ("circle",  [x, y, r])           => Marker::circle(*x, *y, *r),
        ("box",     [x, y, w, h, th])    => Marker::rbox(*x, *y, *w, *h, *th),
        ("box",     [x, y, w, h])        => Marker::rbox(*x, *y, *w, *h, 0.0),
        ("ellipse", [x, y, a, b, th])    => Marker {
            shape: Shape::Ellipse { center: PixelPos { x: *x, y: *y }, a: *a, b: *b, theta_deg: *th },
            ..Default::default()
        },
        ("ellipse", [x, y, a, b])        => Marker {
            shape: Shape::Ellipse { center: PixelPos { x: *x, y: *y }, a: *a, b: *b, theta_deg: 0.0 },
            ..Default::default()
        },
        ("annulus", [x, y, ri, ro])      => Marker {
            shape: Shape::Annulus { center: PixelPos { x: *x, y: *y }, r_inner: *ri, r_outer: *ro },
            ..Default::default()
        },
        ("point",   [x, y])              => Marker {
            shape: Shape::Point { center: PixelPos { x: *x, y: *y } },
            ..Default::default()
        },
        ("line",    [x1, y1, x2, y2])    => Marker {
            shape: Shape::Line {
                from: PixelPos { x: *x1, y: *y1 },
                to:   PixelPos { x: *x2, y: *y2 },
            },
            ..Default::default()
        },
        ("polygon", pts) if pts.len() >= 6 && pts.len() % 2 == 0 => {
            let points = pts.chunks_exact(2)
                .map(|c| PixelPos { x: c[0], y: c[1] })
                .collect();
            Marker { shape: Shape::Polygon { points }, ..Default::default() }
        }
        _ => return None,
    };
    Some(m)
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
