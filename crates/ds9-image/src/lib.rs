//! Pixel-pipeline: limits → stretch → 8-bit index → colormap LUT → RGBA.
//! Mirrors the `scale` / `zscale` and `cmap` modes of DS9.

use ds9_fits::FitsImage;

// -------------------------------------------------------------- colormaps --

/// DS9 colormap names. The visual vocabulary matches the original; the
/// numerical recipes are inspired by classic IRAF / DS9 LUTs without copying
/// the C source verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Grey,
    Red,
    Green,
    Blue,
    Heat,
    Cool,
    Rainbow,
    B,    // ds9 "b" — bright purple ramp
    Bb,   // black-body
    Sls,  // Stern Special Linear
    Hsv,
    A,    // ds9 "a"
}

impl Default for Colormap { fn default() -> Self { Colormap::Grey } }

impl Colormap {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "grey" | "gray"     => Colormap::Grey,
            "red"               => Colormap::Red,
            "green"             => Colormap::Green,
            "blue"              => Colormap::Blue,
            "heat"              => Colormap::Heat,
            "cool"              => Colormap::Cool,
            "rainbow"           => Colormap::Rainbow,
            "b"                 => Colormap::B,
            "bb"                => Colormap::Bb,
            "sls"               => Colormap::Sls,
            "hsv"               => Colormap::Hsv,
            "a"                 => Colormap::A,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Colormap::Grey    => "grey",
            Colormap::Red     => "red",
            Colormap::Green   => "green",
            Colormap::Blue    => "blue",
            Colormap::Heat    => "heat",
            Colormap::Cool    => "cool",
            Colormap::Rainbow => "rainbow",
            Colormap::B       => "b",
            Colormap::Bb      => "bb",
            Colormap::Sls     => "sls",
            Colormap::Hsv     => "hsv",
            Colormap::A       => "a",
        }
    }

    /// Build a 256-entry RGB LUT.
    pub fn lut(&self) -> [[u8; 3]; 256] {
        let mut lut = [[0u8; 3]; 256];
        for i in 0..256 {
            let t = i as f32 / 255.0;
            lut[i] = match self {
                Colormap::Grey    => { let g = (t * 255.0) as u8; [g, g, g] }
                Colormap::Red     => [(t * 255.0) as u8, 0, 0],
                Colormap::Green   => [0, (t * 255.0) as u8, 0],
                Colormap::Blue    => [0, 0, (t * 255.0) as u8],
                Colormap::Heat    => {
                    let r = (t * 3.0).clamp(0.0, 1.0);
                    let g = ((t - 0.33) * 3.0).clamp(0.0, 1.0);
                    let b = ((t - 0.66) * 3.0).clamp(0.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::Cool    => {
                    let r = t;
                    let g = 1.0 - t;
                    let b = 1.0;
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::Rainbow => {
                    let h = (1.0 - t) * 280.0;  // 280 deg → red, 0 → magenta-ish
                    let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::Hsv     => {
                    let (r, g, b) = hsv_to_rgb(t * 360.0, 1.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::B       => {
                    // dark navy → magenta → bright pink — DS9 "b" feel
                    let r = (t * 1.4 - 0.2).clamp(0.0, 1.0);
                    let g = ((t - 0.5) * 2.0).clamp(0.0, 1.0);
                    let b = (0.4 + t * 0.6).clamp(0.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::Bb      => {
                    // black-body radiator: black → red → orange → yellow → white
                    let r = (t * 1.5).clamp(0.0, 1.0);
                    let g = ((t - 0.4) * 1.6).clamp(0.0, 1.0);
                    let b = ((t - 0.75) * 4.0).clamp(0.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
                Colormap::Sls     => stern_special_linear(t),
                Colormap::A       => {
                    // bright warm: dim navy → orange → bright yellow
                    let r = (t * 1.2 + 0.05).clamp(0.0, 1.0);
                    let g = ((t - 0.35) * 1.7).clamp(0.0, 1.0);
                    let b = ((t - 0.65) * 2.5).clamp(0.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
                }
            };
        }
        lut
    }

    /// 9-stop CSS gradient string for the slint colorbar — sampled from the LUT.
    pub fn gradient_stops(&self) -> [(u8, u8, u8); 9] {
        let lut = self.lut();
        std::array::from_fn(|i| {
            let idx = i * 255 / 8;
            (lut[idx][0], lut[idx][1], lut[idx][2])
        })
    }

    /// 1 × 256 RGBA strip suitable for an `Image` element. Top of the strip
    /// corresponds to LUT[255] (high values) so it lines up with how DS9
    /// displays its vertical colorbar.
    pub fn rgba_strip(&self) -> Vec<u8> {
        let lut = self.lut();
        let mut out = Vec::with_capacity(256 * 4);
        for i in (0..256).rev() {
            let [r, g, b] = lut[i];
            out.extend_from_slice(&[r, g, b, 255]);
        }
        out
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = ((h % 360.0) + 360.0) % 360.0;
    let c = v * s;
    let hh = h / 60.0;
    let x = c * (1.0 - (hh % 2.0 - 1.0).abs());
    let (r, g, b) = match hh as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r + m, g + m, b + m)
}

/// Stern Special Linear: a multi-banded ramp DS9 inherits from IRAF.
fn stern_special_linear(t: f32) -> [u8; 3] {
    // simple 4-band approximation
    let r = if t < 0.0625 {
        t * 4.0 * 4.0
    } else if t < 0.5 {
        1.0 - (t - 0.0625) / 0.4375
    } else {
        (t - 0.5) * 2.0
    };
    let g = (t - 0.0).clamp(0.0, 1.0);
    let b = if t < 0.25 { t * 4.0 } else if t < 0.5 { 1.0 - (t - 0.25) * 4.0 } else { 0.0 };
    [(r.clamp(0.0,1.0) * 255.0) as u8, (g.clamp(0.0,1.0) * 255.0) as u8, (b.clamp(0.0,1.0) * 255.0) as u8]
}

// -------------------------------------------------------------- stretch --

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stretch {
    Linear,
    Log,
    Power(f32),
    Sqrt,
    Squared,
    Asinh,
    Sinh,
}

impl Default for Stretch {
    fn default() -> Self { Stretch::Linear }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub low: f32,
    pub high: f32,
}

impl Limits {
    pub fn minmax(img: &FitsImage) -> Self {
        Self { low: img.min, high: img.max }
    }

    /// Approximate IRAF zscale: drop the wildest 2.5 % on each tail.
    pub fn zscale(img: &FitsImage) -> Self {
        let mut s: Vec<f32> = img.data.iter().copied().filter(|v| v.is_finite()).collect();
        if s.is_empty() { return Self { low: 0.0, high: 1.0 }; }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        let lo_i = ((n as f32) * 0.025) as usize;
        let hi_i = (((n as f32) * 0.975) as usize).min(n - 1);
        let lo = s[lo_i];
        let hi = s[hi_i];
        if hi <= lo { Self { low: img.min, high: img.max } } else { Self { low: lo, high: hi } }
    }
}

#[inline]
fn apply(s: Stretch, n: f32) -> f32 {
    let n = n.clamp(0.0, 1.0);
    match s {
        Stretch::Linear     => n,
        Stretch::Log        => (n * 999.0 + 1.0).ln() / 1000f32.ln(),
        Stretch::Power(g)   => n.powf(g.max(0.05)),
        Stretch::Sqrt       => n.sqrt(),
        Stretch::Squared    => n * n,
        Stretch::Asinh      => (n * 10.0).asinh() / 10f32.asinh(),
        Stretch::Sinh       => n.sinh() / 1f32.sinh(),
    }
}

pub fn render_grayscale(img: &FitsImage, lim: Limits, stretch: Stretch) -> Vec<u8> {
    let span = (lim.high - lim.low).max(1e-30);
    let mut out = Vec::with_capacity(img.data.len());
    for &v in &img.data {
        let n = if v.is_finite() { (v - lim.low) / span } else { 0.0 };
        out.push((apply(stretch, n) * 255.0) as u8);
    }
    out
}

/// FITS pixel order is bottom-up; flip rows so the displayed image matches
/// what DS9 / SAO show by default. Applies stretch + colormap LUT in one pass.
pub fn render_rgba_flipped(
    img: &FitsImage,
    lim: Limits,
    stretch: Stretch,
    cmap: Colormap,
) -> Vec<u8> {
    let lut = cmap.lut();
    let span = (lim.high - lim.low).max(1e-30);
    let w = img.width;
    let h = img.height;
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        let src_row = (h - 1 - y) * w;
        let dst_row = y * w * 4;
        for x in 0..w {
            let v = img.data[src_row + x];
            let n = if v.is_finite() { (v - lim.low) / span } else { 0.0 };
            let idx = (apply(stretch, n) * 255.0).clamp(0.0, 255.0) as usize;
            let [r, g, b] = lut[idx];
            let i = dst_row + x * 4;
            out[i]     = r;
            out[i + 1] = g;
            out[i + 2] = b;
            out[i + 3] = 255;
        }
    }
    out
}
