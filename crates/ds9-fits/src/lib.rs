//! Minimal FITS reader wrapping `fitsrs`. Returns the primary HDU's 2-D image
//! as a flat `Vec<f32>` plus dimensions, finite min/max, and (optionally) a
//! TAN-projection WCS parsed from the header.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use fitsrs::{Fits, HDU, Pixels};
use fitsrs::card::Value;
use fitsrs::hdu::data::bintable::data::BinaryTableData;
use fitsrs::hdu::data::bintable::tile_compressed::pixels::Pixels as TilePixels;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FitsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("fits: {0}")]
    Parse(String),
    #[error("file contains no 2-D image HDU")]
    NoImageHdu,
}

pub struct FitsImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub min: f32,
    pub max: f32,
    pub wcs: Option<Wcs>,
}

// ----------------------------------------------------------------- WCS --

/// Minimal WCS — gnomonic ("TAN") projection only. Linear and SIN/CAR are
/// pretty close to TAN at small angular scales so we use TAN as a fallback
/// even when CTYPE says LINEAR; if the image has no projection at all, the
/// readout falls back to "wcs: ——".
#[derive(Debug, Clone)]
pub struct Wcs {
    pub ctype1: String,
    pub ctype2: String,
    pub radesys: String,
    pub crpix1: f64,
    pub crpix2: f64,
    pub crval1: f64,   // deg
    pub crval2: f64,   // deg
    pub cd11: f64,     // deg / pix
    pub cd12: f64,
    pub cd21: f64,
    pub cd22: f64,
}

impl Wcs {
    /// Convert a 1-based FITS pixel position to world (RA, Dec) in degrees,
    /// using the inverse gnomonic projection.
    pub fn pix_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.crpix1;
        let dy = y - self.crpix2;
        // intermediate world coords (deg)
        let xi  = self.cd11 * dx + self.cd12 * dy;
        let eta = self.cd21 * dx + self.cd22 * dy;
        // TAN deprojection
        let xi_r  = xi.to_radians();
        let eta_r = eta.to_radians();
        let ra0   = self.crval1.to_radians();
        let dec0  = self.crval2.to_radians();
        let rho = (xi_r * xi_r + eta_r * eta_r).sqrt();
        if rho < 1e-15 {
            return (self.crval1, self.crval2);
        }
        let c = rho.atan();
        let sin_c = c.sin();
        let cos_c = c.cos();
        let dec = (cos_c * dec0.sin() + eta_r * sin_c * dec0.cos() / rho).asin();
        let ra  = ra0 + (xi_r * sin_c)
            .atan2(rho * dec0.cos() * cos_c - eta_r * dec0.sin() * sin_c);
        let mut ra_d = ra.to_degrees();
        let dec_d = dec.to_degrees();
        // wrap RA to [0, 360)
        ra_d = ra_d.rem_euclid(360.0);
        (ra_d, dec_d)
    }

    /// Inverse gnomonic projection: convert world (RA, Dec) in degrees to a
    /// 1-based FITS pixel position. Returns `None` if the CD matrix is singular
    /// (or the point is on the opposite hemisphere with no valid intermediate).
    pub fn world_to_pix(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        let ra   = ra_deg.to_radians();
        let dec  = dec_deg.to_radians();
        let ra0  = self.crval1.to_radians();
        let dec0 = self.crval2.to_radians();
        let cos_dec = dec.cos();
        let sin_dec = dec.sin();
        let cos_dec0 = dec0.cos();
        let sin_dec0 = dec0.sin();
        let dra = ra - ra0;
        // standard TAN forward projection
        let denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * dra.cos();
        if denom <= 0.0 { return None; }
        let xi  = (cos_dec * dra.sin() / denom).to_degrees();
        let eta = ((cos_dec0 * sin_dec - sin_dec0 * cos_dec * dra.cos()) / denom).to_degrees();
        // invert the CD matrix
        let det = self.cd11 * self.cd22 - self.cd12 * self.cd21;
        if det.abs() < 1e-30 { return None; }
        let inv = 1.0 / det;
        let dx =  inv * ( self.cd22 * xi - self.cd12 * eta);
        let dy =  inv * (-self.cd21 * xi + self.cd11 * eta);
        Some((dx + self.crpix1, dy + self.crpix2))
    }
}

/// Approximate galactic-to-ICRS rotation (J2000 epoch, IAU 1958 pole).
/// Good to a few arcseconds — enough for region overlays.
pub fn galactic_to_icrs(l_deg: f64, b_deg: f64) -> (f64, f64) {
    // J2000 NGP and l-of-NCP per Liu et al. 2011 / IAU 1958
    let ngp_ra  = 192.859508_f64.to_radians();
    let ngp_dec =  27.128336_f64.to_radians();
    let lncp    = 122.932000_f64.to_radians();
    let l = l_deg.to_radians();
    let b = b_deg.to_radians();
    let sin_dec = b.sin() * ngp_dec.sin() + b.cos() * ngp_dec.cos() * (lncp - l).cos();
    let dec = sin_dec.asin();
    let sin_ra_off = b.cos() * (lncp - l).sin() / dec.cos();
    let cos_ra_off = (b.sin() - ngp_dec.sin() * sin_dec) / (ngp_dec.cos() * dec.cos());
    let ra_off = sin_ra_off.atan2(cos_ra_off);
    let ra = (ngp_ra + ra_off).rem_euclid(std::f64::consts::TAU);
    (ra.to_degrees(), dec.to_degrees())
}

/// Format (ra_deg, dec_deg) as DS9-style sexagesimal: `12:34:56.78  +12:34:56.7`.
pub fn format_sexagesimal(ra_deg: f64, dec_deg: f64) -> String {
    let ra_h = ra_deg / 15.0;
    let rh = ra_h.floor() as i32;
    let rm_full = (ra_h - rh as f64) * 60.0;
    let rm = rm_full.floor() as i32;
    let rs = (rm_full - rm as f64) * 60.0;

    let sign = if dec_deg.is_sign_negative() { '-' } else { '+' };
    let dec_a = dec_deg.abs();
    let dd = dec_a.floor() as i32;
    let dm_full = (dec_a - dd as f64) * 60.0;
    let dm = dm_full.floor() as i32;
    let ds = (dm_full - dm as f64) * 60.0;
    format!("{rh:02}:{rm:02}:{rs:05.2}  {sign}{dd:02}:{dm:02}:{ds:04.1}")
}

fn get_f64<X>(h: &fitsrs::hdu::header::Header<X>, key: &str) -> Option<f64>
where X: fitsrs::hdu::header::extension::Xtension + std::fmt::Debug {
    match h.get(key)? {
        Value::Float   { value, .. } => Some(*value),
        Value::Integer { value, .. } => Some(*value as f64),
        _ => None,
    }
}

fn get_str<X>(h: &fitsrs::hdu::header::Header<X>, key: &str) -> Option<String>
where X: fitsrs::hdu::header::extension::Xtension + std::fmt::Debug {
    match h.get(key)? {
        Value::String { value, .. } => Some(value.trim().to_string()),
        _ => None,
    }
}

fn try_parse_wcs<X>(h: &fitsrs::hdu::header::Header<X>) -> Option<Wcs>
where X: fitsrs::hdu::header::extension::Xtension + std::fmt::Debug {
    let crpix1 = get_f64(h, "CRPIX1")?;
    let crpix2 = get_f64(h, "CRPIX2")?;
    let crval1 = get_f64(h, "CRVAL1")?;
    let crval2 = get_f64(h, "CRVAL2")?;

    // CD matrix preferred; fall back to PC * CDELT; finally CROTA2 + CDELT.
    let (cd11, cd12, cd21, cd22) = if let (Some(a), Some(b), Some(c), Some(d)) = (
        get_f64(h, "CD1_1"), get_f64(h, "CD1_2"),
        get_f64(h, "CD2_1"), get_f64(h, "CD2_2"),
    ) {
        (a, b, c, d)
    } else {
        let cdelt1 = get_f64(h, "CDELT1").unwrap_or(1.0);
        let cdelt2 = get_f64(h, "CDELT2").unwrap_or(1.0);
        let pc11 = get_f64(h, "PC1_1").unwrap_or(1.0);
        let pc12 = get_f64(h, "PC1_2").unwrap_or(0.0);
        let pc21 = get_f64(h, "PC2_1").unwrap_or(0.0);
        let pc22 = get_f64(h, "PC2_2").unwrap_or(1.0);
        if (pc12 == 0.0 && pc21 == 0.0 && pc11 == 1.0 && pc22 == 1.0)
            && get_f64(h, "CROTA2").is_some()
        {
            let rot = get_f64(h, "CROTA2").unwrap().to_radians();
            let (cr, sr) = (rot.cos(), rot.sin());
            (cdelt1 * cr,  -cdelt2 * sr,
             cdelt1 * sr,   cdelt2 * cr)
        } else {
            (pc11 * cdelt1, pc12 * cdelt1,
             pc21 * cdelt2, pc22 * cdelt2)
        }
    };

    Some(Wcs {
        ctype1: get_str(h, "CTYPE1").unwrap_or_else(|| "PIXEL".into()),
        ctype2: get_str(h, "CTYPE2").unwrap_or_else(|| "PIXEL".into()),
        radesys: get_str(h, "RADESYS").unwrap_or_else(|| "ICRS".into()),
        crpix1, crpix2, crval1, crval2, cd11, cd12, cd21, cd22,
    })
}

// ----------------------------------------------------------------- pixels --

fn pixels_to_f32<R: std::io::Read + std::io::BufRead + std::io::Seek>(
    image: fitsrs::ImageData<R>, n: usize,
) -> Vec<f32> {
    let mut data: Vec<f32> = match image.pixels() {
        Pixels::F32(it) => it.take(n).collect(),
        Pixels::F64(it) => it.take(n).map(|v| v as f32).collect(),
        Pixels::I64(it) => it.take(n).map(|v| v as f32).collect(),
        Pixels::I32(it) => it.take(n).map(|v| v as f32).collect(),
        Pixels::I16(it) => it.take(n).map(|v| v as f32).collect(),
        Pixels::U8(it)  => it.take(n).map(|v| v as f32).collect(),
    };
    if data.len() < n {
        data.resize(n, f32::NAN);
    }
    data
}

fn finite_minmax(data: &[f32]) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        if v.is_finite() {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
    }
    if !lo.is_finite() { lo = 0.0; }
    if !hi.is_finite() { hi = 1.0; }
    (lo, hi)
}

// -------------------------------------------------- tile-compressed images --

/// De-tile a pixel stream that arrives in tile-major order into a flat
/// row-major image. Handles ragged final rows / cols where the image dim is
/// not a multiple of the tile dim.
fn detile<I: Iterator<Item = f32>>(
    iter: I,
    img_w: usize, img_h: usize,
    tile_w: usize, tile_h: usize,
) -> Vec<f32> {
    let n_tiles_x = (img_w + tile_w - 1) / tile_w.max(1);
    let n_tiles_y = (img_h + tile_h - 1) / tile_h.max(1);
    let mut out = vec![f32::NAN; img_w * img_h];
    let mut tile_idx = 0_usize;
    let mut local_idx = 0_usize;
    let n_tiles = n_tiles_x * n_tiles_y;

    let cur_tile_dims = |t: usize| -> (usize, usize, usize, usize) {
        let tx = t % n_tiles_x;
        let ty = t / n_tiles_x;
        let tw = if tx == n_tiles_x - 1 { img_w - tx * tile_w } else { tile_w };
        let th = if ty == n_tiles_y - 1 { img_h - ty * tile_h } else { tile_h };
        (tx, ty, tw, th)
    };

    for v in iter {
        if tile_idx >= n_tiles { break; }
        let (tx, ty, tw, th) = cur_tile_dims(tile_idx);
        let lx = local_idx % tw.max(1);
        let ly = local_idx / tw.max(1);
        let ix = tx * tile_w + lx;
        let iy = ty * tile_h + ly;
        if ix < img_w && iy < img_h {
            out[iy * img_w + ix] = v;
        }
        local_idx += 1;
        if local_idx >= tw * th {
            local_idx = 0;
            tile_idx += 1;
        }
    }
    out
}

fn load_tile_compressed<X, R>(
    header: &fitsrs::hdu::header::Header<X>,
    data: BinaryTableData<R>,
) -> Option<FitsImage>
where
    X: fitsrs::hdu::header::extension::Xtension + std::fmt::Debug,
    R: std::io::Read + std::io::Seek + std::fmt::Debug,
{
    let znaxis1 = get_f64(header, "ZNAXIS1")? as usize;
    let znaxis2 = get_f64(header, "ZNAXIS2")? as usize;
    if znaxis1 == 0 || znaxis2 == 0 { return None; }
    let ztile1 = get_f64(header, "ZTILE1").map(|v| v as usize).unwrap_or(znaxis1);
    let ztile2 = get_f64(header, "ZTILE2").map(|v| v as usize).unwrap_or(1);
    let bscale = get_f64(header, "BSCALE").unwrap_or(1.0) as f32;
    let bzero  = get_f64(header, "BZERO").unwrap_or(0.0)  as f32;
    let blank  = get_f64(header, "ZBLANK").or_else(|| get_f64(header, "BLANK"));

    let raster: Vec<f32> = match data {
        BinaryTableData::TileCompressed(p) => match p {
            TilePixels::F32(it) => detile(it, znaxis1, znaxis2, ztile1, ztile2),
            TilePixels::F64(it) => detile(it.map(|v| v as f32), znaxis1, znaxis2, ztile1, ztile2),
            TilePixels::U8(it)  => detile(it.map(|v| v as f32 * bscale + bzero), znaxis1, znaxis2, ztile1, ztile2),
            TilePixels::I16(it) => detile(it.map(|v| {
                if blank.map(|b| b as i64 == v as i64).unwrap_or(false) { f32::NAN }
                else { v as f32 * bscale + bzero }
            }), znaxis1, znaxis2, ztile1, ztile2),
            TilePixels::I32(it) => detile(it.map(|v| {
                if blank.map(|b| b as i64 == v as i64).unwrap_or(false) { f32::NAN }
                else { v as f32 * bscale + bzero }
            }), znaxis1, znaxis2, ztile1, ztile2),
        },
        BinaryTableData::Table(_) => return None,
    };
    let (min, max) = finite_minmax(&raster);
    let wcs = try_parse_wcs(header);
    Some(FitsImage { width: znaxis1, height: znaxis2, data: raster, min, max, wcs })
}

// ------------------------------------------------------------------ load --

/// One row of [`enumerate_hdus`] output: an HDU's index in the file plus its
/// kind, EXTNAME (or "PRIMARY"/"XImage"/"XTable" if absent), and 2-D image
/// dimensions when applicable.
#[derive(Debug, Clone)]
pub struct HduInfo {
    /// 0-based HDU index in the file.
    pub idx: usize,
    /// Short label: "PRIMARY", "IMAGE", "TILE-COMPRESSED", "TABLE".
    pub kind: &'static str,
    /// EXTNAME header value, falling back to `kind` if missing.
    pub name: String,
    /// 2-D image dimensions (None for tables that aren't tile-compressed).
    pub dims: Option<(usize, usize)>,
}

fn extname<X>(h: &fitsrs::hdu::header::Header<X>) -> Option<String>
where X: fitsrs::hdu::header::extension::Xtension + std::fmt::Debug {
    get_str(h, "EXTNAME")
}

/// Enumerate every HDU in the file along with its kind / dimensions. Useful
/// for an "HDU navigator" — the user picks one and we re-load with [`load_hdu`].
pub fn enumerate_hdus<P: AsRef<Path>>(path: P) -> Result<Vec<HduInfo>, FitsError> {
    let f = File::open(path)?;
    let mut hdus = Fits::from_reader(BufReader::new(f));
    let mut out = Vec::new();
    let mut idx = 0_usize;
    while let Some(next) = hdus.next() {
        let hdu = next.map_err(|e| FitsError::Parse(format!("{e:?}")))?;
        let info = match &hdu {
            HDU::Primary(p) => {
                let n = p.get_header().get_xtension().get_naxis().to_vec();
                let dims = if n.len() >= 2 && n[0] > 0 && n[1] > 0 {
                    Some((n[0] as usize, n[1] as usize))
                } else { None };
                HduInfo {
                    idx, kind: "PRIMARY",
                    name: extname(p.get_header()).unwrap_or_else(|| "PRIMARY".into()),
                    dims,
                }
            }
            HDU::XImage(x) => {
                let n = x.get_header().get_xtension().get_naxis().to_vec();
                let dims = if n.len() >= 2 && n[0] > 0 && n[1] > 0 {
                    Some((n[0] as usize, n[1] as usize))
                } else { None };
                HduInfo {
                    idx, kind: "IMAGE",
                    name: extname(x.get_header()).unwrap_or_else(|| format!("IMAGE#{idx}")),
                    dims,
                }
            }
            HDU::XBinaryTable(b) => {
                let h = b.get_header();
                let tile = h.get_xtension().get_z_image().is_some();
                let dims = if tile {
                    let w = get_f64(h, "ZNAXIS1").map(|v| v as usize);
                    let hh = get_f64(h, "ZNAXIS2").map(|v| v as usize);
                    match (w, hh) { (Some(a), Some(b)) => Some((a, b)), _ => None }
                } else { None };
                HduInfo {
                    idx, kind: if tile { "TILE-COMPRESSED" } else { "TABLE" },
                    name: extname(h).unwrap_or_else(|| {
                        if tile { format!("TILE#{idx}") } else { format!("TABLE#{idx}") }
                    }),
                    dims,
                }
            }
            _ => HduInfo { idx, kind: "OTHER", name: format!("HDU#{idx}"), dims: None },
        };
        out.push(info);
        idx += 1;
    }
    Ok(out)
}

/// Load a specific HDU index. Falls back to `NoImageHdu` if the requested HDU
/// isn't a 2-D image / tile-compressed table.
pub fn load_hdu<P: AsRef<Path>>(path: P, target_idx: usize) -> Result<FitsImage, FitsError> {
    let f = File::open(path)?;
    let mut hdus = Fits::from_reader(BufReader::new(f));
    let mut idx = 0_usize;
    while let Some(next) = hdus.next() {
        let hdu = next.map_err(|e| FitsError::Parse(format!("{e:?}")))?;
        if idx != target_idx { idx += 1; continue; }
        match hdu {
            HDU::Primary(p) => {
                let naxis: Vec<u64> = p.get_header().get_xtension().get_naxis().to_vec();
                if naxis.len() < 2 || naxis[0] == 0 || naxis[1] == 0 {
                    return Err(FitsError::NoImageHdu);
                }
                let w = naxis[0] as usize;
                let h = naxis[1] as usize;
                let plane = w.saturating_mul(h);
                let wcs = try_parse_wcs(p.get_header());
                let data = pixels_to_f32(hdus.get_data(&p), plane);
                let (min, max) = finite_minmax(&data);
                return Ok(FitsImage { width: w, height: h, data, min, max, wcs });
            }
            HDU::XImage(x) => {
                let naxis: Vec<u64> = x.get_header().get_xtension().get_naxis().to_vec();
                if naxis.len() < 2 || naxis[0] == 0 || naxis[1] == 0 {
                    return Err(FitsError::NoImageHdu);
                }
                let w = naxis[0] as usize;
                let h = naxis[1] as usize;
                let plane = w.saturating_mul(h);
                let wcs = try_parse_wcs(x.get_header());
                let data = pixels_to_f32(hdus.get_data(&x), plane);
                let (min, max) = finite_minmax(&data);
                return Ok(FitsImage { width: w, height: h, data, min, max, wcs });
            }
            HDU::XBinaryTable(b) => {
                if b.get_header().get_xtension().get_z_image().is_none() {
                    return Err(FitsError::NoImageHdu);
                }
                let header_copy_needed = b.get_header().clone();
                let data = hdus.get_data(&b);
                if let Some(img) = load_tile_compressed(&header_copy_needed, data) {
                    return Ok(img);
                }
                return Err(FitsError::NoImageHdu);
            }
            _ => return Err(FitsError::NoImageHdu),
        }
    }
    Err(FitsError::NoImageHdu)
}

/// Load the first 2-D image we can find — primary HDU, an `XImage` extension,
/// or a tile-compressed `XBinaryTable` (DESI / Legacy Survey `.fz` files).
pub fn load<P: AsRef<Path>>(path: P) -> Result<FitsImage, FitsError> {
    let f = File::open(path)?;
    let mut hdus = Fits::from_reader(BufReader::new(f));

    while let Some(next) = hdus.next() {
        let hdu = next.map_err(|e| FitsError::Parse(format!("{e:?}")))?;
        match hdu {
            HDU::Primary(p) => {
                let naxis: Vec<u64> = p.get_header().get_xtension().get_naxis().to_vec();
                if naxis.len() < 2 || naxis[0] == 0 || naxis[1] == 0 {
                    continue;  // empty primary in MEF; keep walking
                }
                let w = naxis[0] as usize;
                let h = naxis[1] as usize;
                let plane = w.saturating_mul(h);
                let wcs = try_parse_wcs(p.get_header());
                let data = pixels_to_f32(hdus.get_data(&p), plane);
                let (min, max) = finite_minmax(&data);
                return Ok(FitsImage { width: w, height: h, data, min, max, wcs });
            }
            HDU::XImage(x) => {
                let naxis: Vec<u64> = x.get_header().get_xtension().get_naxis().to_vec();
                if naxis.len() < 2 || naxis[0] == 0 || naxis[1] == 0 {
                    continue;
                }
                let w = naxis[0] as usize;
                let h = naxis[1] as usize;
                let plane = w.saturating_mul(h);
                let wcs = try_parse_wcs(x.get_header());
                let data = pixels_to_f32(hdus.get_data(&x), plane);
                let (min, max) = finite_minmax(&data);
                return Ok(FitsImage { width: w, height: h, data, min, max, wcs });
            }
            HDU::XBinaryTable(b) => {
                if b.get_header().get_xtension().get_z_image().is_none() {
                    continue;  // ordinary table, not a tile-compressed image
                }
                // clone what we need from the header before borrowing `hdus` mutably
                let header_copy_needed = b.get_header().clone();
                let data = hdus.get_data(&b);
                if let Some(img) = load_tile_compressed(&header_copy_needed, data) {
                    return Ok(img);
                }
            }
            _ => continue,
        }
    }
    Err(FitsError::NoImageHdu)
}
