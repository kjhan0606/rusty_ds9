fn main() {
    let path = std::env::args().nth(1).expect("usage: wcs_check FITS");
    let img = ds9_fits::load(&path).expect("load");
    println!("size: {} × {}", img.width, img.height);
    println!("min/max: {} / {}", img.min, img.max);
    if let Some(wcs) = &img.wcs {
        println!("wcs:");
        println!("  CTYPE = {} / {}", wcs.ctype1, wcs.ctype2);
        println!("  RADESYS = {}", wcs.radesys);
        println!("  CRPIX = ({}, {})", wcs.crpix1, wcs.crpix2);
        println!("  CRVAL = ({}, {})", wcs.crval1, wcs.crval2);
        println!("  CD = [[{}, {}], [{}, {}]]", wcs.cd11, wcs.cd12, wcs.cd21, wcs.cd22);
        let (cx, cy) = (img.width as f64 / 2.0, img.height as f64 / 2.0);
        let (ra, dec) = wcs.pix_to_world(cx, cy);
        println!("  center pixel ({:.1}, {:.1}) → RA = {:.6}°, Dec = {:.6}°", cx, cy, ra, dec);
        println!("  sexagesimal: {}", ds9_fits::format_sexagesimal(ra, dec));
    } else {
        println!("no WCS in header");
    }
}
