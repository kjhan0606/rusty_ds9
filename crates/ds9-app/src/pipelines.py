#!/usr/bin/env python3
"""ds9-rust analysis pipelines (sidecar).

Dispatches on `sys.argv[1]` (task name). Argv layout:
    task fits_path outdir [--cx=X --cy=Y --r=R]

Tasks:
    starfind      -> out.tsv  (DAOStarFinder)
    psfphot       -> out.tsv  (PSFPhotometry, IntegratedGaussianPRF)
    crowded       -> out.tsv  (IterativePSFPhotometry)
    lsbg          -> out.tsv  (smooth + segmentation + low SB filter)
    deconv        -> out.fits (Richardson-Lucy)
    morphometry   -> stdout   (CAS / Gini / M20)
    sersic        -> stdout   (Sersic2D fit)
    bulgedisk     -> stdout   (bulge n=4 + disk n=1 fit)
    icl           -> stdout   (residual flux in annulus)
    photoz / sed / aimerge -> stdout (stubs that point at $OGFINDER_HOME)

Outputs to stdout: a one-line summary that ds9-rust shows in the status bar.
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _read_image(path):
    from astropy.io import fits
    hdul = fits.open(path, memmap=True)
    for h in hdul:
        if h.data is not None and getattr(h.data, "ndim", 0) >= 2:
            d = h.data
            while d.ndim > 2:
                d = d[0]
            arr = np.ascontiguousarray(d.astype(np.float32))
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.byteswap().newbyteorder()
            return arr, h.header
    raise RuntimeError("no 2D image HDU")


def _cutout(data, cx, cy, r):
    h, w = data.shape
    x0 = int(max(0, cx - r))
    x1 = int(min(w, cx + r))
    y0 = int(max(0, cy - r))
    y1 = int(min(h, cy + r))
    return data[y0:y1, x0:x1], (x0, y0)


def _safe_log10(x):
    return np.log10(max(float(x), 1e-12))


def _write_catalog_tsv(path, rows, cols):
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r) + "\n")


# --------------------------------------------------------------- tasks --

def task_starfind(args):
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    data, _ = _read_image(args.fits)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    df = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    s = df(data - med)
    out = os.path.join(args.outdir, "out.tsv")
    if s is None or len(s) == 0:
        _write_catalog_tsv(out, [], ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])
        print("starfind: 0 sources")
        return
    rows = []
    for i, r in enumerate(s, 1):
        mag = -2.5 * _safe_log10(r["flux"])
        rows.append([i, f"{r['xcentroid']+1:.3f}", f"{r['ycentroid']+1:.3f}", f"{mag:.3f}", 0])
    _write_catalog_tsv(out, rows, ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])
    print(f"starfind: {len(s)} stars")


def task_psfphot(args):
    from photutils.detection import DAOStarFinder
    from photutils.psf import IntegratedGaussianPRF, PSFPhotometry
    from astropy.stats import sigma_clipped_stats
    data, _ = _read_image(args.fits)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    psf = IntegratedGaussianPRF(sigma=2.0)
    photometry = PSFPhotometry(
        psf_model=psf, fit_shape=(11, 11), finder=finder, aperture_radius=4.0
    )
    res = photometry(data - med)
    out = os.path.join(args.outdir, "out.tsv")
    rows = []
    for i, r in enumerate(res, 1):
        mag = -2.5 * _safe_log10(float(r["flux_fit"]))
        rows.append([i, f"{r['x_fit']+1:.3f}", f"{r['y_fit']+1:.3f}", f"{mag:.3f}", 0])
    _write_catalog_tsv(out, rows, ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])
    print(f"psfphot: {len(rows)} sources")


def task_crowded(args):
    from photutils.detection import DAOStarFinder
    from photutils.psf import IntegratedGaussianPRF, IterativePSFPhotometry
    from astropy.stats import sigma_clipped_stats
    data, _ = _read_image(args.fits)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=3.0 * std)
    psf = IntegratedGaussianPRF(sigma=2.0)
    iph = IterativePSFPhotometry(
        psf_model=psf, fit_shape=(11, 11), finder=finder, maxiters=3,
        aperture_radius=4.0,
    )
    res = iph(data - med)
    out = os.path.join(args.outdir, "out.tsv")
    rows = []
    for i, r in enumerate(res, 1):
        mag = -2.5 * _safe_log10(float(r["flux_fit"]))
        flag = int(r.get("iter_detected", 0)) if hasattr(r, "get") else 0
        rows.append([i, f"{r['x_fit']+1:.3f}", f"{r['y_fit']+1:.3f}", f"{mag:.3f}", flag])
    _write_catalog_tsv(out, rows, ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "FLAGS"])
    print(f"crowded: {len(rows)} sources")


def task_lsbg(args):
    from astropy.convolution import Gaussian2DKernel, convolve
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import detect_sources, SourceCatalog
    data, _ = _read_image(args.fits)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    smooth = convolve(data - med, Gaussian2DKernel(x_stddev=4.0))
    seg = detect_sources(smooth, threshold=1.0 * std, npixels=200)
    out = os.path.join(args.outdir, "out.tsv")
    if seg is None:
        _write_catalog_tsv(out, [], ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO",
                                     "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "FLAGS"])
        print("lsbg: 0 candidates")
        return
    cat = SourceCatalog(data - med, seg)
    rows = []
    for i, r in enumerate(cat, 1):
        flux = float(getattr(r, "kron_flux", 0) or 0)
        mag = -2.5 * _safe_log10(flux)
        a = float(r.semimajor_sigma.value)
        b = float(r.semiminor_sigma.value)
        try:
            theta = float(r.orientation.to("deg").value)
        except Exception:
            theta = 0.0
        rows.append([i,
                     f"{r.xcentroid+1:.3f}", f"{r.ycentroid+1:.3f}",
                     f"{mag:.3f}", f"{a:.3f}", f"{b:.3f}", f"{theta:.2f}", 0])
    _write_catalog_tsv(out, rows, ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO",
                                   "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "FLAGS"])
    print(f"lsbg: {len(rows)} candidates")


def task_deconv(args):
    from astropy.io import fits
    try:
        from skimage.restoration import richardson_lucy
    except Exception:
        # fallback: a couple of unsharp-mask iterations
        from scipy.ndimage import gaussian_filter
        data, hdr = _read_image(args.fits)
        out = data + 0.5 * (data - gaussian_filter(data, 2.0))
        path = os.path.join(args.outdir, "out.fits")
        fits.PrimaryHDU(out.astype(np.float32), hdr).writeto(path, overwrite=True)
        print("deconv: scipy fallback (unsharp mask)")
        return
    data, hdr = _read_image(args.fits)
    sx = 2.0
    k = int(np.ceil(6 * sx)) | 1
    yk, xk = np.mgrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1]
    psf = np.exp(-(xk * xk + yk * yk) / (2 * sx * sx)).astype(np.float32)
    psf /= psf.sum()
    dmin = float(np.nanmin(data))
    dmax = float(np.nanmax(data))
    span = max(dmax - dmin, 1e-9)
    norm = np.clip((data - dmin) / span, 0, 1)
    deconv = richardson_lucy(norm, psf, num_iter=20)
    out = (deconv * span + dmin).astype(np.float32)
    path = os.path.join(args.outdir, "out.fits")
    fits.PrimaryHDU(out, hdr).writeto(path, overwrite=True)
    print("deconv: richardson_lucy 20 iter")


def task_morphometry(args):
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage import uniform_filter
    if args.cx is None or args.cy is None:
        print("err: morphometry needs --cx --cy --r")
        sys.exit(2)
    data, _ = _read_image(args.fits)
    cut, _ = _cutout(data, args.cx, args.cy, args.r or 30.0)
    if cut.size == 0:
        print("morphometry: empty cutout")
        return
    _, med, _ = sigma_clipped_stats(cut, sigma=3.0)
    sub = (cut - med).astype(np.float64)
    # Gini
    flat = np.sort(np.abs(sub).ravel())
    n = flat.size
    s = float(flat.sum()) + 1e-12
    g = (2.0 * np.sum((np.arange(1, n + 1)) * flat) - (n + 1) * s) / (n * s)
    # M20
    pos = np.clip(sub, 0, None)
    yy, xx = np.indices(sub.shape)
    flux = float(pos.sum())
    if flux <= 0:
        flux = 1e-9
    cx_loc = float((pos * xx).sum() / flux)
    cy_loc = float((pos * yy).sum() / flux)
    r2 = (xx - cx_loc) ** 2 + (yy - cy_loc) ** 2
    mtot = float((pos * r2).sum())
    flat_idx = np.argsort(pos.ravel())[::-1]
    cum = np.cumsum(pos.ravel()[flat_idx])
    target = 0.2 * pos.sum()
    i20 = int(np.searchsorted(cum, target))
    sel = flat_idx[:max(i20, 1)]
    m20 = np.log10(max((pos.ravel()[sel] * r2.ravel()[sel]).sum(), 1e-12) / max(mtot, 1e-12)) if mtot > 0 else float("nan")
    # Concentration
    cum_total = np.cumsum(np.sort(pos.ravel())[::-1])
    if cum_total[-1] > 0:
        i20a = int(np.searchsorted(cum_total, 0.2 * cum_total[-1]))
        i80a = int(np.searchsorted(cum_total, 0.8 * cum_total[-1]))
        r20 = np.sqrt(max(i20a, 1) / np.pi)
        r80 = np.sqrt(max(i80a, 1) / np.pi)
        C = 5.0 * np.log10(r80 / r20) if r20 > 0 else float("nan")
    else:
        C = float("nan")
    # Asymmetry
    rot = sub[::-1, ::-1]
    A = float(np.abs(sub - rot).sum()) / (2.0 * float(np.abs(sub).sum()) + 1e-12)
    # Smoothness
    sm = uniform_filter(sub, size=5)
    S = float(np.abs(sub - sm).sum()) / (float(np.abs(sub).sum()) + 1e-12)
    print(f"GINI {g:.4f}  M20 {m20:.4f}  C {C:.4f}  A {A:.4f}  S {S:.4f}")


def task_sersic(args):
    from astropy.modeling import models, fitting
    if args.cx is None or args.cy is None:
        print("err: sersic needs --cx --cy --r")
        sys.exit(2)
    data, _ = _read_image(args.fits)
    cut, _ = _cutout(data, args.cx, args.cy, args.r or 50.0)
    if cut.size == 0:
        print("sersic: empty cutout")
        return
    yy, xx = np.indices(cut.shape)
    p_init = models.Sersic2D(
        amplitude=float(np.nanmax(cut)),
        r_eff=10.0, n=2.5,
        x_0=cut.shape[1] / 2.0, y_0=cut.shape[0] / 2.0,
        ellip=0.1, theta=0.0,
    )
    fit = fitting.LevMarLSQFitter()
    m = fit(p_init, xx, yy, cut, maxiter=300)
    print(f"SERSIC n={m.n.value:.3f}  re={m.r_eff.value:.2f}  amp={m.amplitude.value:.4g}  "
          f"ellip={m.ellip.value:.3f}  theta={m.theta.value:.3f}")


def task_bulgedisk(args):
    from astropy.modeling import models, fitting
    if args.cx is None or args.cy is None:
        print("err: bulgedisk needs --cx --cy --r")
        sys.exit(2)
    data, _ = _read_image(args.fits)
    cut, _ = _cutout(data, args.cx, args.cy, args.r or 60.0)
    if cut.size == 0:
        print("bulgedisk: empty cutout")
        return
    yy, xx = np.indices(cut.shape)
    cxl, cyl = cut.shape[1] / 2.0, cut.shape[0] / 2.0
    amp = float(np.nanmax(cut))
    bulge = models.Sersic2D(amplitude=amp * 0.5, r_eff=3.0, n=4.0,
                            x_0=cxl, y_0=cyl, ellip=0.1, theta=0.0)
    disk = models.Sersic2D(amplitude=amp * 0.5, r_eff=15.0, n=1.0,
                           x_0=cxl, y_0=cyl, ellip=0.2, theta=0.0)
    model = bulge + disk
    fit = fitting.LevMarLSQFitter()
    m = fit(model, xx, yy, cut, maxiter=500)
    bulge_eval = m[0](xx, yy)
    disk_eval = m[1](xx, yy)
    bf = float(np.sum(bulge_eval))
    df = float(np.sum(disk_eval))
    bt = bf / max(bf + df, 1e-9)
    print(f"BULGEDISK B/T={bt:.3f}  bulge_re={m[0].r_eff.value:.2f}  "
          f"bulge_n={m[0].n.value:.3f}  disk_re={m[1].r_eff.value:.2f}")


def task_icl(args):
    from astropy.stats import sigma_clipped_stats
    if args.cx is None or args.cy is None:
        print("err: icl needs --cx --cy --r")
        sys.exit(2)
    data, _ = _read_image(args.fits)
    h, w = data.shape
    yy, xx = np.indices(data.shape)
    cx, cy = args.cx, args.cy
    r_outer = args.r or min(h, w) / 3.0
    r_inner = max(20.0, r_outer * 0.05)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (rr > r_inner) & (rr < r_outer) & (data < med + 5.0 * std)
    flux = float(np.sum(data[mask] - med))
    area = int(mask.sum())
    sb = flux / max(area, 1)
    print(f"ICL flux={flux:.4g}  area={area}px  inner={r_inner:.1f}  outer={r_outer:.1f}  mean_sb={sb:.4g}")


def _ogfinder_root():
    h = os.environ.get("OGFINDER_HOME")
    if h and os.path.isdir(h):
        return h
    home = os.path.expanduser("~/BACKUP/ds9/OGFinder")
    if os.path.isdir(home):
        return home
    return None


def _stub(label, subdir):
    root = _ogfinder_root()
    if root is None:
        print(f"{label}: needs $OGFINDER_HOME (or ~/BACKUP/ds9/OGFinder); not found")
        return
    target = os.path.join(root, subdir)
    if os.path.isdir(target):
        print(f"{label}: stub — wire {target} into ds9-rust to enable full pipeline")
    else:
        print(f"{label}: needs {subdir}/ under $OGFINDER_HOME")


def task_photoz(args):
    _stub("photoz", "photo_z")


def task_sed(args):
    _stub("sed", "sed_fit")


def task_aimerge(args):
    _stub("aimerge", "ai_merge")


# ----------------------------------------------------------------- main --

DISPATCH = {
    "starfind":    task_starfind,
    "psfphot":     task_psfphot,
    "crowded":     task_crowded,
    "lsbg":        task_lsbg,
    "deconv":      task_deconv,
    "morphometry": task_morphometry,
    "sersic":      task_sersic,
    "bulgedisk":   task_bulgedisk,
    "icl":         task_icl,
    "photoz":      task_photoz,
    "sed":         task_sed,
    "aimerge":     task_aimerge,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("fits")
    p.add_argument("outdir")
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--r", type=float, default=None)
    args = p.parse_args()
    fn = DISPATCH.get(args.task)
    if fn is None:
        print(f"err: unknown task '{args.task}'")
        sys.exit(2)
    try:
        fn(args)
    except Exception as e:
        print(f"{args.task}: {type(e).__name__}: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
