"""Cell 2: do darks and flats earn their place on Station 1, when `docs/CALIBRATION_FRAMES.md`
found they did not for Bruns 2017 or the 2026 L/R calibration?

Douglas, 2026-09-03. That document's verdict table is careful about what is measured and what
is not: flat-darks and the 2026 CAL_piLeo calibration "not needed" (measured), the zenith
calibration "not needed" on one controlled A/B, the 2017 L/R pair "unresolved at 30 ppm", and
**the eclipse field itself "untested, and there is a specific risk"** -- a hot pixel promoted
to a star reaches the deflection fit unchallenged. It also says its measurements are all at
-20 C (2017) or +10 C (2026) and do not bear on campaigns where dark current is measurable.

Station 1 needs its own answer for three reasons: a full-frame sensor with real vignetting,
0.25-0.4 s eclipse exposures, and a calibration set taken 50-75 minutes AFTER totality --
`G:\\Mexico April 2024\\Station-1-Eclipse-Data\\{bias,dark-250ms,dark-300ms,dark-400ms,
darkflats,flat}`, 40 frames each, 19:00-19:26 UTC, at CCD-TEMP **25-27 C** against the
eclipse frames' **-10 C**.

Three measurements:

  1. **Is the dark usable as a dark?** The master dark's excess over bias and its hot-pixel
     census, then the same pixels examined in the eclipse frames themselves at their real
     temperature, after a local background subtraction (without one the corona swamps
     everything -- an earlier pass of this tool reported "48 % of pixels hot", which was the
     corona). The question the subtraction has to pass is whether the +25 C hot pixels are
     the ones the -10 C frames actually have.

  2. **What does the flat do to a centroid?** Not by the smooth-field approximation
     sigma^2 * dlnF/dx, which is wrong for pixel-scale PRNU, but by injection: a Gaussian PSF
     of the measured FWHM 3.74 px is placed at 4000 random sub-pixel positions, multiplied by
     the real master flat there, and centroided both ways (windowed with sigma 2.0 px, and
     the footprint moment). The shift distribution is what flat-fielding would remove.

  3. **Is the flat's structure smooth enough to be absorbed by the distortion model?** The
     injected shift field is refitted by a quintic in position: the part a quintic takes
     cancels in the transfer (same sensor, same model on both sides), the residual does not.

Writes station1_record/darks_flats/ (masters as FITS, so a re-stack can use them without
rebuilding).
"""
import glob, os
import numpy as np
from astropy.io import fits

G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/darks_flats"
NX, NY, PS = 9576, 6388, 1.84847
FWHM_PX = 3.74
SIG = FWHM_PX/2.355
HOT_ADU = 200
os.makedirs(OUT, exist_ok=True)


def median_stack(fs, rows=400):
    """Row-chunked median. A whole-frame np.median over forty 61-Mpx frames upcasts to
    float64 and needs ~20 GB; this needs 40 * 400 * 9576 * 4 bytes at a time. `.section`
    reads just the requested rows and applies BZERO/BSCALE -- plain `.data[y0:y1]` under
    memmap raises on these frames because they carry BZERO."""
    out = np.empty((NY, NX), dtype=np.float32)
    handles = [fits.open(f) for f in fs]
    try:
        for y0 in range(0, NY, rows):
            y1 = min(y0+rows, NY)
            block = np.stack([h[0].section[y0:y1].astype(np.float32) for h in handles])
            out[y0:y1] = np.median(block, axis=0)
            del block
    finally:
        for h in handles:
            h.close()
    return out


def master(name, limit=40):
    p = os.path.join(OUT, 'master_%s.fits' % name)
    if os.path.exists(p):
        return fits.getdata(p).astype(np.float32)
    fs = sorted(glob.glob(os.path.join(G, name, '**', '*.FIT'), recursive=True))[:limit]
    if not fs:
        return None
    m = median_stack(fs)
    fits.PrimaryHDU(m).writeto(p, overwrite=True)
    print('  built master_%s from %d frames' % (name, len(fs)), flush=True)
    return m


def block_background(img, blk=64):
    """Median in blk x blk blocks, expanded back to full size: a crude but adequate local
    background, which the corona makes essential."""
    ny, nx = (NY//blk)*blk, (NX//blk)*blk
    v = img[:ny, :nx].reshape(ny//blk, blk, nx//blk, blk)
    m = np.median(v, axis=(1, 3))
    bg = np.repeat(np.repeat(m, blk, axis=0), blk, axis=1)
    full = np.empty_like(img)
    full[:ny, :nx] = bg
    full[ny:, :] = full[ny-1:ny, :]
    full[:, nx:] = full[:, nx-1:nx]
    return full


print('=== masters ===', flush=True)
bias = master('bias')
dark4 = master('dark-400ms')
flat = master('flat')
dflat = master('darkflats')

print('\n=== 1. is the dark usable as a dark? ===')
ex = dark4 - bias
print('  master dark-400ms minus bias, at CCD-TEMP 25-27 C: median %+.2f ADU, mean %+.2f, 99.9%% %+.1f, max %+.0f'
      % (np.median(ex), ex.mean(), np.percentile(ex, 99.9), ex.max()))
for t in (20, 50, 200, 1000):
    print('     %8d pixels more than %5d ADU above bias  (%.5f %% of the frame)' % ((ex > t).sum(), t, 100*(ex > t).mean()))
print('  so at 0.4 s even at 25 C the dark is a bias plus a defect map: the median excess is'
      ' %.2f ADU. Dark CURRENT is not the issue; the %d hot pixels are.' % (np.median(ex), (ex > HOT_ADU).sum()))

fs = sorted(glob.glob(os.path.join(G, 'CapObj', '2024-04-08_18_12_30Z', '*.FIT')))
print('\n  the eclipse frames at their real -10 C (median of 12 of %d, corona removed by a 64 px block background):' % len(fs))
E = median_stack(fs[:12])
Eb = E - block_background(E)
noise = 1.4826*np.median(np.abs(Eb[::7, ::7]))
print('     local-background-subtracted frame: noise %.1f ADU, 99.9%% %+.1f, max %+.0f' % (noise, np.percentile(Eb, 99.9), Eb.max()))
hot = ex > HOT_ADU
hy, hx = np.nonzero(hot)
vals = Eb[hy, hx]
print('     at the %d positions the 25 C dark calls hot: median excess in the eclipse frames %+.1f ADU'
      ' (%.1f sigma), 90th %+.1f, max %+.0f' % (hot.sum(), np.median(vals), np.median(vals)/noise, np.percentile(vals, 90), vals.max()))
print('     of those %d, above 5 sigma in the eclipse frames: %d (%.0f %%)'
      % (hot.sum(), (vals > 5*noise).sum(), 100*(vals > 5*noise).mean()))
rng = np.random.default_rng(5)
ry, rx = rng.integers(0, NY, 20000), rng.integers(0, NX, 20000)
rv = Eb[ry, rx]
print('     at random positions, for comparison: median %+.1f ADU, above 5 sigma %.2f %%' % (np.median(rv), 100*(rv > 5*noise).mean()))
print('     the dark says these pixels carry %+.0f ADU; the eclipse frames show %+.1f ADU at the same places.'
      % (np.median(ex[hy, hx]), np.median(vals)))

print('\n=== 2. what the flat does to a centroid, by injection ===')
F = flat - dflat
F = F/np.median(F)
bad = ~np.isfinite(F) | (F < 0.2)
print('  master flat (flat minus darkflat, normalised): 1st-99th percentile %.3f-%.3f; %d pixels below 0.2 (dead), masked'
      % (np.percentile(F, 1), np.percentile(F, 99), bad.sum()))
cen = F[NY//2-500:NY//2+500, NX//2-500:NX//2+500].mean()
corners = [F[:1000, :1000].mean(), F[:1000, -1000:].mean(), F[-1000:, :1000].mean(), F[-1000:, -1000:].mean()]
print('  vignetting: centre %.4f, corners %.4f-%.4f, a %.1f %% fall-off' % (cen, min(corners), max(corners), 100*(1-min(corners)/cen)))
Fc = np.where(bad, 1.0, F)

half = 8
yy, xx = np.mgrid[-half:half+1, -half:half+1]
N = 4000
cx = rng.uniform(half+2, NX-half-2, N); cy = rng.uniform(half+2, NY-half-2, N)
sx_w, sy_w, sx_m, sy_m = (np.zeros(N) for _ in range(4))
for i in range(N):
    x0, y0 = cx[i], cy[i]
    ix, iy = int(round(x0)), int(round(y0))
    dxg, dyg = xx + ix - x0, yy + iy - y0
    psf = np.exp(-(dxg**2 + dyg**2)/(2*SIG**2))
    patch = Fc[iy-half:iy+half+1, ix-half:ix+half+1]
    obs = psf*patch
    # footprint moment over the whole patch
    t = obs.sum()
    sx_m[i] = (obs*dxg).sum()/t; sy_m[i] = (obs*dyg).sum()/t
    # windowed: iterate a fixed Gaussian window of sigma 2.0 px, as windowed_centroid does
    wx, wy = 0.0, 0.0
    for _ in range(6):
        w = np.exp(-((dxg-wx)**2 + (dyg-wy)**2)/(2*2.0**2))
        ow = obs*w; tw = ow.sum()
        wx, wy = (ow*dxg).sum()/tw, (ow*dyg).sum()/tw
    sx_w[i], sy_w[i] = wx, wy
for nm, a, b in (('footprint moment', sx_m, sy_m), ('windowed (sigma 2.0 px)', sx_w, sy_w)):
    r = np.hypot(a, b)*PS*1000
    print('  %-24s centroid shift from the flat: rms %.1f mas, median %.1f, 99th %.1f, max %.0f'
          % (nm, np.sqrt((r**2).mean()), np.median(r), np.percentile(r, 99), r.max()))
print('  (against Station 1\'s per-star fit residual of 47-56 mas at G <= 12 and the 90 mas transfer floor)')

print('\n=== 3. how much of it would the distortion model absorb? ===')
xn, yn = (cx-NX/2)/(NX/2), (cy-NY/2)/(NX/2)
A = np.column_stack([xn**i*yn**j for i in range(6) for j in range(6-i)])
for nm, a, b in (('footprint moment', sx_m, sy_m), ('windowed', sx_w, sy_w)):
    tot, res = [], []
    for v in (a*PS*1000, b*PS*1000):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        tot.append(np.sqrt((v**2).mean())); res.append(np.sqrt(((v - A@c)**2).mean()))
    print('  %-18s total rms %.1f / %.1f mas (x / y) -> after a quintic in position %.1f / %.1f mas'
          % (nm, tot[0], tot[1], res[0], res[1]))
print('  a quintic takes the vignetting; what is left is pixel-scale PRNU, which is random per')
print('  star position and so adds in quadrature to the per-star scatter rather than biasing L.')
print('\n->', OUT)
