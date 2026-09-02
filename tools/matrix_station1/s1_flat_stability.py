"""Cell 2: did the flat survive the refocus? The zenith-session flat against the post-eclipse one.

Douglas, 2026-09-03, having found `I:\\Mexico 2024\\Station 1 Zenith\\Flats\\2024-04-08_06_28_18Z`.

This is the measurement the earlier flat verdict could not make. `s1_darks_flats.py` showed
the POST-ECLIPSE flat (`G:` set, 19:21 UTC, gain 0, 27.6 C) moves a centroid by only
2.4-3.3 mas, so flat-fielding is not worth doing for its own sake. But that left the
transfer argument untested: the calibration fields were shot at 05:32-06:15 UTC and the
eclipse at 18:12, with a daytime refocus between them worth -600 ppm of plate scale, and
dust shadows change size with focus. A flat that changed between the two sides is a
systematic that does NOT cancel in the transfer, however small each side's flat-fielding
effect is on its own.

Now both flats exist:

    zenith session   06:28:19 UTC, 2 s, gain 100, offset 1,  -1.5 C, 20 frames  (I:)
    post-eclipse     19:21:27 UTC, 2 s, gain 0,   offset 50, 27.6 C, 40 frames  (G:)

Different gain and offset, so each is normalised by its own median before comparison -- a
flat is a response map and the pedestal divides out. `CALIBRATION_FRAMES.md` measured
flat-darks at 87 ppm of the normalised flat, so a scalar pedestal is adequate here and no
matching dark-flat is needed (there is none at gain 100 offset 1 in any case).

Reports:

  1. the ratio of the two normalised flats -- if the optics were stable it is flat and near
     unity everywhere, and its structure is where they differ;
  2. the vignetting profile of each, which is the focus-independent part;
  3. the CENTROID SHIFT the difference would cause, by the same PSF injection
     `s1_darks_flats.py` uses -- which is the number that matters, because it is what fails
     to cancel between the calibration side and the eclipse side.

Writes station1_record/darks_flats/master_flat_zenith.fits and a summary.
"""
import glob, os
import numpy as np
from astropy.io import fits

ZEN = r"I:/Mexico 2024/Station 1 Zenith/Flats"
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/darks_flats"
NX, NY, PS = 9576, 6388, 1.84847
FWHM_PX = 3.74
SIG = FWHM_PX/2.355
os.makedirs(OUT, exist_ok=True)


def median_stack(fs, rows=400):
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


def cached(name, fs):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        return fits.getdata(p).astype(np.float32)
    m = median_stack(fs)
    fits.PrimaryHDU(m).writeto(p, overwrite=True)
    print('  built %s from %d frames' % (name, len(fs)), flush=True)
    return m


print('=== masters ===', flush=True)
zfs = sorted(glob.glob(os.path.join(ZEN, '**', '*.FIT'), recursive=True))
h = fits.getheader(zfs[0])
print('  zenith flats: %d frames, %s s, gain %s, offset %s, %s C, %s'
      % (len(zfs), h.get('EXPTIME'), h.get('GAIN'), h.get('OFFSET'), h.get('CCD-TEMP'), h.get('DATE-OBS')))
FZ = cached('master_flat_zenith.fits', zfs)
FE = fits.getdata(os.path.join(OUT, 'master_flat.fits')).astype(np.float32)
DE = fits.getdata(os.path.join(OUT, 'master_darkflats.fits')).astype(np.float32)

# Each normalised by its own median. The eclipse flat keeps its matching dark-flat; the
# zenith flat has none (there is no 2 s dark at gain 100 offset 1), so a scalar pedestal is
# used -- adequate because CALIBRATION_FRAMES.md measures flat-darks at 87 ppm of the
# normalised flat, and because both flats sit at a third of full well where a few hundred
# ADU of pedestal is a per-cent effect on the shape. The sensitivity to that choice is
# reported below rather than assumed away.
#
# NOT the 2nd percentile of the flat itself: that is the VIGNETTED CORNER (16.8 kADU of a
# 23.7 kADU flat), and subtracting it manufactures a 60 % fall-off that is not there. An
# earlier version of this tool did exactly that and reported a spurious 13 % change in
# response between the two sessions.
PEDESTAL_Z = 500.0
fe = (FE - DE); fe = fe/np.median(fe)
fz = (FZ - PEDESTAL_Z); fz = fz/np.median(fz)
bad = ~np.isfinite(fz) | ~np.isfinite(fe) | (fz < 0.2) | (fe < 0.2)
print('  normalised: zenith 1st-99th %.3f-%.3f, post-eclipse %.3f-%.3f; %d pixels masked'
      % (np.percentile(fz, 1), np.percentile(fz, 99), np.percentile(fe, 1), np.percentile(fe, 99), bad.sum()))

print('\n=== 1. the ratio, zenith flat / post-eclipse flat ===')
ratio = np.where(bad, 1.0, fz/np.where(fe < 0.2, 1.0, fe))
ratio = ratio/np.median(ratio)
print('  ratio: median %.4f, 1st-99th percentile %.4f-%.4f, rms about 1 %.4f'
      % (np.median(ratio), np.percentile(ratio, 1), np.percentile(ratio, 99), np.std(ratio)))
for ped in (0.0, 250.0, 500.0, 1000.0):
    t = (FZ - ped); t = t/np.median(t)
    q = np.where(bad, 1.0, t/np.where(fe < 0.2, 1.0, fe)); q = q/np.median(q)
    c = q[:(NY//64)*64, :(NX//64)*64].reshape(NY//64, 64, NX//64, 64).mean(axis=(1, 3))
    print('     with a %6.0f ADU pedestal assumed for the zenith flat: smoothed ratio spans %.4f-%.4f'
          % (ped, c.min(), c.max()))
blk = 64
ny, nx = (NY//blk)*blk, (NX//blk)*blk
coarse = ratio[:ny, :nx].reshape(ny//blk, blk, nx//blk, blk).mean(axis=(1, 3))
print('  smoothed to %d px blocks (removes pixel noise): min %.4f, max %.4f, rms about 1 %.4f'
      % (blk, coarse.min(), coarse.max(), np.std(coarse)))
print('  -> a value of 1.000 everywhere would mean the response did not change between the')
print('     zenith session and the eclipse; structure here is what the refocus moved.')

print('\n=== 2. vignetting profile of each ===')
yy, xx = np.mgrid[0:NY, 0:NX]
r = np.hypot(xx - NX/2, yy - NY/2)
print('  %-10s %10s %10s %10s' % ('r (px)', 'zenith', 'post-ecl', 'ratio'))
for lo, hi in ((0, 500), (1000, 1500), (2000, 2500), (3000, 3500), (4000, 4500), (5000, 5600)):
    k = (r >= lo) & (r < hi) & ~bad
    print('  %-10s %10.4f %10.4f %10.4f' % ('%d-%d' % (lo, hi), fz[k].mean(), fe[k].mean(), fz[k].mean()/fe[k].mean()))

print('\n=== 3. what the DIFFERENCE would do to a centroid ===')
print('  (the flat-fielding error that fails to cancel between the two sides of the transfer)')
rng = np.random.default_rng(11)
half = 8
gy, gx = np.mgrid[-half:half+1, -half:half+1]
N = 4000
cx = rng.uniform(half+2, NX-half-2, N); cy = rng.uniform(half+2, NY-half-2, N)
Rc = np.where(bad, 1.0, ratio)
sx_w, sy_w, sx_m, sy_m = (np.zeros(N) for _ in range(4))
for i in range(N):
    x0, y0 = cx[i], cy[i]
    ix, iy = int(round(x0)), int(round(y0))
    dxg, dyg = gx + ix - x0, gy + iy - y0
    psf = np.exp(-(dxg**2 + dyg**2)/(2*SIG**2))
    obs = psf*Rc[iy-half:iy+half+1, ix-half:ix+half+1]
    t = obs.sum()
    sx_m[i] = (obs*dxg).sum()/t; sy_m[i] = (obs*dyg).sum()/t
    wx, wy = 0.0, 0.0
    for _ in range(6):
        w = np.exp(-((dxg-wx)**2 + (dyg-wy)**2)/(2*2.0**2))
        ow = obs*w; tw = ow.sum()
        wx, wy = (ow*dxg).sum()/tw, (ow*dyg).sum()/tw
    sx_w[i], sy_w[i] = wx, wy
for nm, a, b in (('footprint moment', sx_m, sy_m), ('windowed (sigma 2.0 px)', sx_w, sy_w)):
    v = np.hypot(a, b)*PS*1000
    print('  %-24s shift from the flat CHANGE: rms %.1f mas, median %.1f, 99th %.1f, max %.0f'
          % (nm, np.sqrt((v**2).mean()), np.median(v), np.percentile(v, 99), v.max()))
xn, yn = (cx-NX/2)/(NX/2), (cy-NY/2)/(NX/2)
A = np.column_stack([xn**i*yn**j for i in range(6) for j in range(6-i)])
for nm, a, b in (('footprint moment', sx_m, sy_m), ('windowed', sx_w, sy_w)):
    out = []
    for v in (a*PS*1000, b*PS*1000):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        out.append((np.sqrt((v**2).mean()), np.sqrt(((v - A@c)**2).mean())))
    print('  %-18s total %.1f / %.1f mas -> after a quintic in position %.1f / %.1f mas'
          % (nm, out[0][0], out[1][0], out[0][1], out[1][1]))
print('\n  compare: the post-eclipse flat on its own moves a centroid 2.4 mas (moment) /'
      ' 3.3 mas (windowed),')
print('  and Station 1\'s per-star fit residual is 47-56 mas at G <= 12.')
print('\n->', OUT)
