"""The anchor treatment for Bruns' two inner stars: per-frame centroids, raw frames.

Why: the stacked-path measurement of G 7.52 (1.49 R_sun) returned +0.04 arcsec of
radial deflection vs GR's +1.17 -- and the follow-up measurements cleared the suspects
one by one: NOT saturation (raw peak 24.6 kADU, zero clipped pixels), NOT an oversized
mask (2 % of pixels in its 650-700 px ring genuinely saturate), leaving the steep
coronal gradient: ~2.7 kADU of star on a ~22 kADU sloping background, blur-subtracted
with a 10 px Gaussian that cannot follow the inner-annulus radial curvature. This is
the regime Bruns' continuity correction existed for.

So: measure both stars the way the Leon anchor was admitted -- independently on every
raw E2 frame, with a LOCAL background model per window (border-ring plane fit, robust
to the radial gradient's linear part), Gaussian-weighted centroid iteration (sigma 2.5,
5 iterations), saturated pixels excluded. Per-frame scatter is the honest error; the
median is the measurement. Displacements go through the same host chain and corrected
catalogue as the union, so the numbers are directly comparable.

G 7.09 (1.62 R_sun), whose stacked value (+1.44 arcsec vs GR 1.08) looked sane, runs
through the identical machinery as the method check.
"""
import glob, os, sys
import numpy as np
from astropy.io import fits

sys.argv = ['x']
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
exec(src.split("# ---- per-tier star tables")[0])       # host chain, catalogue, affines

RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
STARS = {'G7.52': (2102.0, 1241.0), 'G7.09': (1179.0, 2314.0)}
W = 15                    # window half-size, px
SIG = 2.5                 # centroid weighting sigma, px

def measure(img, x0, y0):
    """Plane-fit background on the window border ring, then iterated
    Gaussian-weighted centroid. Returns (x, y, peak_above_bg) or None."""
    xi, yi = int(round(x0)), int(round(y0))
    w = img[yi-W:yi+W+1, xi-W:xi+W+1].astype(np.float64)
    if w.shape != (2*W+1, 2*W+1):
        return None
    yy, xx = np.mgrid[-W:W+1, -W:W+1]
    ring = (np.maximum(np.abs(xx), np.abs(yy)) >= W-2) & (w < 65535)
    A = np.c_[xx[ring], yy[ring], np.ones(ring.sum())]
    cf, *_ = np.linalg.lstsq(A, w[ring], rcond=None)
    bg = cf[0]*xx + cf[1]*yy + cf[2]
    s = w - bg
    s[w >= 65535] = 0.0
    cx, cy = 0.0, 0.0
    for _ in range(5):
        g = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*SIG**2))
        ws = np.clip(s, 0, None)*g
        tot = ws.sum()
        if tot <= 0: return None
        cx, cy = float((ws*xx).sum()/tot), float((ws*yy).sum()/tot)
    return xi+cx, yi+cy, float(s[W-3:W+4, W-3:W+4].max())

print(f'{"star":6} {"frame":>5} {"px":>8} {"py":>8} {"peak-bg (ADU)":>13}')
results = {}
for name, (x0, y0) in STARS.items():
    xs, ys, pk = [], [], []
    for k, f in enumerate(sorted(glob.glob(os.path.join(RAW, 'E2_*.fit')))):
        img = fits.getdata(f)
        m = measure(img, x0, y0)
        if m is None:
            print(f'{name:6} {k:5d}   -- failed'); continue
        xs.append(m[0]); ys.append(m[1]); pk.append(m[2])
        print(f'{name:6} {k:5d} {m[0]:8.2f} {m[1]:8.2f} {m[2]:13.0f}')
    mx, my = float(np.median(xs)), float(np.median(ys))
    sx = 1.4826*np.median(np.abs(np.array(xs)-mx))
    sy = 1.4826*np.median(np.abs(np.array(ys)-my))
    print(f'{name}: median ({mx:.2f}, {my:.2f}) px, per-frame MAD scatter '
          f'({sx*PS:.2f}, {sy*PS:.2f}) arcsec, se of median '
          f'({sx*PS/np.sqrt(len(xs)):.2f}, {sy*PS/np.sqrt(len(xs)):.2f}) arcsec, '
          f'median peak {np.median(pk):.0f} ADU', flush=True)
    results[name] = (mx, my, len(xs))

# NOTE: per-frame positions need no inter-frame alignment IF the drift within the 15 s
# E2 series is small against the scatter -- measured directly: the per-frame positions
# of BOTH stars drift together if the mount moves; their DIFFERENCE is drift-free.
print()
for name, (mx, my, nfr) in results.items():
    sky = chain(np.array([[my, mx]]))
    cra, cdec = corrs['E2']
    d = np.hypot((sky[0,1]-cra)*np.cos(np.radians(cdec)), sky[0,0]-cdec)*3600
    i = int(np.argmin(d))
    dxi = (sky[0,1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
    deta = (sky[0,0]-cdec[i])*3600
    dpx = ax[0]*dxi/3600 + ax[1]*deta/3600
    dpy = ay[0]*dxi/3600 + ay[1]*deta/3600
    # E2's per-tier constant pointing offset, from the union build (the tier table's
    # pre-median-removal offset is not stored; use the OTHER tier stars via the union
    # machinery instead: recompute the tier table exactly as b17_union does)
    results[name] = (mx, my, nfr, dpx*PS, dpy*PS, cmag[i], i)
    print(f'{name}: matched G {cmag[i]:.2f} at {d[i]:.2f} arcsec; raw displacement '
          f'({dpx*PS:+.2f}, {dpy*PS:+.2f}) arcsec sensor-axes (pre-offset)', flush=True)

# tier offset: median displacement of ALL E2 detections through the same chain
import pandas as pd, zipfile
det = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(B, 'E2', 'centroid_data*.zip'))[0])
                  .open('STACKED_CENTROIDS_DATA.csv'))
sky = chain(det[['py','px']].values.astype(float))
cra, cdec = corrs['E2']
offs = []
for i in np.where(not_dbl)[0]:
    d = np.hypot((sky[:,1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:,0]-cdec[i])*3600
    j = int(np.argmin(d))
    if d[j] < 8.0:
        dxi = (sky[j,1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
        deta = (sky[j,0]-cdec[i])*3600
        offs.append((ax[0]*dxi/3600 + ax[1]*deta/3600, ay[0]*dxi/3600 + ay[1]*deta/3600))
offs = np.array(offs)*PS
ox, oy = np.median(offs[:,0]), np.median(offs[:,1])
print(f'\nE2 tier constant offset (median of {len(offs)} wide-gate matches): '
      f'({ox:+.2f}, {oy:+.2f}) arcsec')
print(f'{"star":6} {"R (Rsun)":>8} {"radial defl (as)":>16} {"GR (as)":>8} {"per-frame se (as)":>17}')
for name, (mx, my, nfr, dx, dy, mag, i) in results.items():
    rx_ = (mx-SUNPX)*PS; ry_ = (my-SUNPY)*PS
    R_ = np.hypot(rx_, ry_)
    dr = (dx-ox)*(rx_/R_) + (dy-oy)*(ry_/R_)
    print(f'{name:6} {R_/R_SUN_AS:8.2f} {dr:16.3f} {L_REF*R_SUN_AS/R_:8.3f}')
print('done', flush=True)
