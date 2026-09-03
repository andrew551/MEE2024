"""Cell 2: do the clouds actually move the stars?

Douglas, 2026-09-03: "let's remember that as it could definitely creates some gradients that
could move stars. Is the position variance greater in those frames with clouds than the ones
where the presumed clouds were not present (in the same 0.4s exposure tier)?"

`s1_cloud_gradient.py`'s finding was that the sky level and its **tilt across the frame** both
vary within every block, the tilt swinging 60-100 ADU/s per 1000 px and rotating inside a
minute, which only something moving in the atmosphere can do. That establishes the gradient
exists. It does not establish that it moves a centroid.

This measures that directly, per frame, without stacking anything:

  * a dozen bright stars are picked from the block's stacked image, and each is re-centroided
    in **every one of the 123 frames** with the same windowed estimator, reading only a small
    box around each so the whole block costs a few hundred megabytes rather than fifteen
    gigabytes;
  * four background patches in the same pass give that frame's sky level and tilt, so the
    cloud indicator and the positions come from identical data;
  * **any flux measured alongside must use an ANNULUS, not a box median.** Under the
    steep moving gradient these frames carry, a box median sits below the true
    background on the brighter side and inflates the star. Measured with a box median
    the stars appeared 4 % BRIGHTER in the cloudy frames; measured with a 9-14 px
    annulus they are 14 % FAINTER, which is the real answer and is ordinary extinction;
  * each star's position is referred to its own mean over the block, and the frames are then
    split into the cloudy ones and the clear ones by their tilt.

Three questions, in order of how much they would matter:

  1. is the per-frame position SCATTER larger when the tilt is large?
  2. does the per-frame mean position SHIFT correlate with the tilt vector -- that is, does
     the sky gradient pull the stars in the direction it points, which is the mechanism a
     background gradient would use?
  3. does it depend on stellar brightness, as a background-gradient effect must (a faint star
     is pulled further than a bright one by the same gradient)?

A yes to 2 is the one that would matter for L, because a gradient fixed in the sky frame for
tens of seconds pulls a whole region of stars the same way, which is exactly the shape the
deflection fit is looking for.

**What the answer turned out to be.** The stars dim 14 % and the sky rises 21 % in the frames
with the steeper gradient, which for a background-limited centroid predicts a scatter ratio of
sqrt(1.21)/0.859 = 1.28. The raw split gives 1.68 and the detrended split 1.06; since the
cloud thickened monotonically through the block, detrending removes the signal along with the
confound, so 1.06 is a lower bound and 1.68 an upper one. Roughly +28 % is the number to
carry, it is NOISE rather than bias (the increase is the same for the bright and faint halves,
1.76 against 1.79, which rules out a background-gradient pull), and it therefore averages down
in a 123-frame stack. It argues for frame weighting, not for a correction to L.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

G = r"G:/Mexico April 2024/Station-1-Eclipse-Data/CapObj/2024-04-08_18_12_30Z"
ECL24 = r"D:/MEE2024 output/Station 1/eclipse fields/CENTROID_OUTPUT20240416232626"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/cloud_positions"
NX, NY, PS = 9576, 6388, 1.84847
BIAS = 503.0
BOX = 20            # half-size of the box read around each star
SIGW = 2.0          # the windowed estimator's window, as everywhere else
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- pick the stars
z = glob.glob(os.path.join(ECL24, '..', 'centroid_data20240416232626.zip'))
z = z[0] if z else os.path.join(os.path.dirname(ECL24), 'centroid_data20240416232626.zip')
zf = zipfile.ZipFile(z)
c = pd.read_csv(zf.open('STACKED_CENTROIDS_DATA.csv'))
c.columns = [k.strip() for k in c.columns]
fluxcol = [k for k in c.columns if 'flux' in k.lower()][0]
# bright, compact, away from the Sun and the frame edge
c['r'] = np.hypot(c.px - 4309, c.py - 2730)
sel = c[(c.r > 1200) & (c.px > 3*BOX) & (c.py > 3*BOX) & (c.px < NX-3*BOX) & (c.py < NY-3*BOX)]
sel = sel.nlargest(14, fluxcol)
STARS = list(zip(sel.px.values, sel.py.values, sel[fluxcol].values))
print('tracking %d stars, flux %.0f to %.0f (noise-normed), %.0f to %.0f px from the Sun'
      % (len(STARS), sel[fluxcol].min(), sel[fluxcol].max(), sel.r.min(), sel.r.max()), flush=True)

PATCH = [(600, 600), (8400, 600), (600, 5400), (8400, 5400)]


def windowed(sub, x0, y0):
    """The same fixed-Gaussian-window iterated centroid the pipeline uses, on one box."""
    yy, xx = np.mgrid[-BOX:BOX+1, -BOX:BOX+1]
    s = np.clip(sub - np.median(sub), 0, None)
    cx, cy = x0, y0
    for _ in range(8):
        w = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*SIGW**2))*s
        t = w.sum()
        if t <= 0:
            return None
        nx, ny = (w*xx).sum()/t, (w*yy).sum()/t
        if abs(nx-cx) < 1e-4 and abs(ny-cy) < 1e-4:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    return cx, cy


fs = sorted(glob.glob(os.path.join(G, '*.FIT')))
rows = []
for i, f in enumerate(fs):
    with fits.open(f) as h:
        hdu = h[0]
        expo = hdu.header.get('EXPTIME', 0.4)
        t = hdu.header.get('DATE-OBS', '')[11:19]
        sky = []
        for (x, y) in PATCH:
            sky.append(float(np.median(hdu.section[y:y+300, x:x+300])))
        pos = []
        for (sx, sy, fl) in STARS:
            iy, ix = int(round(sy)), int(round(sx))
            sub = hdu.section[iy-BOX:iy+BOX+1, ix-BOX:ix+BOX+1].astype(np.float32)
            r = windowed(sub, sx-ix, sy-iy)
            pos.append((np.nan, np.nan) if r is None else (ix+r[0], iy+r[1]))
    sky = (np.array(sky) - BIAS)/expo
    X = np.array([[p[0]-NX/2, p[1]-NY/2, 1.0] for p in PATCH])
    cf, *_ = np.linalg.lstsq(X, sky, rcond=None)
    rows.append(dict(frame=i, t=t, level=sky.mean(), tiltx=cf[0]*1000, tilty=cf[1]*1000,
                     **{('x%d' % k): pos[k][0] for k in range(len(STARS))},
                     **{('y%d' % k): pos[k][1] for k in range(len(STARS))}))
    if i % 25 == 0:
        print('   frame %3d %s  sky %.0f ADU/s  tilt (%+.1f, %+.1f)' % (i, t, sky.mean(), cf[0]*1000, cf[1]*1000), flush=True)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, 'per_frame.csv'), index=False)

# ---------------------------------------------------------------- the three questions
ns = len(STARS)
dx = np.column_stack([D['x%d' % k] - D['x%d' % k].mean() for k in range(ns)])
dy = np.column_stack([D['y%d' % k] - D['y%d' % k].mean() for k in range(ns)])
# remove each frame's rigid shift: that is tracking, not the sky
rigx, rigy = np.nanmedian(dx, axis=1), np.nanmedian(dy, axis=1)
resx, resy = dx - rigx[:, None], dy - rigy[:, None]
tilt = np.hypot(D.tiltx.values, D.tilty.values)
scatter = np.sqrt(np.nanmean(resx**2 + resy**2, axis=1))*PS       # arcsec, per frame
cloudy = tilt > np.median(tilt)

print('%s=== 1. does the position scatter grow with the tilt? ===' % chr(10))
print('   tilt (ADU/s per 1000 px): median %.1f, range %.1f-%.1f' % (np.median(tilt), tilt.min(), tilt.max()))
for nm, k in (('the calmer half', ~cloudy), ('the cloudier half', cloudy)):
    print('   %-18s %3d frames: per-frame star scatter %.4f " (rigid shift removed)'
          % (nm, k.sum(), np.nanmean(scatter[k])))
lo, hi = np.nanmean(scatter[~cloudy]), np.nanmean(scatter[cloudy])
print('   raw: the cloudier half looks %.0f %%%% noisier, r = %+.3f -- but see the control below'
      % (100*(hi/lo-1), np.corrcoef(tilt, scatter)[0, 1]))
print()
print('   THE CONTROL THAT MATTERS. The sky level, the tilt and the scatter all climb')
print('   monotonically through the block, so splitting at the median tilt very nearly')
print('   splits at the midpoint in TIME, and any two of them correlate whether or not one')
print('   causes the other. Detrending both against frame number separates cloud from clock:')
tt = D.frame.values.astype(float)


def _detrend(v, deg):
    return v - np.polyval(np.polyfit(tt, v, deg), tt)


for deg in (1, 2, 3, 5):
    print('     after removing a degree-%d trend: r = %+.3f'
          % (deg, np.corrcoef(_detrend(tilt, deg), _detrend(scatter, deg))[0, 1]))
dt = _detrend(tilt, 3)
kk = dt > np.median(dt)
print('     split on the DETRENDED tilt: %.4f " against %.4f ", ratio %.2f'
      % (np.nanmean(scatter[~kk]), np.nanmean(scatter[kk]),
         np.nanmean(scatter[kk])/np.nanmean(scatter[~kk])))
print('   -> the cloud-specific effect on position scatter is a few per cent, not the raw')
print('      difference. Most of the raw gap is the shared trend in time.')

print('\n=== 2. does the rigid shift follow the tilt vector? (the mechanism that would matter) ===')
for nm, a, b in (('x', D.tiltx.values, rigx*PS), ('y', D.tilty.values, rigy*PS)):
    good = np.isfinite(b)
    r = np.corrcoef(a[good], b[good])[0, 1]
    sl = np.polyfit(a[good], b[good], 1)[0]
    print('   tilt_%s against the frame\'s rigid shift in %s: r = %+.3f, slope %+.5f "/ (ADU/s per 1000 px)'
          % (nm, nm, r, sl))
print('   (a real background-gradient pull would show a consistent sign and a slope that,')
print('    over the observed +-40 tilt range, is worth something against the 0.14-0.28" residual)')

print('\n=== 3. is any effect brightness dependent, as a background pull must be? ===')
fl = np.array([s[2] for s in STARS])
bright, faint = fl > np.median(fl), fl <= np.median(fl)
for nm, k in (('brightest half', bright), ('faintest half', faint)):
    sc_lo = np.sqrt(np.nanmean(resx[~cloudy][:, k]**2 + resy[~cloudy][:, k]**2))*PS
    sc_hi = np.sqrt(np.nanmean(resx[cloudy][:, k]**2 + resy[cloudy][:, k]**2))*PS
    print('   %-15s calm %.4f "  cloudy %.4f "  ratio %.2f' % (nm, sc_lo, sc_hi, sc_hi/sc_lo))
print('\n->', OUT)
