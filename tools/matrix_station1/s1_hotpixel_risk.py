"""Cell 2: the specific risk `docs/CALIBRATION_FRAMES.md` flags for eclipse fields, measured
on Station 1.

That document's verdict table ends with "the eclipse field itself -- untested, and there is a
specific risk": a hot pixel promoted to a star reaches the deflection fit unchallenged,
because stage 3 does not reject on residual, and the dark-free hot-pixel search
"correctly declines" below `hotpixels.MIN_DITHER_PX` = 3 px of dither. It names the test:
"run it both ways and compare the STAR LIST, not just the plate scale."

Station 1 is where that matters, and this measures the three things that decide it:

  1. **The dither.** Measured straight from the raw frames by phase correlation on a
     2048 px window away from the Sun, after a 64 px block background is removed -- without
     that the correlation locks onto the vignetting, which is fixed to the detector, and
     returns zero for the wrong reason. If the dither is below 3 px the dark-free path
     cannot run and a master dark is the only hot-pixel rejection available; if it is also
     near zero, hot pixels do not smear either, so they stack coherently.

  2. **Whether the hot pixels are really there at -10 C.** The calibration set was shot at
     +25 C, 50-75 minutes after totality, so the dark's amplitudes are wrong -- but the
     pipeline **excludes** dark-flagged pixels from the stack rather than subtracting them
     (`stacker_implementation.py` ~1352), so it only needs to know *which* pixels, not how
     much. This checks the dark's hot list against the eclipse frames themselves.

  3. **Whether any of them reached the star list.** The stacked detections, the ones matched
     to a catalogue star, and the science set that actually carries L, each tested for
     coincidence with the dark's hot pixels under the shifts the stacker used, against the
     coincidence rate at random positions.

Needs `s1_darks_flats.py` (masters) and `s1_eclipse_convention.py` (the stacks).
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

REC = r"D:/MEE2024 output/MEE_output/station1_record"
DF = os.path.join(REC, 'darks_flats')
EC = os.path.join(REC, 'eclipse_convention')
TIER = r"G:/Mexico April 2024/Station-1-Eclipse-Data/CapObj/2024-04-08_18_12_30Z"
NX, NY, PS = 9576, 6388, 1.84847
HOT_ADU = 200
MIN_DITHER_PX = 3.0          # hotpixels.MIN_DITHER_PX

# ---------------------------------------------------------------- 1. the dither
print('=== 1. the dither, measured from the raw frames ===')
fs = sorted(glob.glob(os.path.join(TIER, '*.FIT')))
y0, x0, S, blk = 4000, 400, 2048, 64


def highpass(f):
    with fits.open(f) as h:
        s = h[0].section[y0:y0+S, x0:x0+S].astype(np.float32)
    v = s.reshape(S//blk, blk, S//blk, blk)
    bg = np.repeat(np.repeat(np.median(v, axis=(1, 3)), blk, 0), blk, 1)
    r = s - bg
    n = 1.4826*np.median(np.abs(r))
    return np.clip(r, 0, 50*n), n


ref, n0 = highpass(fs[0])
Fr = np.fft.rfft2(ref)
shifts = [(0, 0)]
for i in (1, 5, 20, 40, 61, 90, len(fs)-1):
    sub, _ = highpass(fs[i])
    R = Fr*np.conj(np.fft.rfft2(sub)); R /= np.abs(R) + 1e-9
    cc = np.fft.irfft2(R, s=ref.shape)
    p = np.unravel_index(np.argmax(cc), cc.shape)
    dy = p[0] - S*(p[0] > S//2); dx = p[1] - S*(p[1] > S//2)
    q = cc.max()/np.median(np.abs(cc))
    shifts.append((int(dx), int(dy)))
    print('   frame %3d vs 0: dx %+3d dy %+3d px (correlation peak %.0fx the median%s)'
          % (i, dx, dy, q, '' if q > 20 else ' -- WEAK'))
span = max(max(abs(a) for a, b in shifts), max(abs(b) for a, b in shifts))
print('   dither span over the block: %d px, against the dark-free search\'s %.0f px minimum.' % (span, MIN_DITHER_PX))
print('   -> the dark-free persistence search %s on this field.'
      % ('CANNOT run and will correctly decline' if span < MIN_DITHER_PX else 'can run'))
print('   -> hot pixels %s across the stack.'
      % ('do NOT smear: they land on the same stacked pixel in every frame and add coherently'
         if span < MIN_DITHER_PX else 'smear over the dither and lose contrast'))

# ---------------------------------------------------------------- 2. are they there at -10 C
print('\n=== 2. the dark\'s hot pixels, checked against the eclipse frames ===')
bias = fits.getdata(os.path.join(DF, 'master_bias.fits')).astype(np.float32)
dark = fits.getdata(os.path.join(DF, 'master_dark-400ms.fits')).astype(np.float32)
ex = dark - bias
hot = ex > HOT_ADU
hy, hx = np.nonzero(hot)
print('   master dark-400ms at +25 C: %d pixels above %d ADU (%.5f %% of the frame)' % (hot.sum(), HOT_ADU, 100*hot.mean()))
print('   (from s1_darks_flats.py: in the -10 C eclipse frames those same pixels sit %s)'
      % '+41.5 ADU median = 3.3 sigma, 44 % of them above 5 sigma, against 5.4 % at random')
print('   in a 123-frame stack with no dither the per-frame noise falls as sqrt(123) while a')
print('   fixed excess does not, so a 41 ADU pixel reaches roughly %.0f sigma on the stacked image.'
      % (41.5/(12.6/np.sqrt(123))))

# ---------------------------------------------------------------- 3. did any reach the star list
print('\n=== 3. did any reach the star list? ===')
rng = np.random.default_rng(17)
hotset = set(zip(hy.tolist(), hx.tolist()))


def coincides(px, py, used):
    out = np.zeros(len(px), bool)
    for dx, dy in used:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                xs = np.round(px).astype(int) - dx + ox
                ys = np.round(py).astype(int) - dy + oy
                out |= np.array([(int(b), int(a)) in hotset for a, b in zip(xs, ys)])
    return out


for est in ('windowed', 'moments'):
    z = glob.glob(os.path.join(EC, est, 'centroid_data*.zip'))
    if not z:
        print('   %s: no stack yet' % est); continue
    zf = zipfile.ZipFile(z[0])
    r = json.load(zf.open('results.txt'))
    c = pd.read_csv(zf.open('STACKED_CENTROIDS_DATA.csv')); c.columns = [k.strip() for k in c.columns]
    sh = np.array(r.get('alignment', {}).get('shifts_px', [[0, 0]]))
    used = sorted(set(map(tuple, np.round(sh).astype(int))))
    print('\n   --- %s stack: %d centroids from %d frames; stacker shifts used %s'
          % (est, len(c), r.get('#frames stacked'), used[:6]))
    sets = [('all stacked detections', c.px.values, c.py.values)]
    for refname in ('A_2024_17field_moments', 'C_2field_windowed'):
        zz = glob.glob(os.path.join(EC, est, refname, '**', 'distortion_data*.zip'), recursive=True)
        if not zz:
            continue
        zf2 = zipfile.ZipFile(zz[0])
        d = pd.read_csv(zf2.open([n for n in zf2.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
        d.columns = [k.strip() for k in d.columns]
        sets.append(('matched to a catalogue star', d.px.values, d.py.values))
        k = (d.magV.values <= 12) & (d['flag_is_outlier'].values == False)
        sets.append(('the science set, G<=12, non-outlier', d.px.values[k], d.py.values[k]))
        break
    rx, ry = rng.integers(2, NX-2, 20000), rng.integers(2, NY-2, 20000)
    null = 100*coincides(rx, ry, used).mean()
    for name, px, py in sets:
        if len(px) == 0:
            continue
        n = int(coincides(px, py, used).sum())
        print('      %-36s %5d positions, %3d on a dark-hot pixel (%.2f %%); random %.2f %%'
              % (name, len(px), n, 100*n/len(px), null))
print('\n->', EC)
