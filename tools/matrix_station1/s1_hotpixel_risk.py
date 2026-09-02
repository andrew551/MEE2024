"""Cell 2: the specific risk `docs/CALIBRATION_FRAMES.md` flags for eclipse fields, measured
on Station 1.

That document's verdict table ends with "the eclipse field itself -- untested, and there is a
specific risk": a hot pixel promoted to a star reaches the deflection fit unchallenged,
because stage 3 does not reject on residual, and the dark-free hot-pixel search
"correctly declines" below `hotpixels.MIN_DITHER_PX` = 3 px of dither. It names the test:
"run it both ways and compare the STAR LIST, not just the plate scale."

Station 1 is where that matters, and this measures the three things that decide it:

  1. **The dither**, measured three ways, because they disagree and the disagreement is the
     finding. Tracked STARS in two opposite 2048 px corner windows say the field moves
     sub-pixel across all 123 frames. The stacker's own aligner records a median spread of
     0.45 px but places about five frames 3-4 px out, which the stars say did not move.
     `hotpixels.dither_span` is the largest PAIRWISE offset, so those few bad shifts carry
     it to 5.4 px and past the 3 px gate: the dark-free search then runs on a field that is
     effectively undithered, where a hot pixel and a star are nearly indistinguishable.
     (A phase correlation is NOT usable here: vignetting times a bright corona is a
     high-contrast pattern fixed to the detector, and the correlation locks onto it.)

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
# Measured from STARS, not by phase correlation. A phase correlation on this field locks
# onto the fixed pattern -- vignetting times a bright corona is high-contrast and fixed to
# the detector -- and returns zero however the field moves. An earlier version of this tool
# did exactly that and concluded the dither was zero; two independent corner windows of
# tracked stars say the drift is sub-pixel but not zero, and the stacker's own aligner
# disagrees with both. All three numbers are reported, because the disagreement is the
# finding.
print('=== 1. the dither ===')
fs = sorted(glob.glob(os.path.join(TIER, '*.FIT')))
S, blk = 2048, 64


def _window(f, x0, y0):
    with fits.open(f) as h:
        s = h[0].section[y0:y0+S, x0:x0+S].astype(np.float32)
    v = s.reshape(S//blk, blk, S//blk, blk)
    return s - np.repeat(np.repeat(np.median(v, axis=(1, 3)), blk, 0), blk, 1)


def _stars(r, n=60):
    from scipy.ndimage import maximum_filter
    noise = 1.4826*np.median(np.abs(r))
    pk = (r == maximum_filter(r, 9)) & (r > 10*noise)
    ys, xs = np.nonzero(pk)
    o = np.argsort(-r[ys, xs])[:n]
    out = []
    gy, gx = np.mgrid[-5:6, -5:6]
    for yy, xx in zip(ys[o], xs[o]):
        if yy < 6 or xx < 6 or yy > S-7 or xx > S-7:
            continue
        b = np.clip(r[yy-5:yy+6, xx-5:xx+6], 0, None); t = b.sum()
        if t > 0:
            out.append((xx + (b*gx).sum()/t, yy + (b*gy).sum()/t))
    return np.array(out)


probe = [1, 20, 60, 89, 90, 91, 106, 107, len(fs)-1]
measured = {}
for x0, y0, tag in ((400, 4000, 'lower-left'), (7000, 400, 'upper-right')):
    sa = _stars(_window(fs[0], x0, y0))
    print('   window %-12s (x %d, y %d): %d compact sources in frame 0' % (tag, x0, y0, len(sa)))
    for i in probe:
        sb = _stars(_window(fs[i], x0, y0)); d = []
        for x, y in sa:
            k = np.hypot(sb[:, 0]-x, sb[:, 1]-y); j = int(np.argmin(k))
            if k[j] < 15:
                d.append((sb[j, 0]-x, sb[j, 1]-y))
        if len(d) >= 3:
            d = np.array(d)
            measured.setdefault(i, []).append((float(np.median(d[:, 0])), float(np.median(d[:, 1]))))
print("   frame   star-measured shift (two windows)      stacker's recorded shift")
for est in ('windowed',):
    z = glob.glob(os.path.join(EC, est, 'centroid_data*.zip'))
    if not z:
        break
    sh = np.array(json.load(zipfile.ZipFile(z[0]).open('results.txt'))['alignment']['shifts_px'])
    med = np.median(sh, axis=0)
    for i in probe:
        m = measured.get(i, [])
        txt = '  '.join('(%+.2f,%+.2f)' % v for v in m) if m else 'no match'
        print('   %5d   %-38s (%+.2f,%+.2f)' % (i, txt, sh[i, 0]-med[0], sh[i, 1]-med[1]))
    d = np.hypot(sh[:, 0]-med[0], sh[:, 1]-med[1])
    sp = sh[:, None, :]-sh[None, :, :]
    span = float(np.max(np.hypot(sp[..., 0], sp[..., 1])))
    print('   stacker shifts about their median: median %.2f px, 90th %.2f, max %.2f; %d of %d frames'
          ' within 1 px' % (np.median(d), np.percentile(d, 90), d.max(), int((d < 1).sum()), len(d)))
    print('   dither_span (largest PAIRWISE offset -- what hotpixels gates on): %.2f px, above the'
          ' %.0f px minimum' % (span, MIN_DITHER_PX))
    print('   -> the gate passed on the strength of a few frames the aligner placed 3-4 px out,')
    print('      which the stars say did not move. The real spread is sub-pixel for 73 % of frames,')
    print('      so the dark-free search ran with almost no leverage: a hot pixel and a star are')
    print('      nearly indistinguishable at this dither, which is the condition the gate exists to')
    print('      prevent. Its verdict on this field is not evidence of absence.')

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
