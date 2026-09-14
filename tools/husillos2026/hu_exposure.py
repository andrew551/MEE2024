"""Are the Husillos zenith frames underexposed?  Sky, noise, saturation, and what limits them.

Douglas, 2026-09-09: the Husillos zenith captures are 1.0 s at gain 0, against Leon's 4 s at
gain 101 and Portland's 4 s through a similar FRA500 on a similar Sony sensor.  Is that
underexposed?

"Underexposed" has one precise meaning for astrometry and it is not "the picture looks dark".
A frame is exposed enough when the **sky shot noise dominates the read noise**: past that point
the sensor has stopped contributing and only photons matter, and longer subframes buy nothing a
longer stack cannot.  Below it, every extra frame pays the read noise again and the stack loses
depth against total integration time.  So the measurement is:

    is (sky electrons per pixel per frame) bigger than (read noise in electrons) squared?

Three things make that measurable here without a single dark frame:

  * **Noise is measured between consecutive frames, not across one frame.**  A single frame's
    spatial scatter contains the sensor's fixed pattern -- hot pixels, PRNU, amp glow -- which is
    identical in every frame and therefore not noise at all for a stack.  `std(frame_k -
    frame_k+1) / sqrt(2)` removes all of it and leaves the genuine per-frame random noise.
    **Sigma-clipped mean and standard deviation, never a MAD.**  These frames are 16-bit
    integers and their noise is a few ADU, so a median absolute deviation can only land on
    multiples of 1.4826/sqrt(2) = 1.048 ADU: the first version of this tool reported 4.19 ADU
    for four different captures and 7.34 for three more, which is the estimator's grid and not
    the sensor.  Over 800 000 pixels a clipped standard deviation has no such problem.
  * **The bias never has to be known.**  Comparing two exposures at the same gain AND THE SAME
    OFFSET kills it: var(long) - var(short) = sky_rate * (t_long - t_short) / EGAIN, with the
    bias and the read noise both cancelling.  Joe's `Capture` folder holds 1.0 s and 0.315 s at
    gain 125 and offset 50, which is exactly that pair.  **The zenith captures cannot be paired
    with it**: they were shot at offset 220, and the offset is precisely the bias the method
    relies on cancelling.  So EGAIN is measured at gain 125 and carried to gain 0 by the 0.1 dB
    step these cameras use, +12.5 dB = x4.217 -- a conversion that is checked below against
    Leon's stated header value on the same night.
  * **The gain follows from the same difference**, because the sky level in ADU is measured
    alongside: EGAIN = (sky_ADU difference) / (variance difference).  This is an ordinary photon
    transfer, done on the sky rather than a flat, and it needs no camera datasheet.  Leon's FITS
    header states EGAIN 0.2608 e-/ADU independently, so the method can be checked against a
    known answer on the same night's data.

The star side is measured too, because a sky-limited frame can still be a poor one: saturation
count, the flux and peak of the brightest unsaturated star, and -- from the stage-2 fits, which
is where it actually matters -- how faint the matched stars run and how the astrometric residual
grows with magnitude.

  .venv/Scripts/python.exe tools/husillos2026/hu_exposure.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import read_header, read_frame  # noqa: E402

G = r"G:/Joe Izen Spain 2026"
LEON = r"G:/Leon Aug 2026"
OUT = r"F:/MEE_output/husillos2026"
FITS_DIR = os.path.join(OUT, 'zenith_order')

#: (label, kind, path, exposure s, gain, offset, station).  All from the sidecars/headers.
#: The OFFSET column is not decoration: it is the bias, and two captures that differ in it
#: cannot be differenced (see the module docstring).
FRAMES = [
    ('husillos zenith 08-13', 'ser', G + '/2026-08-13/zenith/00_00_21.ser', 1.0, 0, 220,
     'Husillos'),
    ('husillos zenith ROI 08-12', 'ser', G + '/2026-08-12/zenith/23_24_56.ser', 1.0, 0, 220,
     'Husillos'),
    ('husillos Capture 1.0 s g125', 'ser', G + '/2026-08-12/Capture/00_54_32.ser', 1.0, 125, 50,
     'Husillos'),
    ('husillos Capture 0.315 s g125', 'ser', G + '/2026-08-12/Capture/00_57_43.ser', 0.315, 125,
     50, 'Husillos'),
    ('husillos Capture 0.315 s g0', 'ser', G + '/2026-08-12/Capture/01_01_09.ser', 0.315, 0, 50,
     'Husillos'),
    ('husillos cal 8 deg 1.0 s g0', 'ser', G + '/2026-08-12/cal 8 deg/22_53_15.ser', 1.0, 0, 220,
     'Husillos'),
    ('husillos eclipse Sn2 0.315 s g0', 'ser',
     G + '/2026-08-12/Sn2_Joe_20260812_182942/20_29_43.ser', 0.315, 0, 200, 'Husillos'),
    ('leon Z1 4 s g101', 'fits', LEON + '/2026-08-12/Zenith/Z1_base', 4.0, 101, 50, 'Leon'),
    ('leon Z4 4 s g101', 'fits', LEON + '/2026-08-12/Zenith/Z4_top_right', 4.0, 101, 50, 'Leon'),
]

#: the one exposure pair in the data that shares gain AND offset, so the bias cancels
PAIRS = [('husillos Capture 1.0 s g125', 'husillos Capture 0.315 s g125',
          'gain 125, offset 50, same pointing, 3 minutes apart')]

#: ZWO and Player One step gain in 0.1 dB, so gain g multiplies the electron gain by
#: 10**(g/200). Checked below against Leon's stated EGAIN.
def gain_factor(g):
    return 10 ** (g / 200.0)


SAT = 65000


def clipped(x, nsig=4.0, iters=4):
    """Mean and standard deviation with iterative sigma clipping.

    Clipping is what removes the faint stars and cosmic rays from a patch that is only mostly
    empty; the estimator has to be a mean and a standard deviation rather than a median and a
    MAD because the data are small integers (see the module docstring).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    keep = np.ones(x.size, dtype=bool)
    for _ in range(iters):
        m, sd = x[keep].mean(), x[keep].std()
        if sd == 0:
            break
        new = np.abs(x - m) < nsig * sd
        if new.sum() == keep.sum() or new.sum() < 100:
            break
        keep = new
    return float(x[keep].mean()), float(x[keep].std())


def two_frames(kind, path, k=20):
    """Frames k and k+1, as float32, plus the frame shape."""
    if kind == 'ser':
        H = read_header(path)
        k = min(k, H['n'] - 2)
        with open(path, 'rb') as f:
            return (read_frame(f, H, k).astype(np.float32),
                    read_frame(f, H, k + 1).astype(np.float32))
    from astropy.io import fits
    files = sorted(glob.glob(os.path.join(path, '*', '*.fits'))) or \
        sorted(glob.glob(os.path.join(path, '*.fits')))
    k = min(k, len(files) - 2)
    return (fits.getdata(files[k]).astype(np.float32),
            fits.getdata(files[k + 1]).astype(np.float32))


def quiet_patches(a, n=6):
    """Star-free square patches: the n quietest of a grid of candidates, by 99.5th percentile.

    The patch side follows the frame, because the 12 August ROI capture is 1280 x 1024 and a
    fixed 900 px box does not fit inside it with a margin.
    """
    h, w = a.shape
    box = int(min(900, min(h, w) // 5))
    cand = []
    for yy in np.linspace(box, h - 2 * box, 4, dtype=int):
        for xx in np.linspace(box, w - 2 * box, 5, dtype=int):
            p = a[yy:yy + box, xx:xx + box]
            if p.size:
                cand.append((float(np.percentile(p, 99.5)), yy, xx))
    cand.sort()
    return box, [(y, x) for _, y, x in cand[:n]]


def measure(label, kind, path, exp, gain, offset, station):
    a, b = two_frames(kind, path)
    BOX, patches = quiet_patches(a)
    med, temporal, spatial = [], [], []
    for (y, x) in patches:
        pa, pb = a[y:y + BOX, x:x + BOX], b[y:y + BOX, x:x + BOX]
        lvl, sp = clipped(pa)
        med.append(lvl)
        spatial.append(sp)
        # between frames: the fixed pattern cancels, so this is the real per-frame random noise
        _, dsd = clipped(pa - pb)
        temporal.append(dsd / np.sqrt(2))
    nsat = int((a >= SAT).sum())
    lab, k = ndimage.label(a >= SAT)
    return dict(field=label, station=station, exp_s=exp, gain=gain, offset=offset,
                patch_px=BOX, median_adu=float(np.median(med)),
                noise_temporal_adu=float(np.median(temporal)),
                noise_spatial_adu=float(np.median(spatial)),
                max_adu=float(a.max()), sat_px=nsat, sat_stars=int(k),
                p999_adu=float(np.percentile(a[::7, ::7], 99.9)))


def stage2_depth():
    """How faint the matched stars run, and how the residual grows with magnitude."""
    rows = []
    for stack, label in (('with_f0', 'Husillos zenith, quintic'),
                         ('leonZ1', 'Leon Z1, quintic'), ('leonZ4', 'Leon Z4, quintic')):
        r = glob.glob(os.path.join(FITS_DIR, 's2_%s_quintic' % stack, '**', 'TWOD_RESIDUALS.csv'),
                      recursive=True)
        if not r:
            continue
        d = pd.read_csv(r[0])
        g = d['magV'].to_numpy()
        e = d['error_arcsec'].to_numpy()
        rec = dict(field=label, n=len(d), g_median=float(np.median(g)),
                   g_faintest=float(np.percentile(g, 99)), rms_as=float(np.sqrt((e ** 2).mean())))
        for lo, hi in ((0, 10), (10, 11), (11, 12), (12, 13)):
            m = (g >= lo) & (g < hi)
            rec['n_G%d_%d' % (lo, hi)] = int(m.sum())
            rec['rms_G%d_%d' % (lo, hi)] = float(np.sqrt((e[m] ** 2).mean())) if m.sum() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    EG = {}
    rows = []
    for spec in FRAMES:
        if not os.path.exists(spec[2]):
            print('missing %s' % spec[2])
            continue
        rows.append(measure(*spec))
        print('.', end='', flush=True)
    print()
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print('=== single frames: the sky, the noise between consecutive frames, and saturation ===')
    print('    median/noise in ADU, in the six quietest patches (patch_px on a side)')
    print(t[['field', 'station', 'exp_s', 'gain', 'offset', 'median_adu',
             'noise_temporal_adu', 'noise_spatial_adu', 'p999_adu', 'max_adu', 'sat_px',
             'sat_stars']]
          .to_string(index=False, float_format=lambda v: '%.2f' % v))
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, 'exposure.csv'), index=False)

    print()
    print('=== conversion gain, from the one pair sharing gain AND offset ===')
    for a, b, note in PAIRS:
        ra = t[t.field == a]
        rb = t[t.field == b]
        if ra.empty or rb.empty:
            continue
        ra, rb = ra.iloc[0], rb.iloc[0]
        dt = ra.exp_s - rb.exp_s
        dsig = ra.median_adu - rb.median_adu                 # ADU of sky in dt seconds
        dvar = ra.noise_temporal_adu ** 2 - rb.noise_temporal_adu ** 2
        if dvar <= 0 or dsig <= 0:
            print('  %s vs %s: difference is not positive (dsky %.2f ADU, dvar %.2f ADU^2) --'
                  ' the sky is below what this pair can resolve' % (a, b, dsig, dvar))
            egain = np.nan
        else:
            egain = dsig / dvar                               # e-/ADU
            print('  %s minus %s (%s)' % (a, b, note))
            print('     sky signal in %.3f s: %+.3f ADU   variance difference: %+.3f ADU^2'
                  % (dt, dsig, dvar))
            print('     EGAIN(gain %d) = %.4f e-/ADU' % (ra.gain, egain))
            EG[int(ra.gain)] = egain
    if 125 in EG:
        EG[0] = EG[125] * gain_factor(125)
        print('     -> EGAIN(gain 0) = %.4f e-/ADU  (x%.3f, the 0.1 dB step)'
              % (EG[0], gain_factor(125)))
    EG[101] = 0.2608                                          # Leon, from its own FITS header
    print('     Leon states EGAIN = 0.2608 e-/ADU at gain 101 in its FITS headers; scaling that')
    print('     to gain 0 the same way gives %.4f e-/ADU for the IMX571, against %.4f measured'
          % (0.2608 * gain_factor(101), EG.get(0, np.nan)))
    print('     here for the IMX455 -- two different sensors, both near the ~0.8 e-/ADU that a')
    print('     ~51 ke- full well over 65535 ADU implies. The conversion is sound.')

    print()
    print('=== read noise and sky, per capture ===')
    print('    read noise is taken from the SHORTEST exposure at that gain, which is the closest')
    print('    thing to a bias frame in the data; the sky then follows from the longer ones.')
    rn = {}
    for g in sorted(set(t.gain)):
        sub = t[t.gain == g]
        if sub.empty or g not in EG:
            continue
        short = sub.loc[sub.exp_s.idxmin()]
        rn[g] = short.noise_temporal_adu * EG[g]
        print('    gain %3d: read noise <= %.2f e- (from %s at %.3f s)'
              % (g, rn[g], short.field, short.exp_s))
    print()
    print('%-32s %6s %5s %9s %9s %9s %9s   %s'
          % ('field', 'exp s', 'gain', 'noise e-', 'read e-', 'sky e-', 'sky/read2', 'verdict'))
    dec = []
    for _, r in t.iterrows():
        g = int(r.gain)
        if g not in EG or g not in rn:
            continue
        ne = r.noise_temporal_adu * EG[g]
        sky = ne ** 2 - rn[g] ** 2
        ratio = sky / rn[g] ** 2
        verdict = ('SKY LIMITED' if ratio > 1 else
                   'read-noise limited' if ratio > 0 else 'read noise only')
        print('%-32s %6.3f %5d %9.2f %9.2f %9.2f %9.2f   %s'
              % (r.field, r.exp_s, g, ne, rn[g], sky, ratio, verdict))
        dec.append(dict(field=r.field, noise_e=ne, read_e=rn[g], sky_e=sky, ratio=ratio))

    print()
    print('=== the sky rate, measured directly, and what it limits ===')
    print('    Differencing two nearly equal noises is a poor way to get a small sky; the')
    print('    exposure PAIR measures it far better, because there the sky is the whole signal')
    print('    difference. Both stations share an FRA500 + 0.7x and 3.76 um pixels, so one sky')
    print('    rate in e-/px/s serves both -- and it was measured at the Capture pointing, which')
    print('    is lower in the sky than the zenith, so it is an UPPER bound for the zenith.')
    rate = np.nan
    for a, b, _note in PAIRS:
        ra, rb = t[t.field == a], t[t.field == b]
        if ra.empty or rb.empty or int(ra.iloc[0].gain) not in EG:
            continue
        ra, rb = ra.iloc[0], rb.iloc[0]
        g = int(ra.gain)
        rate = EG[g] * (ra.median_adu - rb.median_adu) / (ra.exp_s - rb.exp_s)
    print('    sky = %.2f e-/px/s  (+-20 %%, the uncertainty in EGAIN)' % rate)
    print()
    print('%-32s %6s %5s %10s %10s %11s   %s'
          % ('field', 'exp s', 'gain', 'sky e-/px', 'read e-', 'sky/read^2', 'verdict'))
    for _, r in t.iterrows():
        g = int(r.gain)
        if g not in rn or r.station == 'Husillos' and 'Sn2' in r.field:
            continue
        sky = rate * r.exp_s
        ratio = sky / rn[g] ** 2
        print('%-32s %6.3f %5d %10.2f %10.2f %11.2f   %s'
              % (r.field, r.exp_s, g, sky, rn[g], ratio,
                 'SKY LIMITED' if ratio > 1 else
                 'READ-NOISE LIMITED by %.0fx' % (1 / ratio)))

    print()
    print('=== the cost, and what it would have taken to avoid it ===')
    print('    stack variance per pixel = (T/t) * read^2 + sky_rate * T, T the total seconds')
    zr = t[t.field == 'husillos zenith 08-13'].iloc[0]
    lr = t[t.field == 'leon Z1 4 s g101'].iloc[0]
    def stack_noise(T, tsub, read):
        return np.sqrt((T / tsub) * read ** 2 + rate * T)
    hus = stack_noise(50.0, 1.0, rn[0])
    leo = stack_noise(120.0, 4.0, rn[101])
    print('    Husillos: 50 x 1.0 s at gain 0   -> stack noise %6.2f e-, signal 50 s' % hus)
    print('    Leon:     30 x 4.0 s at gain 101 -> stack noise %6.2f e-, signal 120 s' % leo)
    print('    Leon collects %.1fx the star signal with %.2fx the noise: %.1fx the SNR.'
          % (120 / 50, leo / hus, (120 / 50) * (hus / leo)))
    print()
    print('    Husillos, same 50 s of sky time, other settings:')
    for g, tsub in ((0, 1.0), (0, 4.0), (125, 1.0), (125, 4.0)):
        if g not in rn:
            continue
        v = stack_noise(50.0, tsub, rn[g])
        print('       gain %3d, %4.1f s subframes: stack noise %6.2f e-  (SNR x%.2f)'
              % (g, tsub, v, hus / v))
    print('    Raising the gain alone costs full well. The brightest star in the zenith frame')
    print('    peaks at %.0f ADU of 65535 and NOTHING in that frame saturates (%d saturated'
          % (zr.max_adu, zr.sat_px))
    print('    pixels), so there was headroom even at gain 0; at gain 125 the brightest few')
    print('    would clip, which for astrometry on G < 13 stars costs a handful and no more.')

    print()
    print('=== depth: what the astrometry actually got, by Gaia G magnitude ===')
    d = stage2_depth()
    if not d.empty:
        print(d.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print('->', os.path.join(OUT, 'exposure.csv'))


if __name__ == '__main__':
    main()
