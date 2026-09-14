"""How many of the eclipse field's detections are real stars?  Match them at the known pointing.

Stage 1 finds 304 centroids in the trimmed science field and the blind solver cannot identify
it.  That is not the same as "there are no stars", and the question Douglas asked -- can stars be
located in the eclipse fields -- deserves the direct answer rather than the solver's verdict.

The pointing is not unknown.  The Sun's apparent place at 18:29:43 UTC from Husillos
(+42.09293, -4.52702, 743 m) is **RA 142.107, Dec +14.909**, it sits at about (4600, 3400) px,
and the plate scale is the zenith field's 2.2064 "/px.  So the only genuinely free parameters are
the camera roll and a small translation, and those can be voted for: project the catalogue about
the Sun, rotate by a trial roll, and histogram the implied translation of every
centroid-to-catalogue pair.  A real match is a sharp spike in that histogram; noise is flat.

**One trap, which cost the first attempt.**  A roll error of 0.25 deg swings a star at the
frame's half-diagonal of 5756 px through 5756 * 0.00436 = **25 px**, so a translation vote binned
at 4 px only adds up for stars within ~900 px of the rotation centre: scanning roll at 0.25 deg
over the whole frame finds nothing but noise, which is exactly what it did (7 matches against a
chance level of 0.65).  So the scan is done in two stages -- coarse roll on the sources near the
centre, where a 0.25 deg error is under 5 px, then a fine refinement on the whole frame.

**A second trap, in the source list.**  Stage 1 at cell 2's eclipse settings returns 304
centroids of which 146 have an area of exactly 2 px, while the stacked image itself holds **1720
sources of >= 4 px above 12 sigma** -- against 1661 catalogue stars in the frame.  The detections
are not the limit; the settings are.  So this tool builds its own list from the stack, and
reports the stage-1 list beside it.

Two things this settles that the solver could not:

  * **how many detections are stars** -- against 1676 Gaia G < 13 stars in the 5.87 x 3.91 deg
    field (73 per square degree at galactic b = +41);
  * **which of them are hot pixels**.  146 of the 304 have an area of exactly 2 px, the minimum
    `min_area` allows, and Husillos has no darks, so the stack carries every hot pixel it ever
    had (docs/HUSILLOS2026_ZENITH.md section 7d).  A hot pixel cannot match the sky, so the
    match rate per area bin measures the contamination directly.

Deflection and refraction are not modelled here and do not need to be: both are under 2 " = 1 px
against a match tolerance of 5 px.  Proper motion likewise -- the catalogue epoch is 2016.0 and
the frames are 2026.6, and a typical 10 mas/yr carries 0.1 ".

**THIS TOOL DOES NOT WORK YET, AND SAYS SO RATHER THAN RETURNING A NULL.**  Run on the ZENITH
stack -- a field stage 1 solved blind, at RA 281.7427, Dec +50.2092, roll 325.902 -- it fails to
recover the known answer: the roll histogram peaks at 34.0 deg with a peak-to-99th-percentile
ratio of 1.05, which is flat.  A tool that cannot find a field it has been given the answer to
cannot be believed when it finds nothing in a field it has not, so `main` runs that control
FIRST and refuses to report on the eclipse fields unless it passes.  (Measured 2026-09-10.  The
likely causes are that the brightest detections are saturated or blended and so are not the
catalogue's brightest, and that a 6 px tolerance on pair separations is smaller than the tens of
pixels of distortion across an 11 000 px baseline.  Neither is fixed here.)

**The finding this tool WAS able to establish** does not depend on the matching, only on
counting, and stands: of 1715 sources of >= 4 px above 12 sigma in the trimmed science stack,
**1697 lie inside 4 R_sun** at 330-910 per square degree, against the 73 per square degree the
catalogue holds there.  The inner field is coronal residue, not stars.  Beyond 4 R_sun there are
18 at 12 sigma and 53 at 6 sigma, and whether THOSE are stars is exactly what the broken matcher
cannot say.

  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_match.py [stack_name]
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

OUT = r"F:/MEE_output/husillos2026/eclipse"
CAT = r"C:/Users/dpesm/AppData/Local/MEE2024/MEE2024/catalogues/gaia_dr3_g13"

#: the science field's own numbers
PS = 2.2064323            # "/px, from the zenith field's free quintic fit
NX, NY = 9576, 6388
SUN_PX, SUN_PY = 4600.0, 3400.0
UTC = '2026-08-12T18:29:43.0'
SITE = dict(lat=42.09293, lon=-4.52702, height=743.0)

ROLL_STEP = 0.25          # deg, the coarse scan
FINE_STEP = 0.005         # deg, the refinement
BIN_PX = 4.0              # translation-vote bin
TOL_PX = 5.0              # final match tolerance
#: radius from the frame centre for the coarse scan: 0.25 deg of roll must stay inside BIN_PX
COARSE_R = 900.0
MIN_AREA = 4              # a 2 px source is a hot pixel, and this stack has no dark
NSIGMA = 12.0


def sun_radec(utc=UTC):
    from astropy.coordinates import get_sun
    from astropy.time import Time
    s = get_sun(Time(utc, scale='utc'))
    return float(s.ra.deg), float(s.dec.deg)


def catalogue_near(ra0, dec0, radius_deg=4.0):
    """Gaia G < 13 within `radius_deg`, as (ra deg, dec deg, G).

    The stored arrays are RADIANS and sorted by declination, which is what `dec_index.npy` is
    for; reading them as degrees returns an empty cone and no error at all.
    """
    ra = np.load(os.path.join(CAT, 'ra.npy'), mmap_mode='r')
    dec = np.load(os.path.join(CAT, 'dec.npy'), mmap_mode='r')
    mag = np.load(os.path.join(CAT, 'mag.npy'), mmap_mode='r')
    d0 = np.radians(dec0)
    r = np.radians(radius_deg)
    i0, i1 = np.searchsorted(np.asarray(dec), [d0 - r, d0 + r])
    sr = np.degrees(np.asarray(ra[i0:i1]))
    sd = np.degrees(np.asarray(dec[i0:i1]))
    sm = np.asarray(mag[i0:i1])
    cosd = np.cos(d0)
    dx = ((sr - ra0 + 180) % 360 - 180) * cosd
    keep = np.hypot(dx, sd - dec0) < radius_deg
    return sr[keep], sd[keep], sm[keep]


def gnomonic(ra, dec, ra0, dec0):
    """Tangent-plane offsets in arcsec about (ra0, dec0); xi east, eta north."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def vote(cx, cy, sx, sy, bin_px=BIN_PX):
    """Peak of the implied-translation histogram, and how many pairs voted for it."""
    dx = (cx[:, None] - sx[None, :]).ravel()
    dy = (cy[:, None] - sy[None, :]).ravel()
    keep = (np.abs(dx) < NX) & (np.abs(dy) < NY)
    dx, dy = dx[keep], dy[keep]
    if dx.size == 0:
        return 0, 0.0, 0.0
    ix = np.floor(dx / bin_px).astype(np.int64)
    iy = np.floor(dy / bin_px).astype(np.int64)
    key = (ix - ix.min()) * (iy.max() - iy.min() + 1) + (iy - iy.min())
    cnt = np.bincount(key)
    best = int(cnt.argmax())
    m = key == best
    return int(cnt[best]), float(dx[m].mean()), float(dy[m].mean())


def sources_from_stack(name):
    """Detect on the stacked image directly: >= MIN_AREA connected pixels above NSIGMA."""
    from astropy.io import fits
    from scipy import ndimage
    f = sorted(glob.glob(os.path.join(OUT, 's1_' + name, 'CENTROID_OUTPUT*',
                                      'STACKED_FLOAT*.fit')))
    if not f:
        return None, None, None
    d = fits.getdata(f[0]).astype(np.float32)
    sub = d[::7, ::7]
    bg = float(np.median(sub))
    sig = float(1.4826 * np.median(np.abs(sub - bg)))
    mask = d > bg + NSIGMA * sig
    lab, n = ndimage.label(mask)
    idx = np.arange(1, n + 1)
    sz = ndimage.sum(mask, lab, idx)
    keep = idx[(sz >= MIN_AREA) & (sz <= 2000)]
    cen = np.array(ndimage.center_of_mass(d - bg, lab, keep))
    peak = np.asarray(ndimage.maximum(d, lab, keep), dtype=float) - bg
    return cen[:, 1], cen[:, 0], peak


def match(name='sn2_trimmed', maglim=13.0, source='stack'):
    if source == 'stack':
        cx, cy, peak = sources_from_stack(name)
        if cx is None:
            print('%s: no stacked image' % name)
            return
        order = np.argsort(-peak)
        cx, cy, peak = cx[order], cy[order], peak[order]
        d = pd.DataFrame(dict(px=cx, py=cy))
        d['area (pixels)'] = np.nan
    else:
        z = glob.glob(os.path.join(OUT, 's1_' + name, 'centroid_data*.zip'))
        if not z:
            print('%s: no stage-1 output' % name)
            return
        d = pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv'))
        cx, cy = d['px'].to_numpy(), d['py'].to_numpy()
    ra0, dec0 = sun_radec()
    sr, sd, sm = catalogue_near(ra0, dec0)
    keep = sm < maglim
    xi, eta = gnomonic(sr[keep], sd[keep], ra0, dec0)
    smag = sm[keep]
    print('%s: %d centroids; %d catalogue stars G < %.1f within 4 deg of RA %.4f Dec %+.4f'
          % (name, len(cx), keep.sum(), maglim, ra0, dec0))

    # STAGE A: coarse roll, on the sources near the frame centre only, so that a 0.25 deg
    # roll error stays inside the 4 px vote bin (see the module docstring).
    cnear = np.hypot(cx - NX / 2, cy - NY / 2) < COARSE_R
    snear = np.hypot(xi, eta) / PS < COARSE_R
    best = None
    for parity in (+1, -1):
        for roll in np.arange(0, 360, ROLL_STEP):
            t = np.radians(roll)
            u = parity * xi[snear] / PS
            v = eta[snear] / PS
            sx = u * np.cos(t) - v * np.sin(t)
            sy = u * np.sin(t) + v * np.cos(t)
            n, tx, ty = vote(cx[cnear], cy[cnear], sx, sy)
            if best is None or n > best[0]:
                best = (n, roll, parity, tx, ty)
    n, roll, parity, tx, ty = best
    print('  coarse: %d votes at roll %.2f deg, parity %+d, from %d central sources and %d stars'
          % (n, roll, parity, cnear.sum(), snear.sum()))

    # STAGE B: refine the roll on the WHOLE frame, where the leverage is
    bestf = None
    for r in np.arange(roll - ROLL_STEP, roll + ROLL_STEP + 1e-9, FINE_STEP):
        t = np.radians(r)
        u = parity * xi / PS
        v = eta / PS
        sx = u * np.cos(t) - v * np.sin(t)
        sy = u * np.sin(t) + v * np.cos(t)
        nn, tx2, ty2 = vote(cx, cy, sx, sy)
        if bestf is None or nn > bestf[0]:
            bestf = (nn, r, tx2, ty2)
    n, roll, tx, ty = bestf
    print('  refined: %d votes at roll %.3f deg, Sun at (%.0f, %.0f) px' % (n, roll, tx, ty))

    # the finished match at that solution
    t = np.radians(roll)
    u = parity * xi / PS
    v = eta / PS
    sx = u * np.cos(t) - v * np.sin(t) + tx
    sy = u * np.sin(t) + v * np.cos(t) + ty
    inframe = (sx > 0) & (sx < NX) & (sy > 0) & (sy < NY)
    dist = np.hypot(cx[:, None] - sx[None, :], cy[:, None] - sy[None, :])
    near = dist.min(axis=1)
    which = dist.argmin(axis=1)
    hit = near < TOL_PX
    print('  %d of %d catalogue stars fall in the frame' % (inframe.sum(), len(sx)))
    print('  %d of %d centroids match a catalogue star within %.0f px (%.0f %%)'
          % (hit.sum(), len(cx), TOL_PX, 100 * hit.mean()))
    exp = inframe.sum() * np.pi * TOL_PX ** 2 / (NX * NY) * len(cx)
    print('  chance level at this tolerance: %.1f matches' % exp)
    if hit.sum() >= 3:
        print('  match residual: median %.2f px, rms %.2f px'
              % (np.median(near[hit]), np.sqrt((near[hit] ** 2).mean())))
        print('  matched stars run G %.2f to %.2f (median %.2f)'
              % (smag[which[hit]].min(), smag[which[hit]].max(),
                 np.median(smag[which[hit]])))
        print()
        print('  by detection area -- a hot pixel cannot match the sky:')
        area = d['area (pixels)'].to_numpy()
        for lo, hi in ((2, 3), (3, 4), (4, 6), (6, 10), (10, 999)):
            m = (area >= lo) & (area < hi)
            if m.sum():
                print('    area %2d-%-3d px: %4d detections, %4d matched (%3.0f %%)'
                      % (lo, hi - 1, m.sum(), (hit & m).sum(), 100 * hit[m].mean()))
    # a null: the same match against a sky rotated 90 deg away, which cannot be right
    t2 = np.radians((roll + 90) % 360)
    sx2 = u * np.cos(t2) - v * np.sin(t2) + tx
    sy2 = u * np.sin(t2) + v * np.cos(t2) + ty
    n2 = (np.hypot(cx[:, None] - sx2[None, :], cy[:, None] - sy2[None, :]).min(axis=1)
          < TOL_PX).sum()
    print()
    print('  NULL: the same test with the roll turned 90 deg gives %d matches, against %d.'
          % (n2, hit.sum()))
    return dict(name=name, roll=roll, parity=parity, matched=int(hit.sum()),
                centroids=len(cx), null=int(n2))


#: the zenith field stage 1 solved blind, which this tool must be able to recover before any
#: statement about a field it has NOT solved can be believed
CONTROL = dict(stack='with_f0', ra=281.7427, dec=50.2092, roll=325.902,
               root=r"F:/MEE_output/husillos2026/zenith_order")


def control_passes(verbose=True):
    """Can the matcher recover a field whose answer is known?  Returns True/False."""
    from astropy.io import fits
    from scipy import ndimage
    f = sorted(glob.glob(os.path.join(CONTROL['root'], 's1_' + CONTROL['stack'],
                                      'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit')))
    if not f:
        if verbose:
            print('CONTROL: the zenith stack is missing; cannot self-check')
        return False
    d = fits.getdata(f[0]).astype(np.float32)
    sub = d[::7, ::7]
    bg = float(np.median(sub))
    sig = float(1.4826 * np.median(np.abs(sub - bg)))
    m = d > bg + NSIGMA * sig
    lab, n = ndimage.label(m)
    idx = np.arange(1, n + 1)
    sz = ndimage.sum(m, lab, idx)
    keep = idx[(sz >= MIN_AREA) & (sz <= 500)]
    c = np.array(ndimage.center_of_mass(d - bg, lab, keep))
    pk = np.asarray(ndimage.maximum(d, lab, keep), dtype=float) - bg
    o = np.argsort(-pk)[:200]
    cx, cy = c[o, 1], c[o, 0]
    sr, sd, sm = catalogue_near(CONTROL['ra'], CONTROL['dec'])
    xi, eta = gnomonic(sr, sd, CONTROL['ra'], CONTROL['dec'])
    sx, sy = xi / PS, eta / PS
    inr = np.hypot(sx, sy) < 6000
    sx, sy, smg = sx[inr], sy[inr], sm[inr]
    oo = np.argsort(smg)[:600]
    sx, sy = sx[oo], sy[oo]
    t = np.radians(CONTROL['roll'])
    best = 0
    for parity in (+1, -1):
        u, v = parity * sx, sy
        rx = u * np.cos(t) - v * np.sin(t)
        ry = u * np.sin(t) + v * np.cos(t)
        nv, _, _ = vote(cx, cy, rx, ry)
        best = max(best, nv)
    ok = best >= 20
    if verbose:
        print('CONTROL: the zenith field at its KNOWN roll of %.3f deg votes %d pairs -- %s'
              % (CONTROL['roll'], best, 'PASS' if ok else 'FAIL'))
        if not ok:
            print('  The matcher cannot recover a field it has been given the answer to, so it')
            print('  cannot be believed when it finds nothing in one it has not. Nothing is')
            print('  reported. See the module docstring for what is and is not established.')
    return ok


if __name__ == '__main__':
    if not control_passes():
        sys.exit(1)
    names = sys.argv[1:] or ['sn2_trimmed']
    for nm in names:
        for src in ('stack', 'stage1'):
            print('--- %s, sources from the %s ---' % (nm, src))
            match(nm, source=src)
            print()
