"""Four plain aligned stacks of the SCI_ladder exposure tiers, as FITS, for reuse.

Douglas, 2026-09-02: "a set of 4 simple FITS stacks of the four exposure tiers ... No
centroiding, plate solve, etc necessary. Just an accurate stack to be used for another
purpose."

So this writes exactly that and nothing else: the RAW frames of each tier, aligned and
averaged. No coronal subtraction, no forbidden disk, no dark, no flat, no hot-pixel
removal, no detection, no solve. What comes out is the tier's own photons on the sky
frame, and every number a user might need to rescale it is in the header.

**The alignment is not re-derived, it is reused.** The reduction of record already measured
a shift for every frame of every tier (`step3_s0_v4/<tier>/centroid_data*.zip`, the
stacker's `alignment.shifts_px`), by matching stars on the coronal-subtracted frames. Those
same frames are these frames -- the preprocessing does not move anything -- so the shifts
transfer, and a stack built here is registered identically to the stack the science used.
Re-measuring alignment on raw frames would have been the wrong thing to do anyway: the
corona dominates a raw frame and a cross-correlation would lock onto it rather than onto
the stars, and the corona is not what the stars are fixed to.

The arithmetic is the stacker's own, reproduced from `add_img_to_stack`: shifts rounded to
whole pixels and applied with `np.roll`, edges filled with zero, and a per-pixel count of
contributing frames so the dithered border divides by what it actually received rather
than by the frame total. Integer rolling is deliberate -- it moves pixel values without
interpolating, so nothing is smoothed and the noise stays uncorrelated. The cost is up to
half a pixel of registration error, against a measured dither span of about two pixels.

A convention check runs first and is printed: the same code applied to the PREPROCESSED
frames must reproduce the pipeline's own stored `STACKED_FLOAT*.fit` for that tier. If the
roll convention or the sign were wrong, that check would not pass.

Output: D:/MEE2024 output/MEE_output/SCI_tier_stacks/SCI_<tier>_mean.fits, float32.
"""
import glob, json, os, re, sys, zipfile
import numpy as np
from astropy.io import fits
from astropy.time import Time

V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
OUT = r"D:/MEE2024 output/MEE_output/SCI_tier_stacks"
TIERS = {'0p1s': 0.1, '0p3s': 0.3, '0p6s': 0.6, '1p2s': 1.2}
os.makedirs(OUT, exist_ok=True)


def roll_fillzero(src, shift):
    """The stacker's own (mee2024/stacker_implementation.py)."""
    rolled = np.roll(src, shift=shift, axis=(0, 1))
    i, j = shift
    if j > 0:
        rolled[:, :j] = 0
    elif j < 0:
        rolled[:, j:] = 0
    if i > 0:
        rolled[:i, :] = 0
    elif i < 0:
        rolled[i:, :] = 0
    return rolled


def combine(paths, shifts):
    """Mean of the frames on the aligned grid, with per-pixel coverage."""
    acc = cnt = None
    for p, s in zip(paths, shifts):
        img = fits.getdata(p).astype(np.float64)
        if acc is None:
            acc = np.zeros(img.shape)
            cnt = np.zeros(img.shape, dtype=int)
        sh = (int(round(s[0])), int(round(s[1])))
        acc += roll_fillzero(img, sh)
        cnt += roll_fillzero(np.ones(img.shape, dtype=int), sh)
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0), cnt


def raw_for(preprocessed_path):
    """The raw frame a preprocessed one came from: the preprocessed name is
    '<block>_<original name>', and block is the capture folder on G:.

    The names are matched with spaces normalised to underscores, because the preprocessing
    step rewrote them: the duplicate-numbered frames on G: carry a space, as in
    'SCI_ladder_00001 (2).fits', which arrives here as '..._00001_(2).fits'.
    """
    base = os.path.basename(preprocessed_path)
    m = re.match(r'(\d\d_\d\d_\d\d)_(.+)$', base)
    assert m, base
    block, name = m.group(1), m.group(2)
    folder = glob.glob(os.path.join(RAW, '*', block))
    assert len(folder) == 1, (base, folder)
    hit = [p for p in glob.glob(os.path.join(folder[0], '*.fits'))
           if os.path.basename(p).replace(' ', '_') == name]
    assert len(hit) == 1, (base, hit)
    return hit[0]


print('%-6s %-9s %s' % ('tier', 'frames', 'check against the pipeline stack'), flush=True)
for tier, exp in TIERS.items():
    z = glob.glob(os.path.join(V4, tier, 'centroid_data*.zip'))
    assert z, tier
    res = json.load(zipfile.ZipFile(z[0]).open('results.txt'))
    pre = res['source_files']
    if isinstance(pre, str):
        pre = json.loads(pre.replace('\\\\', '/').replace("'", '"'))
    shifts = res['alignment']['shifts_px']
    assert len(pre) == len(shifts), (len(pre), len(shifts))

    # --- the convention check: our arithmetic on the preprocessed frames must reproduce
    # the pipeline's own stored stack for this tier
    ref = glob.glob(os.path.join(V4, tier, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit'))
    ours_pre, _ = combine(pre, shifts)
    if ref:
        theirs = fits.getdata(ref[0]).astype(np.float64)
        inner = (slice(50, -50), slice(50, -50))          # ignore the dithered border
        d = np.abs(ours_pre[inner] - theirs[inner])
        # The comparison is on the BULK of the frame, not the maximum, because the two are
        # expected to differ at isolated pixels: the pipeline excludes dark-free hot pixels
        # from its stack and this one keeps everything. Measured on the 0.1 s tier (46
        # frames, the most statistics for that search) that is 1081 single pixels, 0.004 %
        # of the frame, every one of them isolated and fully covered. The other three tiers
        # differ nowhere -- too few frames for the search to fire. So the bulk agreement is
        # what tests the roll convention, and the isolated count is reported, not failed.
        bulk = float(np.percentile(d, 99.99))
        n_hot = int((d > 1.0).sum())
        agree = bulk < 1.0
        check = ('bulk agrees to %.3g ADU; %d isolated pixels differ (the pipeline\'s '
                 'hot-pixel mask, not applied here)' % (bulk, n_hot))
    else:
        check, agree, n_hot = 'no stored stack to compare', False, -1

    raws = [raw_for(p) for p in pre]
    stack, cnt = combine(raws, shifts)
    hdrs = [fits.getheader(p) for p in raws]
    t0 = [Time(h['DATE-OBS'], scale='utc') for h in hdrs]
    mid = Time(np.mean([t.jd for t in t0]) + exp/2/86400.0, format='jd', scale='utc')
    sh = np.array([[round(s[0]), round(s[1])] for s in shifts])

    hdr = fits.Header()
    hdr['OBJECT'] = ('SCI_ladder %s stack' % tier, 'Leon 2026 eclipse field, one exposure tier')
    hdr['MEE2024'] = ('tier stack', 'built by tools/step3_tier_stacks.py')
    hdr['EXPTIME'] = (exp, 'seconds, per frame (the folder is the truth, not EXPTIME)')
    hdr['NFRAMES'] = (len(raws), 'frames averaged')
    hdr['EXPTOTAL'] = (round(exp*len(raws), 3), 'seconds of total integration')
    hdr['COMBINE'] = ('mean', 'multiply by NFRAMES for the sum')
    hdr['DATE-OBS'] = (min(t.isot for t in t0), 'earliest frame start, UTC')
    hdr['DATE-AVG'] = (mid.isot, 'mean frame mid-exposure, UTC')
    hdr['ALIGNSRC'] = (os.path.basename(z[0])[:68], 'alignment shifts reused from this archive')
    hdr['ALIGNMAX'] = (float(np.abs(sh).max()), 'largest applied shift, whole px')
    hdr['DITHERPX'] = (float(res['alignment'].get('dither_span_px', np.nan)),
                       'dither span measured by the stacker, px')
    hdr['CALIB'] = ('none', 'no dark, no flat, no hot-pixel removal')
    hdr['HOTKEPT'] = (n_hot, 'hot pixels the pipeline masks and this stack retains')
    hdr['PROCESS'] = ('none', 'raw frames: no coronal subtraction, no mask, no disk')
    hdr['SITELAT'] = (42.740470, 'degrees N')
    hdr['SITELONG'] = (-5.613780, 'degrees E')
    hdr['SITEELEV'] = (1101.0, 'metres')
    hdr['PLTSCALE'] = (2.2054043, 'arcsec/px, CAL_piLeo canonical (not fitted here)')
    hdr.add_history('raw frames aligned by the shifts measured in %s' % os.path.basename(z[0]))
    hdr.add_history('shifts rounded to whole pixels and applied with numpy.roll, edges zero')
    hdr.add_history('per-pixel frame count used as the divisor, so the dithered border is')
    hdr.add_history('averaged over the frames that actually covered it')
    hdr.add_history('no centroiding, no plate solve, no calibration frames applied')
    path = os.path.join(OUT, 'SCI_%s_mean.fits' % tier)
    fits.writeto(path, stack.astype(np.float32), hdr, overwrite=True)

    print('%-6s %-9d %s%s' % (tier, len(raws), check, '' if agree else '   <-- CHECK FAILED'),
          flush=True)
    print('        -> %s   %.1f s total, mid %s, max shift %d px, full coverage on %.1f%% '
          'of the frame, peak %.0f ADU'
          % (os.path.basename(path), exp*len(raws), mid.isot[11:19], np.abs(sh).max(),
             100.0*(cnt == len(raws)).mean(), stack.max()), flush=True)

print('\nstacks ->', OUT, flush=True)
