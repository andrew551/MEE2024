"""Four plain aligned stacks of the SCI_ladder exposure tiers, as FITS, for reuse -- raw,
or calibrated with the library darks and flats.

Douglas, 2026-09-02: "a set of 4 simple FITS stacks of the four exposure tiers ... No
centroiding, plate solve, etc necessary. Just an accurate stack to be used for another
purpose." Then: "let's create another set using the dark frames and flat frames. See
G:\\MEE_output\\Library 1."

So this writes exactly that and nothing else, in two modes:

  python tools/step3_tier_stacks.py               -> SCI_tier_stacks/            (raw)
  python tools/step3_tier_stacks.py --calibrated  -> SCI_tier_stacks_calibrated/ (dark, flat)

RAW: the frames of each tier aligned and averaged. No coronal subtraction, no forbidden
disk, no dark, no flat, no hot-pixel removal, no detection, no solve.

CALIBRATED: the same, with each frame corrected as `(frame - master dark) / master flat`
BEFORE it is rolled -- the pipeline's own arithmetic (`open_img_and_preprocess`), and the
order matters: a dark and a flat are fixed to the detector while the roll follows the
sky, so they must be applied on the unshifted frame. The masters are chosen by the
program's own matcher (`calibration.resolve_for_field`), so the selection is the one the
pipeline would make and its notes are written into the header. The library holds an
exact-exposure gain-0 dark for every tier (49-50 frames each, taken 2026-08-11, the night
before, on a +10 C setpoint) and one flat for this optical train (FRA500 + 0.7x reducer,
bin 1, 50 frames at 0.43 s, focus 17049 against the eclipse's 17170 -- inside the
matcher's 200-step tolerance). Nothing is scaled: the dark is the tier's own exposure, and
the flat is normalised to 1 by its own median, bias included, as the library builds it.

**The alignment is not re-derived, it is reused.** The reduction of record already measured
a shift for every frame of every tier (`step3_s0_v4/<tier>/centroid_data*.zip`, the
stacker's `alignment.shifts_px`), by matching stars on the coronal-subtracted frames. Those
same frames are these frames -- neither the preprocessing nor a dark/flat moves anything --
so the shifts transfer, and a stack built here is registered identically to the stack the
science used. Re-measuring alignment on raw frames would have been the wrong thing to do
anyway: the corona dominates a raw frame and a cross-correlation would lock onto it rather
than onto the stars, and the corona is not what the stars are fixed to.

The arithmetic is the stacker's own, reproduced from `add_img_to_stack`: shifts rounded to
whole pixels and applied with `np.roll`, edges filled with zero, and a per-pixel count of
contributing frames so the dithered border divides by what it actually received rather
than by the frame total. Integer rolling is deliberate -- it moves pixel values without
interpolating, so nothing is smoothed and the noise stays uncorrelated. The cost is up to
half a pixel of registration error, against a measured dither span of about two pixels.

Checks, printed per tier:
  * raw mode -- the same code applied to the PREPROCESSED frames must reproduce the
    pipeline's own stored `STACKED_FLOAT*.fit` for that tier. It does, to 1e-4 ADU, with
    isolated pixels differing where the pipeline's dark-free hot-pixel mask acted (1020 on
    the 0.1 s tier, none on the others) and this stack deliberately did not;
  * calibrated mode -- the background of the calibrated stack must equal the raw stack's
    background less the dark's level, divided by the flat there. Measured in a star-free
    corner; if the dark and flat were the wrong ones, or applied in the wrong order, this
    would not close.

Outputs float32, with every number a user might need to rescale in the header.
"""
import glob, json, os, re, sys, zipfile
import numpy as np
from astropy.io import fits
from astropy.time import Time

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)
from mee2024 import calibration                                          # noqa: E402

V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
LIB = r"G:/MEE_output/Library 1"
CALIBRATED = '--calibrated' in sys.argv
OUT = r"D:/MEE2024 output/MEE_output/SCI_tier_stacks" + ('_calibrated' if CALIBRATED else '')
RAW_OUT = r"D:/MEE2024 output/MEE_output/SCI_tier_stacks"
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


def combine(paths, shifts, prep=None):
    """Mean of the frames on the aligned grid, with per-pixel coverage. `prep` is applied
    to each frame BEFORE it is rolled -- where a detector-fixed correction belongs."""
    acc = cnt = None
    for p, s in zip(paths, shifts):
        img = fits.getdata(p).astype(np.float64)
        if prep is not None:
            img = prep(img)
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


CORNER = (slice(150, 550), slice(150, 550))      # star-free, well off the corona
print('mode: %s   ->   %s' % ('CALIBRATED (dark + flat from %s)' % LIB if CALIBRATED else 'RAW',
                              OUT), flush=True)
print('%-6s %-9s %s' % ('tier', 'frames', 'check'), flush=True)
for tier, exp in TIERS.items():
    z = glob.glob(os.path.join(V4, tier, 'centroid_data*.zip'))
    assert z, tier
    res = json.load(zipfile.ZipFile(z[0]).open('results.txt'))
    pre = res['source_files']
    if isinstance(pre, str):
        pre = json.loads(pre.replace('\\\\', '/').replace("'", '"'))
    shifts = res['alignment']['shifts_px']
    assert len(pre) == len(shifts), (len(pre), len(shifts))
    raws = [raw_for(p) for p in pre]
    hdrs = [fits.getheader(p) for p in raws]

    # ---- the masters, chosen by the program's own matcher, notes kept for the header
    prep, cal_notes, dark_h, flat_h, dark_med = None, [], None, None, float('nan')
    # The EXPTIME header lies at block boundaries (docs/STEP3_2026.md, "the ladder
    # syntax"): the first frame of this tier's list carries EXPTIME = 0.3 on 0.1 s
    # pixels, and fed to the matcher as-is it selected the 0.3 s dark -- caught by the
    # exposure assertion below on the first run. The FOLDER is the truth about exposure,
    # so the matcher is given the folder's value, and the lying headers are counted.
    n_hdr_lie = int(sum(abs(float(h.get('EXPTIME', -1)) - exp) > 1e-6 for h in hdrs))
    if CALIBRATED:
        light = calibration.read_cal_header(raws[0])
        light['EXPTIME'] = exp
        entries = calibration.read_index(LIB)
        dark_entry, dnote = calibration.match_dark(entries, light)
        flat_entry, fnote = calibration.match_flat(entries, light)
        assert dark_entry is not None, (tier, dnote)
        assert flat_entry is not None, (tier, fnote)
        cal_notes.append('dark %s (%s frames)%s' % (dark_entry['key'], dark_entry.get('n_frames', '?'),
                                                    ' -- ' + dnote if dnote else ''))
        cal_notes.append('flat %s (%s frames)%s' % (flat_entry['key'], flat_entry.get('n_frames', '?'),
                                                    ' -- ' + fnote if fnote else ''))
        dark = fits.getdata(dark_entry['master']).astype(np.float64)
        flat = fits.getdata(flat_entry['master']).astype(np.float64)
        dark_h, flat_h = fits.getheader(dark_entry['master']), fits.getheader(flat_entry['master'])
        dark_med = float(np.median(dark))
        assert abs(float(dark_h['EXPTIME']) - exp) < 1e-6, (tier, dark_h['EXPTIME'])
        assert dark.shape == flat.shape == fits.getdata(raws[0]).shape
        prep = lambda img, d=dark, f=flat: (img - d) / f        # the pipeline's own order

    # ---- the check
    if CALIBRATED:
        stack, cnt = combine(raws, shifts, prep)
        rawstack = os.path.join(RAW_OUT, 'SCI_%s_mean.fits' % tier)
        if os.path.exists(rawstack):
            r = fits.getdata(rawstack).astype(np.float64)
            expected = (np.median(r[CORNER]) - dark_med) / float(np.median(flat[CORNER]))
            got = float(np.median(stack[CORNER]))
            # relative tolerance: the flat is applied per frame before the roll and a
            # median is not linear, so the closure is to ~1e-4 of the level, not to a fixed
            # ADU (0.5 ADU on the 1.2 s tier's 3072 ADU sky tripped an absolute 0.5)
            agree = abs(got - expected) < max(0.5, 1e-3*abs(expected))
            check = ('corner sky: raw %.1f - dark %.1f, / flat %.4f = %.1f expected, %.1f got'
                     % (np.median(r[CORNER]), dark_med, np.median(flat[CORNER]), expected, got))
        else:
            check, agree = 'no raw stack to compare (run the raw mode first)', False
        n_hot = -1
    else:
        ref = glob.glob(os.path.join(V4, tier, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit'))
        ours_pre, _ = combine(pre, shifts)
        if ref:
            theirs = fits.getdata(ref[0]).astype(np.float64)
            inner = (slice(50, -50), slice(50, -50))          # ignore the dithered border
            d = np.abs(ours_pre[inner] - theirs[inner])
            # The comparison is on the BULK of the frame, not the maximum, because the two
            # are expected to differ at isolated pixels: the pipeline excludes dark-free hot
            # pixels from its stack and this one keeps everything. Measured on the 0.1 s
            # tier (46 frames, the most statistics for that search) that is ~1000 single
            # pixels, 0.004 % of the frame, every one isolated and fully covered. The other
            # three tiers differ nowhere -- too few frames for the search to fire.
            bulk = float(np.percentile(d, 99.99))
            n_hot = int((d > 1.0).sum())
            agree = bulk < 1.0
            check = ('bulk agrees to %.3g ADU; %d isolated pixels differ (the pipeline\'s '
                     'hot-pixel mask, not applied here)' % (bulk, n_hot))
        else:
            check, agree, n_hot = 'no stored stack to compare', False, -1
        stack, cnt = combine(raws, shifts)

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
    if CALIBRATED:
        hdr['CALIB'] = ('dark+flat', '(frame - master dark) / master flat, per frame, before the roll')
        hdr['CALLIB'] = (LIB[:68], 'mee2024-callib library the masters came from')
        hdr['DARKSET'] = (dark_entry['key'][:68],
                          'library entry matched by gain, exposure, camera, binning')
        hdr['DARKN'] = (int(dark_h.get('NCOMBINE', 0)), 'frames in the master dark')
        hdr['DARKMED'] = (round(dark_med, 3), 'ADU, median of the master dark (bias included)')
        hdr['DARKTEMP'] = (float(dark_h.get('CCD-TEMP', np.nan)), 'C, master dark sensor temperature')
        hdr['LGHTTEMP'] = (float(np.mean([h.get('CCD-TEMP', np.nan) for h in hdrs])),
                           'C, mean sensor temperature of the light frames')
        hdr['NHDRLIE'] = (n_hdr_lie, 'frames whose EXPTIME header disagrees with the folder')
        hdr['FLATSET'] = (flat_entry['key'][:68],
                          'library entry matched by camera, train, binning, focus')
        hdr['FLATN'] = (int(flat_h.get('NCOMBINE', 0)), 'frames in the master flat')
        hdr['FLATNORM'] = (float(flat_h.get('CALNORM', np.nan)), 'ADU the flat was divided by')
        hdr['FLATFOC'] = (int(float(flat_h.get('FOCUSPOS', 0))), 'focuser position of the flat')
        hdr['LGHTFOC'] = (int(float(hdrs[0].get('FOCUSPOS', 0))), 'focuser position of the lights')
        hdr['HOTKEPT'] = (-1, 'n/a: hot pixels are removed by the dark, not masked')
        hdr['PROCESS'] = ('dark+flat only', 'no coronal subtraction, no mask, no disk')
        for note in cal_notes:
            hdr.add_history(note[:72])
    else:
        hdr['CALIB'] = ('none', 'no dark, no flat, no hot-pixel removal')
        hdr['HOTKEPT'] = (n_hot, 'hot pixels the pipeline masks and this stack retains')
        hdr['PROCESS'] = ('none', 'raw frames: no coronal subtraction, no mask, no disk')
    hdr['SITELAT'] = (42.740470, 'degrees N')
    hdr['SITELONG'] = (-5.613780, 'degrees E')
    hdr['SITEELEV'] = (1101.0, 'metres')
    hdr['PLTSCALE'] = (2.2054043, 'arcsec/px, CAL_piLeo canonical (not fitted here)')
    hdr.add_history('raw frames aligned by the shifts measured in %s' % os.path.basename(z[0]))
    if CALIBRATED:
        hdr.add_history('each frame: (frame - master dark) / master flat, then rolled')
        hdr.add_history('pixels saturated in the raw frames (65535 ADU) stay saturated:')
        hdr.add_history('after the flat they sit near (65535 - dark)/flat, i.e. ~66000 ADU')
        hdr['SATRAW'] = (65535, 'ADU, full scale of the raw frames before calibration')
    hdr.add_history('shifts rounded to whole pixels and applied with numpy.roll, edges zero')
    hdr.add_history('per-pixel frame count used as the divisor, so the dithered border is')
    hdr.add_history('averaged over the frames that actually covered it')
    hdr.add_history('no centroiding, no plate solve')
    path = os.path.join(OUT, 'SCI_%s_mean.fits' % tier)
    fits.writeto(path, stack.astype(np.float32), hdr, overwrite=True)

    print('%-6s %-9d %s%s' % (tier, len(raws), check, '' if agree else '   <-- CHECK FAILED'),
          flush=True)
    print('        -> %s   %.1f s total, mid %s, max shift %d px, full coverage on %.1f%% '
          'of the frame, peak %.0f ADU'
          % (os.path.basename(path), exp*len(raws), mid.isot[11:19], np.abs(sh).max(),
             100.0*(cnt == len(raws)).mean(), stack.max()), flush=True)
    if CALIBRATED:
        print('        dark %s (%d frames, median %.1f ADU, %.1f C vs lights %.1f C); flat %s '
              '(%d frames, focus %d vs lights %d)'
              % (hdr['DARKSET'], hdr['DARKN'], dark_med, hdr['DARKTEMP'], hdr['LGHTTEMP'],
                 hdr['FLATSET'], hdr['FLATN'], hdr['FLATFOC'], hdr['LGHTFOC']), flush=True)

print('\nstacks ->', OUT, flush=True)
