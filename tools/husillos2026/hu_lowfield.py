"""Can the 5.5-degree captures be solved properly, 23_37_17 above all?

Douglas, 2026-09-12: "23_37_17: can we try to solve this one again?  We probably have a very
good idea where it was pointing."

WE DO, AND IT ALREADY SOLVES.  At the zenith star-field preset it failed (35 centroids, 19
matched, one short of a quintic).  With the eclipse blocks' deep detection it solves:

    23_37_17   RA 176.990  Dec +23.284  roll 326.238   alt 5.70 deg   77 stars   rms 1.392 "

and that is pointing B, not pointing A -- the same field as 23_41_01 (Dec +23.283) and
23_42_43 (Dec +23.282), re-pointed 0.88 deg in RA, exactly the way pointing A was re-pointed
0.66 deg between its two captures.  So the plate solve is not what is wrong with it.

WHAT IS WRONG is the plate scale: +1804 ppm against the night zenith, with 23_41_01 at +1222
and 23_42_43 at +1565.  Three captures, two gains, both detection settings, all the same sign.
Record section 3s offers two candidates it could not separate:

    (a) REFRACTION -- the correction runs on assumed weather (926.5 hPa, 25 C, 35 %), and the
        differential across a +-2.9 deg field grows steeply toward the horizon;
    (b) MODEL TRANSFER -- the cubic and above are frozen from a zenith field at 85 deg, and a
        field compressed this hard may not be described by them at all.

THIS TOOL SEPARATES THEM, which is the useful form of "solve it again".  Each capture is fitted
three ways on the same deep stack:

    frozen      cubic+ from the zenith, quadratic free    -- the reduction of record (s2d_)
    free        the whole quintic free, nothing imported  -- kills (b) entirely
    free_nocorr free quintic, corrections OFF             -- kills (a) entirely

If the scale comes back to the zenith's under `free`, the transfer is the fault and the field
is fine.  If it stays 1500 ppm out under `free` but not under `free_nocorr`, the refraction
correction is putting it there.  If it is out under all three, the field itself is distorted
and no reduction will save it.  The 10 and 15 degree captures run as controls, because a test
that fails everywhere says nothing.

    .venv/Scripts/python.exe tools/husillos2026/hu_lowfield.py
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from hu_horizon_reduce import CAPTURES, OUT, PY, SITE, czip, midtime, results, run  # noqa: E402

ZEN_PS = 2.2059136
#: (tag, what it is) -- the three pointing-B captures, then the controls
TAGS = [('h10_g125b', '23_37_17, pointing B'),
        ('h10_g0_c', '23_41_01, pointing B'),
        ('h10_g125c_pre', '23_42_43, pointing B'),
        ('h10_g0', '23_31_59, pointing A (control)'),
        ('h10_g125d', '23_44_06, pointing C (control)')]

MODES = {
    # the reduction of record already exists under s2d_; the other two are built here
    'free': ['--set', 'distortion_fixed_coefficients=None'],
    'free_nocorr': ['--set', 'distortion_fixed_coefficients=None',
                    '--set', 'enable_corrections=False',
                    '--set', 'enable_corrections_ref=False'],
}


def folder_of(tag):
    for t, folder, name, gain, a, b in CAPTURES:
        if t == tag:
            return folder, name
    raise SystemExit('unknown tag ' + tag)


def altaz(d):
    lg = os.path.join(d, 'stage2.log')
    if not os.path.exists(lg):
        return float('nan')
    m = re.search(r'sky mean position alt/az: ([\d.]+)',
                  open(lg, encoding='utf-8', errors='replace').read())
    return float(m.group(1)) if m else float('nan')


def fit(tag, mode):
    """Returns the run's own results dict, or None."""
    if mode == 'frozen':
        d = os.path.join(OUT, 's2d_' + tag)
        return results(d), d
    d = os.path.join(OUT, 's2%s_%s' % ({'free': 'f', 'free_nocorr': 'fn'}[mode], tag))
    if not results(d):
        z = czip(os.path.join(OUT, 's1d_' + tag))
        if not z:
            return None, d
        folder, name = folder_of(tag)
        args = [PY, '-m', 'mee2024.cli', 'distortion', z, '--order', 'quintic',
                *MODES[mode],
                '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100']
        if mode == 'free':
            args += [*SITE, '--set', 'observation_time=' + midtime(folder, name)]
        else:
            args += ['--set', 'observation_date=2026-08-12', '--set', 'guess_date=False']
        run(args + ['--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    return results(d), d


def main():
    print('plate scale against the night zenith reference, %.7f "/px' % ZEN_PS)
    print('%-30s %-12s %6s %8s %12s %11s %9s'
          % ('capture', 'mode', 'stars', 'rms (")', 'ps ("/px)', 'vs zenith', 'alt'))
    for tag, what in TAGS:
        for mode in ('frozen', 'free', 'free_nocorr'):
            j, d = fit(tag, mode)
            if not j:
                print('%-30s %-12s  FAILED (see %s)' % (what, mode, d))
                continue
            ppm = (j['platescale (arcseconds/pixel)'] - ZEN_PS) / ZEN_PS * 1e6
            print('%-30s %-12s %6d %8.3f %12.7f %+10.0f %9.2f'
                  % (what if mode == 'frozen' else '', mode, j['#stars used'],
                     j['final rms error (arcseconds)'],
                     j['platescale (arcseconds/pixel)'], ppm, altaz(d)))
        print()
    print('Read DOWN each capture.  free returning to ~0 ppm means the frozen cubic-and-above')
    print('is the fault (model transfer).  free staying out while free_nocorr returns means the')
    print('refraction correction is.  All three out means the field itself, and no reduction')
    print('will save it.  Compare against the two controls before believing any of it.')


if __name__ == '__main__':
    main()
