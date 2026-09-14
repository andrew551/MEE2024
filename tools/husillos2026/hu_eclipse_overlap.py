"""The two eclipse fields' star lists: how many, how much they overlap, and the annotated masters.

Douglas, 2026-09-10: *"So the gain zero exposures saw 55 stars and the gain 125 exposures saw 68?
What was the overlap between the two?"*

**First, what 55 and 68 are not.**  They are the PLATE SOLVER's verification counts -- how many
catalogue stars it could line up well enough to accept the solution -- printed as "MATCH ACCEPTED
(nstars matched = 55)".  They are not the science star list, they are not fitted, and the solver
stops looking once it is convinced.  The number that matters is what stage 2 matches against the
catalogue afterwards, and that is what this measures.

The overlap is then taken on **Gaia source ids**, not on positions, because the two captures are
the same sky and a positional match would beg the question.  A 19-digit id never goes near a
float here (`tests/test_star_id_handling.py`): the ids are read as strings and intersected as
strings.

The charts follow `tools/matrix_station2/s2_charts_record.py` section 4 -- the arcsinh-stretched
master, yellow circles on the matched stars, the 2 R_sun circle dashed in cyan, and the count in
the legend -- so that cell 4's masters can be read beside cells 1, 2 and 3 without translation.
Both the stretch and the sky-to-sensor affine come from `tools/record_charts.py`
(`arcsinh_stretch`, `SkyFrame`), never a private copy: `tests/test_record_charts.py` fails a tool
that carries its own, and four such copies diverged four ways in one week of 2026-09.

**The Sun is placed by the astrometry, not by image morphology.** A first attempt looked for the
occulted disk in the stack and put the 2 R_sun circle 400 px from where it belonged, because the
occulter's fill value (5276.8 ADU here) is not the darkest thing in a coronal-subtracted frame.
`SkyFrame.from_stars` fits the affine on the matched stars and the Sun's apparent place at the
capture's mid-time goes through it forwards -- which also makes the circle a check on the
solution rather than a decoration.

  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_overlap.py stage2
  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_overlap.py report
  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_overlap.py charts
"""
import glob
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/husillos2026/eclipse"

PS = 2.2064323            # "/px
R_SUN_AS = 958.2
NX, NY = 9576, 6388

#: (stack, label for the chart, gain)
FIELDS = [('sn2_darkall', 'the Sn2 science field, 101 x 0.315 s at gain 0', 0),
          ('sun_dark', 'the Sun capture in totality, 126 x 0.315 s at gain 125', 125)]

#: Corrections OFF, as every diagnostic fit in this cell has been: the site is known
#: (+42.09293, -4.52702, 743 m) but its temperature, pressure and humidity are not, and inventing
#: weather to correct an 8.6 deg altitude is worse than leaving it off and saying so. THE RECORD
#: REDUCTION MUST TURN THEM ON -- nothing at that altitude is refraction-safe. This fit exists to
#: count stars and draw them, not to measure anything.
NOSITE = ['--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
          '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False']
#: The same fit WITH refraction on, as a clearly-labelled sensitivity. The Sun was at 8.6 deg
#: altitude, z = 81.4 deg, where R = k tan z is 384 " and its SECOND derivative across the field
#: is 2k sec^2 z tan z = 33 200 "/rad^2 -- about 19 " of quadratic distortion over a 3.9 deg
#: half-field. A linear fit absorbs the shear; nothing absorbs that. Temperature, pressure and
#: humidity are NOT known for this site: 926.5 hPa is the standard atmosphere at 743 m and the
#: rest are ordinary August evening values, so this is a sensitivity and not a measurement.
#: The record reduction needs the real weather.
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_temp=25.0', '--set', 'observation_humidity=0.35',
        '--set', 'observation_wavelength=0.55']
MIDTIME = {'sn2_darkall': '18:29:59', 'sun_dark': '18:29:20'}

GATES = ('2.0', '0.5')
ORDER = 'cubic'           # free, and low: ~100 centroids will not carry a quintic


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def czip(name):
    z = glob.glob(os.path.join(OUT, 's1_' + name, 'centroid_data*.zip'))
    return z[0] if z else None


def s2dir(name, gate, corr=False):
    return os.path.join(OUT, 's2_%s_g%s%s' % (name, gate.replace('.', 'p'),
                                              '_corr' if corr else ''))


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def matched(d):
    """The catalogue-matched star table, with ids as STRINGS. Never a float (19 digits)."""
    r = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not r:
        return None
    t = pd.read_csv(r[0], dtype={'ID': str})
    t['ID'] = t['ID'].astype(str).str.strip()
    return t


def do_stage2():
    for name, _lab, _g in FIELDS:
        z = czip(name)
        if not z:
            print('%-13s no stage-1 output' % name)
            continue
        for gate, corr in [(g, c) for c in (False, True) for g in GATES]:
            d = s2dir(name, gate, corr)
            os.makedirs(d, exist_ok=True)
            extra = (SITE + ['--set', 'observation_time=' + MIDTIME[name]]) if corr else NOSITE
            if not results(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', ORDER,
                     '--set', 'distortion_fixed_coefficients=None',
                     '--set', 'distortion_fit_tol=%s' % gate,
                     '--set', 'max_star_mag_dist=13',
                     '--set', 'rough_match_threshhold=36', *extra,
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            j = results(d)
            print('%-13s gate %-4s %-14s %s' % (name, gate,
                                                'refraction ON' if corr else 'corrections off',
                                          '%4d stars  rms %.4f "  ps %.6f'
                                          % (j['#stars used'], j['final rms error (arcseconds)'],
                                             j['platescale (arcseconds/pixel)'])
                                          if j else 'FAILED'), flush=True)


def fitted(d):
    """The matched-and-fitted stars, with sky coordinates: CATALOGUE_MATCHED_ERRORS minus
    the outliers the fit rejected."""
    r = glob.glob(os.path.join(d, '**', 'CATALOGUE_MATCHED_ERRORS.csv'), recursive=True)
    if not r:
        return None
    t = pd.read_csv(r[0], dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    return t[~t['flag_is_outlier'].astype(bool)].copy()


def sun_pixels(t, mid_utc):
    """Where the Sun sits on the sensor, from the astrometry rather than from the image.

    `SkyFrame.from_stars` (tools/record_charts.py) fits the sky-to-sensor affine on the matched
    stars; the Sun's apparent place at `mid_utc` then goes through it forwards.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    from record_charts import SkyFrame
    from astropy.coordinates import get_sun
    from astropy.time import Time
    sf = SkyFrame.from_stars(t['RA(catalog)'].to_numpy(), t['DEC(catalog)'].to_numpy(),
                             t['px'].to_numpy(), t['py'].to_numpy(), PS)
    sun = get_sun(Time(mid_utc, scale='utc'))
    X = (sun.ra.deg - sf.ra0) * sf.cos0
    Y = sun.dec.deg - sf.de0
    px = sf.ax_[0] * X + sf.ax_[1] * Y + sf.ax_[2]
    py = sf.ay_[0] * X + sf.ay_[1] * Y + sf.ay_[2]
    return float(px), float(py)


def do_report(gate='2.0', corr='1'):
    corr = str(corr) not in ('0', 'False', 'false')
    print('=' * 100)
    print('WHAT STAGE 2 MATCHED  (free %s, gate %s ", %s)'
          % (ORDER, gate, 'refraction ON -- a SENSITIVITY, the weather is assumed'
             if corr else 'corrections off'))
    print('=' * 100)
    tabs = {}
    for name, lab, g in FIELDS:
        j, t = results(s2dir(name, gate, corr)), matched(s2dir(name, gate, corr))
        if j is None or t is None:
            print('%-13s not fitted' % name)
            continue
        tabs[name] = t
        z = czip(name)
        n1 = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))['n_centroids']
        print('%-13s gain %3d  %4d centroids -> %4d catalogue-matched  rms %.4f "  G %.2f-%.2f'
              % (name, g, n1, len(t), j['final rms error (arcseconds)'],
                 t['magV'].min(), t['magV'].max()))
    if len(tabs) < 2:
        return
    a, b = tabs[FIELDS[0][0]], tabs[FIELDS[1][0]]
    ia, ib = set(a['ID']), set(b['ID'])
    both = ia & ib
    print()
    print('=' * 100)
    print('OVERLAP, on Gaia source ids (strings, never floats)')
    print('=' * 100)
    print('  gain 0   only : %4d' % len(ia - ib))
    print('  BOTH          : %4d   (%.0f %% of the gain-0 list, %.0f %% of the gain-125 list)'
          % (len(both), 100 * len(both) / max(len(ia), 1), 100 * len(both) / max(len(ib), 1)))
    print('  gain 125 only : %4d' % len(ib - ia))
    print('  union         : %4d' % len(ia | ib))
    if both:
        ma = a.set_index('ID').loc[sorted(both)]
        mb = b.set_index('ID').loc[sorted(both)]
        print()
        print('  the %d shared stars: G %.2f-%.2f, median %.2f'
              % (len(both), ma['magV'].min(), ma['magV'].max(), ma['magV'].median()))
        d = np.hypot(ma['px'].to_numpy() - mb['px'].to_numpy(),
                     ma['py'].to_numpy() - mb['py'].to_numpy())
        print('  their pixel positions differ by %.2f px median (%.2f " ) between the two'
              % (np.median(d), np.median(d) * PS))
        print('  captures -- the same stars, 57 s apart, at different gains.')
    for nm, s in (('gain 0', ia - ib), ('gain 125', ib - ia)):
        t = (a if nm == 'gain 0' else b).set_index('ID')
        if s:
            print('  %-9s only: G %.2f-%.2f, median %.2f'
                  % (nm, t.loc[sorted(s), 'magV'].min(), t.loc[sorted(s), 'magV'].max(),
                     t.loc[sorted(s), 'magV'].median()))


def do_charts(gate='2.0', corr='1'):
    corr = str(corr) not in ('0', 'False', 'false')
    """The annotated masters, in the form cells 1-3 already use."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from astropy.io import fits
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    from record_charts import arcsinh_stretch      # the stretch every annotated master uses
    dest = os.path.join(OUT, 'charts')
    os.makedirs(dest, exist_ok=True)
    for name, lab, g in FIELDS:
        t = fitted(s2dir(name, gate, corr))
        if t is None or not len(t):
            print('%s: not fitted' % name)
            continue
        f = sorted(glob.glob(os.path.join(OUT, 's1_' + name, 'CENTROID_OUTPUT*',
                                          'STACKED_FLOAT*.fit')))
        img = fits.getdata(f[-1]).astype(np.float32)
        disp = arcsinh_stretch(img)
        sx, sy = sun_pixels(t, '2026-08-12T' + MIDTIME[name])
        z = czip(name)
        nfr = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))['#frames stacked']
        fig, ax = plt.subplots(figsize=(11.5, 8))
        ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
        for x0, y0 in zip(t['px'].to_numpy(), t['py'].to_numpy()):
            ax.add_patch(Circle((x0, y0), 95, fill=False, color='yellow', lw=1.1))
        ax.add_patch(Circle((sx, sy), 2 * R_SUN_AS / PS, fill=False, color='cyan',
                            lw=1.4, ls='--'))
        ax.legend(handles=[plt.Line2D([], [], color='yellow',
                                      label='matched and fitted (%d)' % len(t)),
                           plt.Line2D([], [], color='cyan', ls='--', label='2 R$_\\odot$')],
                  fontsize=9, loc='upper left', bbox_to_anchor=(0.0, -0.09),
                  borderaxespad=0, frameon=True)
        ax.set_title('%s (%s frames, occulted and coronal-subtracted) \u2014 Husillos 2026'
                     % (lab, nfr), fontsize=11)
        ax.set_xlabel('px')
        ax.set_ylabel('py')
        fig.subplots_adjust(bottom=0.20)
        p = os.path.join(dest, 'master_%s_annotated.png' % name)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print('-> %s   (Sun at (%.0f, %.0f) px from the astrometry)' % (p, sx, sy))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'stage2': do_stage2, 'report': do_report, 'charts': do_charts}[cmd](*sys.argv[2:])
