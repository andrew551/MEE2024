"""Method 1 for cell 4: the eclipse blocks frozen against CalibS, with CalibS' scale imported.

Douglas, 2026-09-11.  This is the ladder's third rung, and it is the first time cell 4 can
run it at all -- until CalibS was reduced there was no same-day calibration field to import
from, and the night zenith's scale is 1326 ppm away and unusable (docs/HUSILLOS2026_ECLIPSE.md
section 3e).

THE PATHWAY, and it is the one Leon used for CAL_piLeo, rung for rung
(docs/STEP3_2026.md, the ladder table; tools/step3_master_vs_union.py:111):

    night zenith            free quintic                        -> 2635 stars, ps 2.2059136
    CalibS                  distortion_fixed_coefficients=QUADRATIC, free scale
                            (cubic and above frozen from the zenith: verified, 15 of 15
                            terms bit-identical)                -> 26 stars, ps 2.2029895
    eclipse blocks          distortion_fixed_coefficients=CONSTANT, free scale OFF
                            (linear and above frozen from CalibS, SCALE IMPORTED)

THE SCALE IMPORT IS NOT A SEPARATE SWITCH.  `distortion_fitter.py:538` decides it:

    'plate scale source': ('imported from the reference files'
                           if options['distortion_fixed_coefficients'] == 'constant'
                           and not options.get('distortion_free_scale')
                           else 'fitted on this field')

so `constant` AND `free_scale=False` must hold together or the scale is silently fitted
instead of imported.  This tool asserts on the run's own results that the import happened --
CLAUDE.md's rule about reading `fixed distortion order` back from the run rather than
believing the arguments that were passed.

WHAT TO EXPECT.  CalibS carries +-25.2 ppm of relative scale uncertainty, and this field's
lever is 0.0324 " of L per ppm (h = 34.24 R_sun^2, the largest in the matrix), so the import
alone contributes about +-0.82 " to L -- larger than Method 2's whole statistical error of
+-0.430 ".  Method 1 here is not an upgrade on Method 2; it is an independent route with a
different error budget, and the comparison is the point.

    .venv/Scripts/python.exe tools/husillos2026/hu_method1.py [stage2|stage3|report]
"""
import glob
import io
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from analysis_window import WINDOWS  # noqa: E402

WIN = WINDOWS['husillos2026']
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
HUS = r'F:\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'step3')
#: Which CalibS reduction to import from.  `s2_calibs_ecl` is the one detected the way the
#: science blocks were (cell 2's eclipse settings): 86 stars and +-18.7 ppm, against the first
#: run's zenith-preset 26 stars and +-25.2 ppm.  The zenith-preset run is kept reachable via
#: HU_CALIBS so the earlier Method 1 numbers stay reproducible.
CALIBS = os.path.join(HUS, 'calibs', os.environ.get('HU_CALIBS', 's2_calibs_ecl'))
#: Output prefix, DERIVED FROM THE CALIBRATION NAME so a swap can neither overwrite the other's
#: results nor silently reuse them.  The first version keyed only on "is it the zenith-preset
#: run", so the settled re-run landed in the eclipse run's directory, found results already
#: there and skipped -- reporting the OLD imported scale as though it were the new one.
TAGP = {'s2_calibs': 'm1_', 's2_calibs_ecl': 'm1e_',
        's2_calibs_ecl_settled': 'm1s_'}[os.path.basename(CALIBS)]

#: the two science blocks, as hu_step3 defines them
BLOCKS = [('gain0', os.path.join(HUS, 'eclipse', 's1_sn2_darkall'), '18:29:59'),
          ('gain125', os.path.join(HUS, 'eclipse', 's1_sun_dark'), '18:29:20')]

#: Husillos, corrections ON -- the Sun is at 8.5 deg and nothing here is refraction-safe.
#: The weather is ASSUMED (926.5 hPa is the standard atmosphere at 743 m, 25 C and 35 % are
#: ordinary August values) and is the largest unquantified term in this cell.
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0',
        '--set', 'observation_temp=25.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_humidity=0.35']


def run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def czip(d):
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return z[0] if z else None


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def calibs_zip():
    z = glob.glob(os.path.join(CALIBS, '**', 'distortion_data*.zip'), recursive=True)
    if not z:
        raise SystemExit('CalibS is not reduced: run hu_calibs.py stack first')
    return z[0]


def dzip(tag):
    z = glob.glob(os.path.join(OUT, TAGP + '%s' % tag, '**', 'distortion_data*.zip'),
                  recursive=True)
    return z[0] if z else None


def do_stage2():
    """Both blocks against CalibS at `constant`, scale imported."""
    cal = calibs_zip()
    calres = results(CALIBS)
    print('importing from CalibS: ps %.7f "/px +- %.1f ppm, %d stars, rms %.4f "'
          % (calres['platescale (arcseconds/pixel)'],
             calres['platescale_relative_uncertainty'] * 1e6,
             calres['#stars used'], calres['final rms error (arcseconds)']))
    print()
    for tag, src, tmid in BLOCKS:
        d = os.path.join(OUT, TAGP + '%s' % tag)
        if not results(d):
            run([PY, '-m', 'mee2024.cli', 'distortion', czip(src), '--order', 'quintic',
                 '--set', 'distortion_reference_files=' + cal,
                 # the ladder's eclipse rung: linear and above frozen from the calibration
                 # field, and with free_scale off this is what imports the plate scale
                 '--set', 'distortion_fixed_coefficients=constant',
                 '--set', 'distortion_free_scale=False',
                 '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                 '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
                 *SITE, '--set', 'observation_time=' + tmid,
                 '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
        j = results(d)
        if not j:
            print('%-8s stage 2 FAILED -- see %s' % (tag, os.path.join(d, 'stage2.log')))
            continue
        # read the pathway back from the run itself, never from the arguments passed
        src_note = j['plate scale source']
        ok = (src_note == 'imported from the reference files'
              and j['fixed distortion order'] == 'constant')
        print('%-8s %d stars, rms %.4f ", ps %.7f "/px'
              % (tag, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)']))
        print('         fixed distortion order: %s | plate scale source: %s   %s'
              % (j['fixed distortion order'], src_note,
                 'OK' if ok else '<-- THE IMPORT DID NOT HAPPEN'))
        if not ok:
            raise SystemExit('refusing to continue: this would not be Method 1')


def do_stage3():
    """Method 1 on each block, uncropped, two-witness applied as in hu_step3."""
    import pandas as pd
    import zipfile
    tabs = {}
    for tag, _s, _t in BLOCKS:
        z = dzip(tag)
        if not z:
            print('%s: no stage-2 output; run stage2 first' % tag)
            return
        zf = zipfile.ZipFile(z)
        n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
        t = pd.read_csv(zf.open(n), dtype={'ID': str})
        t.columns = [c.strip() for c in t.columns]
        t['ID'] = t['ID'].astype(str).str.strip()
        tabs[tag] = t
    both = set(tabs[BLOCKS[0][0]]['ID']).intersection(tabs[BLOCKS[1][0]]['ID'])
    print('two-witness: %s -> %d shared'
          % (' , '.join('%s %d' % (k, len(v)) for k, v in tabs.items()), len(both)))

    for tag, _s, _t in BLOCKS:
        t = tabs[tag]
        keep = t['ID'].isin(both)
        src = dzip(tag)
        dst = os.path.join(OUT, TAGP + 'witness_%s.zip' % tag)
        zin = zipfile.ZipFile(src)
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                    buf = io.StringIO()
                    t[keep].to_csv(buf, index=False)
                    data = buf.getvalue().encode('utf-8')
                zout.writestr(item, data)
        d = os.path.join(OUT, TAGP + 'method1_%s_witness' % tag)
        run([PY, '-m', 'mee2024.cli', 'eclipse', dst,
             '--set', 'eclipse_method=Method 1 & 2',
             '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
             # NOT cropped radially: Douglas, 2026-09-10
             '--set', 'limit_radial_sun_radii=False',
             '--set', 'remove_double_stars_eclipse=False',
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage3.log'))
        # newest, not first-sorted -- see the note in hu_union._stage3
        fs = sorted(glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True),
                    key=os.path.getmtime)
        if not fs:
            print('   %-8s stage 3 FAILED' % tag)
            continue
        for line in io.open(fs[-1], encoding='utf-8', errors='replace').read().splitlines():
            s = line.strip()
            if any(k in s for k in ('Method 1 results', 'Method 2 results',
                                    'number of stars', 'deflected star position rms')):
                print('   %-8s %s' % (tag, s.replace(u'\u00b1', ' +- ')
                                      .encode('ascii', 'replace').decode('ascii')[:140]))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stage2'
    os.makedirs(OUT, exist_ok=True)
    {'stage2': do_stage2, 'stage3': do_stage3}[cmd]()
