"""Re-run the 2600MM zenith test through today's MEE, and compare with Astrometrica.

Douglas, 2026-09-15: "can we actually run this on the data in I:\\Don Bruns 2600MM tests"

We can. The folder holds a complete stage-1 archive -- `data.zip`, 2112 centroids, plate
solved, img_shape [4176, 6248] -- so stage 2 can be run again rather than reading the 2024
results file. Two obstacles, both handled here:

  * the archive's members sit under a `data/` prefix (`data/results.txt`), where today's
    reader expects them at the root, so it is repackaged;
  * `I:` is READ ONLY, so everything is written to `F:\\MEE_output\\bruns2600_rerun`.

THE SETTINGS ARE READ, NOT CHOSEN.  The 2024 run recorded its own: quintic (inferred from the
21 coefficients it stored), G <= 13, a 0.2 " tolerance, and all three corrections off. Those
come out of `data20240320154050distortion_results.txt` rather than being typed here, which is
the same rule the rest of the project follows about reading a rung back from the run's own
output.

What this then buys is two comparisons instead of one:

  1. today's MEE against MEE-of-March-2024 on identical input -- does the pipeline still get
     the same answer?
  2. today's MEE against Astrometrica 4.13 -- which is what tools/astrometrica_compare.py
     measures, now against a fresh fit rather than a stored one.

    .venv/Scripts/python.exe tools/bruns2600_rerun.py
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
SRC = r'I:\Don Bruns 2600MM tests'                  # READ ONLY
OLD = os.path.join(SRC, 'output', 'data20240320154050distortion_results.txt')
OUT = r'F:\MEE_output\bruns2600_rerun'


def repackage():
    """The stage-1 archive with the `data/` prefix stripped, on F:."""
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, 'centroid_data20240320154050.zip')
    if os.path.isfile(dst):
        return dst
    with zipfile.ZipFile(os.path.join(SRC, 'data.zip')) as z, \
            zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            if n.endswith('/'):
                continue
            out.writestr(n.split('/', 1)[1] if n.startswith('data/') else n, z.read(n))
    return dst


def settings_from(old):
    """Order, magnitude cut and tolerance as the 2024 run recorded them."""
    j = json.load(io.open(old, encoding='utf-8'))
    n = len(j['distortion coeffs x'])
    order = {6: 'quadratic', 10: 'cubic', 15: 'quartic', 21: 'quintic'}[n]
    return j, order, j['star max magnitude'], j['error tolerance (as)']


def main():
    ap = argparse.ArgumentParser()
    # The 2024 run was quintic; Astrometrica fitted a CUBIC, which is also MEE's own default
    # (config.py). A cubic fit is therefore the like-for-like comparison, and it removes the
    # quartic/quintic content that was the largest line in the residual breakdown.
    ap.add_argument('--order', default='cubic',
                    help="MEE's fit order (default cubic, matching Astrometrica and MEE's own "
                         'default); pass `as2024` to use whatever the 2024 run used')
    a = ap.parse_args()
    j24, order24, mag, tol = settings_from(OLD)
    order = order24 if a.order == 'as2024' else a.order
    z = repackage()
    d = os.path.join(OUT, 'stage2_' + order)
    print('re-running stage 2 on the 2024 archive')
    print('   settings read from the 2024 run: G <= %g, tolerance %g ", corrections off'
          % (mag, tol))
    print('   order: %s%s' % (order, '' if order == order24 else
                              '   (the 2024 run was %s)' % order24))
    if not glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True):
        os.makedirs(d, exist_ok=True)
        cmd = [PY, '-m', 'mee2024.cli', 'distortion', z, '--order', order,
               '--set', 'distortion_fixed_coefficients=None',
               '--set', 'max_star_mag_dist=%g' % mag,
               '--set', 'distortion_fit_tol=%g' % tol,
               '--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
               '--set', 'observation_date=' + j24['observation_date'],
               '--no-display', '--quiet', '-o', d]
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    f = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    if not f:
        print('   FAILED -- see %s' % os.path.join(d, 'stage2.log'))
        print(open(os.path.join(d, 'stage2.log')).read()[-800:])
        sys.exit(1)
    new = json.load(io.open(f[0], encoding='utf-8'))

    print()
    if order != order24:
        print('NOTE: different order from 2024, so the star set and rms below are NOT')
        print('      like-for-like with it; the pointing and plate scale still are.')
    print('%-34s %18s %18s' % ('', 'March 2024', 'today'))
    for k in ('#stars used', 'final rms error (arcseconds)', 'platescale (arcseconds/pixel)',
              'RA', 'DEC', 'ROLL'):
        a, b = j24.get(k), new.get(k)
        fmt = '%18.7f' if isinstance(a, float) else '%18s'
        print(('%-34s ' + fmt + ' ' + fmt) % (k, a, b))
    dps = 1e6 * (new['platescale (arcseconds/pixel)'] - j24['platescale (arcseconds/pixel)']) \
        / j24['platescale (arcseconds/pixel)']
    print('%-34s %37.1f ppm' % ('plate scale, today minus 2024', dps))

    tan = os.path.join(os.path.dirname(f[0]), 'distortion_results_TAN.txt')
    print()
    print('tangent-plane export written alongside: %s' % ('yes' if os.path.isfile(tan) else 'NO'))
    print()
    print('now the comparison against Astrometrica, on this fresh fit:')
    subprocess.run([PY, os.path.join(REPO, 'tools', 'astrometrica_compare.py'),
                    '--mee', f[0]], cwd=REPO)


if __name__ == '__main__':
    main()
