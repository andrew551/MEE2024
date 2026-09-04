"""Cell 2: is the quintic sufficient? The out-of-sample test.

Douglas approved this on 2026-09-04. A septic refit of a zenith field improves its own rms by
5 %, but that proves nothing -- more parameters always fit better in sample, and the gain was
mostly at the frame CENTRE (0.099 -> 0.085 ") rather than the edge (0.124 -> 0.117), so the
radial ramp got slightly worse.

The test that decides it is out of sample: build the whole seventeen-field reference at septic
order, then use it. Two things are read, and both are independent of the fit that produced the
reference:

  1. **the null scatter.** Sixteen consecutive zenith pairs, each field fitted constant-only
     against the previous one with the eclipse Sun imposed. If the quintic were leaving real
     structure on the table, a septic reference would transfer better and the null floor would
     fall. This is the same measurement as `s1_zenith_floor.py`, run against a septic reference.
  2. **the eclipse blocks.** The four corona-subtracted stacks refitted against the septic
     reference. If the residual and L are unchanged, the quintic carried everything that
     transfers.

If both are flat, the quintic is confirmed sufficient and the septic's in-sample gain is
confirmed as over-fitting. If the null floor falls, it is not.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
REC = r"D:/MEE2024 output/MEE_output/station1_record"
RECEN = os.path.join(REC, 'zenith_recentroid')
CORONA = os.path.join(REC, 'eclipse_corona')
OUT = os.path.join(REC, 'septic_test')
NX, NY, PS = 9576, 6388, 1.84847
SUNPX, SUNPY, R_SUN_AS = 4309.0, 2730.0, 958.2
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=760.0',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
ECL = [('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
       ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02')]
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def field_dirs():
    return sorted(d for d in glob.glob(os.path.join(RECEN, '*')) if os.path.isdir(d)
                  and glob.glob(os.path.join(d, 'centroid_data*.zip')))


def tmid_of(d):
    """The block mid-time this field was fitted at, recovered from its quintic result."""
    hit = glob.glob(os.path.join(d, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
    if hit:
        j = json.load(open(hit[0], encoding='utf-8'))
        t = j.get('observation_time (UTC)')
        if t:
            return t
    return '05:45'


# ---------------------------------------------------------------- 1. the septic reference
print('=== building the seventeen-field SEPTIC reference ===', flush=True)
for d in field_dirs():
    tag = os.path.basename(d)
    o = os.path.join(OUT, 'ref', tag)
    os.makedirs(o, exist_ok=True)
    if glob.glob(os.path.join(o, '**', 'distortion_results.txt'), recursive=True):
        continue
    cz = glob.glob(os.path.join(d, 'centroid_data*.zip'))[0]
    run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'septic',
         '--set', 'distortion_fit_tol=0.3', '--set', 'max_star_mag_dist=15',
         '--set', 'rough_match_threshhold=36', *SITE,
         '--set', 'observation_time=' + tmid_of(d), '--no-display', '--quiet', '-o', o],
        os.path.join(o, 'stage2.log'))
    print('  %s done' % tag, flush=True)
SEPTIC = sorted(glob.glob(os.path.join(OUT, 'ref', '*', '**', 'distortion_results.txt'), recursive=True))
QUINTIC = sorted(glob.glob(os.path.join(RECEN, '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))
print('  septic reference: %d fields; quintic reference: %d' % (len(SEPTIC), len(QUINTIC)), flush=True)
q = [json.load(open(p, encoding='utf-8')) for p in QUINTIC]
s7 = [json.load(open(p, encoding='utf-8')) for p in SEPTIC]
print('  in-sample fit rms: quintic %.4f", septic %.4f"  (%.1f %% better, and that is the'
      ' number that proves nothing)'
      % (np.mean([j['final rms error (arcseconds)'] for j in q]),
         np.mean([j['final rms error (arcseconds)'] for j in s7]),
         100*(1-np.mean([j['final rms error (arcseconds)'] for j in s7])/np.mean([j['final rms error (arcseconds)'] for j in q]))))


def fitL(d, with_scale=True, nuis=0):
    p, qq, r = d.px.values, d.py.values, d.R.values
    ux, uy = d.rx.values/r, d.ry.values/r
    dx = d.dx.values-np.median(d.dx.values); dy = d.dy.values-np.median(d.dy.values)
    n = len(d); Z = np.zeros(n); xs, ys = (p-NX/2)*PS, (qq-NY/2)*PS
    xn, yn = (p-NX/2)/(NX/2), (qq-NY/2)/(NX/2)
    cx = [np.ones(n), Z, -ys]; cy = [Z, np.ones(n), xs]
    if with_scale:
        cx.append(xs); cy.append(ys)
    cx.append(ux*d.RS.values/r); cy.append(uy*d.RS.values/r)
    li = len(cx)-1
    if nuis:
        for i in range(nuis+1):
            for j in range(nuis+1-i):
                if i == 0 and j == 0:
                    continue
                cx.append(Z); cy.append(xn**i*yn**j)
    M = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
    c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
    res = b-Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
    e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn))))
    return (c/sc)[li], (e/sc)[li], np.sqrt(s2)


# ---------------------------------------------------------------- 2. the null test
print('\n=== the null test against each reference ===', flush=True)
FL = field_dirs()
for name, REF in (('quintic', QUINTIC), ('septic', SEPTIC)):
    Ls = []
    for prev, cur in zip(FL, FL[1:]):
        tag = os.path.basename(cur)
        o = os.path.join(OUT, 'null_' + name, tag)
        os.makedirs(o, exist_ok=True)
        hit = glob.glob(os.path.join(o, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        if not hit:
            pr = glob.glob(os.path.join(
                os.path.join(OUT, 'ref', os.path.basename(prev)) if name == 'septic'
                else os.path.join(prev, 'stage2_free'), '**', 'distortion_results.txt'), recursive=True)
            if not pr:
                continue
            cz = glob.glob(os.path.join(cur, 'centroid_data*.zip'))[0]
            run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order',
                 'septic' if name == 'septic' else 'quintic', '--fix-distortion', pr[0],
                 '--set', 'distortion_fixed_coefficients=constant', '--set', 'distortion_fit_tol=2.0',
                 '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=36', *SITE,
                 '--set', 'observation_time=' + tmid_of(cur), '--no-display', '--quiet', '-o', o],
                os.path.join(o, 'stage2.log'))
            hit = glob.glob(os.path.join(o, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        if not hit:
            continue
        t = pd.read_csv(hit[0]); t = t[t.magV <= MAGCUT]
        rx, ry = (t.px.values-SUNPX)*PS, (t.py.values-SUNPY)*PS
        R = np.hypot(rx, ry); k = R > RCUT*R_SUN_AS
        d = pd.DataFrame(dict(px=t.px.values[k], py=t.py.values[k],
                              dx=t.dx_arcsec.values[k], dy=t.dy_arcsec.values[k],
                              rx=rx[k], ry=ry[k], R=R[k], RS=R_SUN_AS))
        L, e, s = fitL(d)
        Ls.append(L)
    if Ls:
        print('  %-8s reference: %2d nulls, Method 2 rms %.4f"  max %.4f"'
              % (name, len(Ls), np.sqrt(np.mean(np.array(Ls)**2)), np.abs(Ls).max()), flush=True)

print('\n->', OUT)
