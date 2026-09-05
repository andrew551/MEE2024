"""Station 1: is the moment estimator's magnitude bias the corona, or the PSF?

Under footprint moments the 2025-analysis eclipse stacks show an apparent deflection that grows
with faintness, +0.36 "/mag at 5.4 sigma; under the windowed estimator, none. The zenith fields
carry the same sign of moment bias with no corona present (bright inward, faint outward,
22-31 mas/mag), which points at the PSF; but moments were never run on the corona-subtracted
stacks, so how much of the 0.36 the coronal background itself contributes was not measured.
Douglas asked for the run. Leon 2026 is the reference case: there the effect was coma and it
was visible in the zenith calibrations too.

The test: re-centroid the four corona-subtracted, occulted stacks (the record's stacks) with
the MOMENT estimator, everything else as the record, fit them two-pass against the same quintic
reference, and measure Lmag -- the coefficient of (G - 10) on the deflection column -- block by
block and pooled. Three arms are then on the table:

    moments  on the 2025-analysis stacks   (corona in the image)      Lmag = +0.36
    moments  on the corona-subtracted stacks                          this run
    windowed on the corona-subtracted stacks (the record)             Lmag ~ 0

If the middle row stays near +0.36 the bias is the estimator on this PSF and the corona is
incidental; if it falls to ~0 the corona was the cause; in between, both.

Why the moments arm is re-stacked from RAW rather than re-centroided from the record's saved
stacks: the occulter and coronal subtraction leave the saved stack with no saturated pixels
(max 49 000 ADU, none at 65 535), so the blob mask that hides the disk never triggers on it,
and the moment detector -- which has no window to keep it honest -- finds eight to ten
thousand "centroids" on the disk edge and the plate solve fails. That is what the first run
did. The record's own windowed stacks were built from raw with the same command, so this is
the like-for-like construction: the record's stacking with one flag changed.

Writes station1_record/moments_on_corona/<tag>/ (stage 1 and stage 2) and lmag_table.csv.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
REC = r"D:/MEE2024 output/MEE_output/station1_record"
OUT = os.path.join(REC, 'moments_on_corona')
NX, NY, PS = 9576, 6388, 1.84847
RCUT, RMAX, MAG = 2.0, 10.0, 13.0
BLOCKS = [('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
          ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02')]
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
# tag -> raw block, dark set, first frame to use (the 0.25 s block starts before second contact)
RAW = {'0p25s_1810': ('2024-04-08_18_10_26Z', 'dark-250ms', 15),
       '0p3s_1811':  ('2024-04-08_18_11_28Z', 'dark-300ms', 0),
       '0p4s_1812':  ('2024-04-08_18_12_30Z', 'dark-400ms', 0),
       '0p3s_1813':  ('2024-04-08_18_13_31Z', 'dark-300ms', 0)}
# the record's stage-1 settings verbatim (s1_eclipse_corona.py), with one flag changed:
# centroid_refine_window=False is the footprint-moment estimator
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=False',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
      '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=10.0',
      '--set', 'coronal_pedestal_adu=2000.0']
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
       '--set', 'observation_temp=15.0', '--set', 'observation_pressure=760.0',
       '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
REFS = sorted(glob.glob(os.path.join(REC, 'zenith_recentroid', '*', 'stage2_free', '**',
                                     'distortion_results.txt'), recursive=True))


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def recentroid(tag):
    """Re-stack this block from raw exactly as the record did, with the moment estimator."""
    block, darkset, first = RAW[tag]
    d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        frames = sorted(glob.glob(os.path.join(G, 'CapObj', block, '*.FIT')))[first:]
        print('  %s: stacking %d raw frames with moments, %s + flat, occulter, coronal subtract...'
              % (tag, len(frames), darkset), flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', *frames,
             '--dark', os.path.join(G, darkset, 'CapObj', '*', '*.FIT'),
             '--flat', os.path.join(G, 'flat', 'CapObj', '2024-04-08*', '*.FIT'),
             *S1, '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return z[0] if z else None


def stage2(tag, cz, tmid):
    d = os.path.join(OUT, tag, 'stage2_twopass'); os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    if not hit:
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic', '--fix-distortion', *REFS,
             '--set', 'distortion_fixed_coefficients=constant', '--set', 'distortion_free_scale=True',
             '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *MET,
             '--set', 'observation_time=' + tmid, '--no-display', '--quiet', '-o', d],
            os.path.join(d, 'stage2.log'))
        hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    return hit[0] if hit else None


def table(zp, tmid):
    sun = get_sun(Time('2024-04-08T' + tmid, scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]; d = d[d['flag_is_outlier'] == False].copy()
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cm = np.c_[X, Y, np.ones(len(d))]
    d['dx'] = (ox@ax - cm@ax)*PS; d['dy'] = (ox@ay - cm@ay)*PS
    d['rx'] = (d.px.values-SPX)*PS; d['ry'] = (d.py.values-SPY)*PS
    d['R'] = np.hypot(d.rx, d.ry); d['Rsun'] = d.R/RS; d['RS'] = RS
    d['key'] = d.ID.astype(str)
    return d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAG)].copy()


def fit_lmag(d, blocks):
    """Method 2 per block plus one L and one Lmag, the coefficient of (G-10) on the deflection
    column. Vetted once at median + 4 MAD, as the record is."""
    for _ in range(2):
        n = len(d); Z = np.zeros(n)
        xs, ys = (d.px.values-NX/2)*PS, (d.py.values-NY/2)*PS
        r = d.R.values; ux, uy = d.rx.values/r, d.ry.values/r; RS = d.RS.values
        cx, cy = [], []
        for b in blocks:
            m = (d.block.values == b).astype(float)
            cx += [m, Z, -ys*m, xs*m]; cy += [Z, m, xs*m, ys*m]
        g = d.magV.values - 10.0
        cx += [ux*RS/r, g*ux*RS/r]; cy += [uy*RS/r, g*uy*RS/r]
        M = np.vstack([np.column_stack(cx), np.column_stack(cy)])
        b_ = np.concatenate([d.dx.values, d.dy.values])
        sc = np.sqrt((M**2).mean(0)); Mn = M/sc
        c, *_ = np.linalg.lstsq(Mn, b_, rcond=None)
        res = b_ - Mn@c; s2 = (res**2).sum()/(len(b_)-Mn.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn)))); c, e = c/sc, e/sc
        per = np.hypot(res[:n], res[n:])
        lim = max(np.median(per) + 4*1.4826*np.median(np.abs(per-np.median(per))), 0.6)
        if (per < lim).all():
            break
        d = d[per < lim]
    return dict(n=n, stars=d.key.nunique(), L=c[-2], eL=e[-2], Lmag=c[-1], eLmag=e[-1], rms=np.sqrt(s2))


os.makedirs(OUT, exist_ok=True)
print('quintic reference: %d fields' % len(REFS), flush=True)
arms = {'moments, 2025-analysis stacks': lambda tag: sorted(glob.glob(os.path.join(
            REC, 'eclipse_tiers', tag + '_moments', 'stage2_F_17field_windowed', '**', 'distortion_data*.zip'), recursive=True)),
        'moments, corona-subtracted stacks': None,
        'windowed, corona-subtracted stacks (record)': lambda tag: sorted(glob.glob(os.path.join(
            REC, 'eclipse_corona', tag, 'stage2_twopass', '**', 'distortion_data*.zip'), recursive=True))}
rows = []
for name, finder in arms.items():
    parts = []
    print('\n=== %s ===' % name, flush=True)
    for tag, tmid in BLOCKS:
        if finder is None:
            cz = recentroid(tag)
            zp = stage2(tag, cz, tmid) if cz else None
            if cz:
                nc = json.load(zipfile.ZipFile(cz).open('results.txt')).get('n_centroids')
                print('  %s: %s centroids' % (tag, nc), flush=True)
        else:
            hit = finder(tag); zp = hit[-1] if hit else None
        if not zp:
            print('  %s: no stage 2' % tag, flush=True); continue
        d = table(zp, tmid); d['block'] = tag; parts.append(d)
        f = fit_lmag(d.copy(), [tag])
        rows.append(dict(arm=name, block=tag, **f))
        print('  %-11s %3d obs  L = %.3f +- %.3f  Lmag = %+.3f +- %.3f "/mag (%.1f sigma)  residual %.3f"'
              % (tag, f['n'], f['L'], f['eL'], f['Lmag'], f['eLmag'], abs(f['Lmag'])/f['eLmag'], f['rms']), flush=True)
    if parts:
        d = pd.concat(parts, ignore_index=True)
        f = fit_lmag(d, [t for t, _ in BLOCKS if t in set(d.block)])
        rows.append(dict(arm=name, block='pooled', **f))
        print('  %-11s %3d obs  L = %.3f +- %.3f  Lmag = %+.3f +- %.3f "/mag (%.1f sigma)  residual %.3f"'
              % ('POOLED', f['n'], f['L'], f['eL'], f['Lmag'], f['eLmag'], abs(f['Lmag'])/f['eLmag'], f['rms']), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'lmag_table.csv'), index=False)
print('\n->', OUT)
