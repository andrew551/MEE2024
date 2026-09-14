"""Station 1: each block alone, the weighted mean of the four, and the pooled fit -- side by side.

Douglas, 2026-09-05: redo the per-block table under the 0.5" reference of record, add the
weighted mean and the pooled fit, and say what the difference between those two is.

THE DIFFERENCE. A per-block fit estimates offset, rotation, scale and L from that block's stars
alone; the weighted mean then averages the four L's with weights 1/sigma_b^2, and its standard
error assumes the four are independent. The pooled fit solves one linear least squares over
all 639 observations with a separate offset, rotation and scale per block and ONE L, so every
block's stars constrain the same L at once; each block's scale still floats, but it floats
against a deflection constant the other three blocks are also pinning. The two central values
agree closely because the model is linear and the per-block nuisances are the same in both.
Where they differ is the error: the weighted mean's standard error treats the blocks as
independent experiments, and they are not -- they share most of their stars, and a star's
catalogue position and its place in the distortion model are common to all four of its
observations. The pooled fit's star bootstrap (and the cluster-robust sandwich, which agrees
with it) charges for that. The weighted-mean SE is therefore too small by about a third, and
the pooled sigma is the one to quote.

Three chains are tabulated so the effect of each change is visible block by block: the
2025-analysis stacks with moment centroids, the same stacks with windowed centroids, and the
record's corona-subtracted windowed stacks. All three are fitted two-pass against the 0.5"
reference; the 2025 stacks' stage 2 is run here if it is missing.

Writes station1_record/blocks_alone.csv.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
REC = r"F:/MEE_output/station1_record"
NX, NY, PS = 9576, 6388, 1.84847
RCUT, RMAX, MAG = 2.0, 10.0, 13.0
GR = 1.7512
BLOCKS = [('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
          ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02')]
REFS = sorted(glob.glob(os.path.join(REC, 'zenith_recentroid_tol', 'tol0p5', '*', '**', 'distortion_results.txt'), recursive=True))
SUB = 'stage2_twopass_reftol0p5'
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
       '--set', 'observation_temp=15.0', '--set', 'observation_pressure=760.0',
       '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']


def stage2(root, tmid):
    """Two-pass stage 2 of the centroid archive in `root` against the 0.5\" reference, cached."""
    d = os.path.join(root, SUB); os.makedirs(d, exist_ok=True)
    hit = sorted(glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True))
    if not hit:
        cz = glob.glob(os.path.join(root, 'centroid_data*.zip'))
        if not cz:
            return None
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', cz[0], '--order', 'quintic', '--fix-distortion', *REFS,
                            '--set', 'distortion_fixed_coefficients=constant', '--set', 'distortion_free_scale=True',
                            '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                            '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *MET,
                            '--set', 'observation_time=' + tmid, '--no-display', '--quiet', '-o', d],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        hit = sorted(glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True))
    return hit[-1] if hit else None


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


def design(d, blocks):
    n = len(d); Z = np.zeros(n)
    xs, ys = (d.px.values-NX/2)*PS, (d.py.values-NY/2)*PS
    r = d.R.values; ux, uy = d.rx.values/r, d.ry.values/r; RS = d.RS.values
    cx, cy = [], []
    for b in blocks:
        m = (d.block.values == b).astype(float)
        cx += [m, Z, -ys*m, xs*m]; cy += [Z, m, xs*m, ys*m]
    cx.append(ux*RS/r); cy.append(uy*RS/r)
    return np.vstack([np.column_stack(cx), np.column_stack(cy)])


def solve(d, blocks, vet_passes=1):
    """Method 2 with per-block offset/rotation/scale and one L; one vet pass at median + 4 MAD;
    OLS sigma and the star-clustered sandwich sigma."""
    for k in range(vet_passes + 1):
        M = design(d, blocks); y = np.concatenate([d.dx.values, d.dy.values])
        c, *_ = np.linalg.lstsq(M, y, rcond=None); res = y - M@c
        n = len(d); per = np.hypot(res[:n], res[n:])
        if k == vet_passes:
            break
        lim = max(np.median(per) + 4*1.4826*np.median(np.abs(per-np.median(per))), 0.6)
        d = d[per < lim]
    XtXi = np.linalg.pinv(M.T@M); s2 = (res**2).sum()/(len(y)-M.shape[1])
    ids = np.concatenate([d.key.values, d.key.values]); meat = np.zeros((M.shape[1],)*2)
    for kk in pd.unique(ids):
        m = ids == kk; uu = M[m].T@res[m]; meat += np.outer(uu, uu)
    G = len(pd.unique(ids)); cl = XtXi@meat@XtXi*(G/(G-1))
    return dict(n=n, stars=d.key.nunique(), L=c[-1], eL=np.sqrt(s2*XtXi[-1, -1]), eL_cl=np.sqrt(cl[-1, -1]), rms=np.sqrt(s2))


ARMS = [('2025 stacks, moments', lambda t: os.path.join(REC, 'eclipse_tiers', t + '_moments')),
        ('2025 stacks, windowed', lambda t: os.path.join(REC, 'eclipse_tiers', t + '_windowed')),
        ('corona stacks, windowed (record)', lambda t: os.path.join(REC, 'eclipse_corona', t))]


def blocks_table(arms=ARMS, label='against the 0.5" reference, 2-10 R_sun, G<=13'):
    rows = []
    print('each block alone (own vet), the weighted mean, and the pooled fit -- %s' % label)
    print('%-34s %s   | weighted mean +- SE(indep)  spread   chi2/3 | pooled L +- clustered' % ('', '  '.join('%-24s' % t for t, _ in BLOCKS)))
    for name, root_of in arms:
        per, parts, cells = [], [], []
        for tag, tmid in BLOCKS:
            zp = stage2(root_of(tag), tmid)
            if not zp:
                cells.append('%-24s' % 'no stage 2'); continue
            d = table(zp, tmid); d['block'] = tag; parts.append(d)
            f = solve(d, [tag]); per.append((f['L'], f['eL']))
            cells.append('%-24s' % ('%.3f +- %.3f (%3d)' % (f['L'], f['eL'], f['stars'])))
            rows.append(dict(arm=name, block=tag, stars=f['stars'], L=f['L'], sigma=f['eL'], residual=f['rms']))
        per = np.array(per); w = 1/per[:, 1]**2
        wm, se = (w*per[:, 0]).sum()/w.sum(), 1/np.sqrt(w.sum())
        chi2 = ((per[:, 0]-wm)**2*w).sum()
        p = solve(pd.concat(parts, ignore_index=True), [t for t, _ in BLOCKS])
        print('%-34s %s   | %.3f +- %.3f            %.3f    %.1f  | %.3f +- %.3f (%d obs, %d stars)'
              % (name, '  '.join(cells), wm, se, per[:, 0].std(ddof=1), chi2/3, p['L'], p['eL_cl'], p['n'], p['stars']))
        rows.append(dict(arm=name, block='weighted mean', stars=int(per.shape[0]), L=wm, sigma=se, residual=np.nan))
        rows.append(dict(arm=name, block='pooled', stars=p['stars'], L=p['L'], sigma=p['eL_cl'], residual=p['rms']))
    print('GR 1.751"')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    out = blocks_table()
    out.to_csv(os.path.join(REC, 'blocks_alone.csv'), index=False)
    print('->', os.path.join(REC, 'blocks_alone.csv'))
