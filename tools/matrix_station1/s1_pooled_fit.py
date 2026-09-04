"""Station 1: the pooled Method-2 fit over every block observation, and its error terms.

The model, stated once so that it is on the record.  For a star i seen in block b, the
stage-2 residual (observed minus catalogue, after the seventeen-field quintic reference and a
constant-only fit) is modelled as

    dx_ib = x0_b  - theta_b * y_i  + S_b * x_i  + L * (R_sun / r_i) * ux_i
    dy_ib = y0_b  + theta_b * x_i  + S_b * y_i  + L * (R_sun / r_i) * uy_i

with (x_i, y_i) the pixel position from the frame centre in arcseconds, r_i the distance from
the Sun's centre, (ux_i, uy_i) the unit vector away from the Sun, and R_sun the solar radius
in arcseconds at the block's mid-time.  Each block b carries its own offset (x0_b, y0_b),
rotation theta_b and scale S_b -- four nuisance parameters -- and every block shares the one
deflection constant L.  Four blocks give 17 parameters; ~540 observations give ~1080
equations.  Method 2 means exactly that S_b is fitted alongside L rather than imported, which
this cell needs because the eclipse plate scale sits -640 ppm from the only calibration.

Why per block and not one scale for all: the four blocks were shot over three minutes under
moving cloud, each is its own stack with its own alignment, and the per-block S_b absorb
whatever differs between them.  The union estimator of `s1_eclipse_tiers.py`, which averaged
each star's residual over its blocks before fitting, could only fit one scale, and its L sat
0.12" above this one for that reason.

Admission: every observation in the science set (G <= --magcut, 2-9 R_sun) is a row; a star
seen in one block contributes one row, in four blocks four.  --min-blocks can impose the old
union-style rule (three of four) for comparison, but the point of pooling is that no such
rule is needed: a single observation is weighed as one observation, not as a star.  Vet: after
a first solve, any observation whose residual exceeds max(median + k * 1.4826 * MAD, 0.6") is
removed and the fit repeated -- 1.4826 * MAD is the robust estimate of the residual scatter,
so median + 4 MAD is a 4-sigma cut that the outliers it removes cannot inflate, and the 0.6"
floor stops a tight fit from cutting real stars.  --vet sets the passes (1 is the record),
--vet-k the multiplier.

The error terms:
  * the FORMAL sigma_L treats every observation as independent, and they are not -- four
    observations of one star share its catalogue position and its place in the model;
  * the STAR BOOTSTRAP resamples whole stars with all their observations and is the
    statistical term of record;
  * the SCALE SHARE answers "is there a separate scale term?".  Under Method 2 the scale is a
    fitted parameter, so its uncertainty is inside sigma_L through the S-L covariance.  The
    share is made visible by refitting with every S_b frozen at its fitted value:
    sqrt(sigma_free^2 - sigma_fixed^2) is the part of the statistical term that the scale
    degeneracy contributes, and rho(S_b, L) says how strongly each block's scale is tied to L.
    Cells 1 and 3 quote a scale term because they IMPORT a scale (Method 1) whose error is not
    in the fit; cell 2 cannot import one and so has no separate term to add.

Writes station1_record/pooled_fit/<ref>/pooled_rows.csv, pooled_summary.json, bootstrap.csv.
"""
import argparse, glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REC = r"D:/MEE2024 output/MEE_output/station1_record"
NX, NY, PS = 9576, 6388, 1.84847
MAGCUT, RCUT, RMAX = 13.0, 2.0, 9.0
BLOCKS = [('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
          ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02')]


def table(zp, tmid):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    d = d[d['flag_is_outlier'] == False].copy()
    sun = get_sun(Time('2024-04-08T' + tmid, scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
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
    d['ra'], d['dec'] = d['RA(catalog)'], d['DEC(catalog)']
    d['sun_px'], d['sun_py'] = SPX, SPY
    d['key'] = d.ID.astype(str)   # the Gaia source id: the same star in every block
    return d


def design(d, blocks, free_scale=True):
    n = len(d); Z = np.zeros(n)
    xs, ys = (d.px.values-NX/2)*PS, (d.py.values-NY/2)*PS
    r = d.R.values; ux, uy = d.rx.values/r, d.ry.values/r; RS = d.RS.values
    cx, cy, names = [], [], []
    for b in blocks:
        m = (d.block.values == b).astype(float)
        cx += [m, Z, -ys*m]; cy += [Z, m, xs*m]; names += [b+':x0', b+':y0', b+':theta']
        if free_scale:
            cx.append(xs*m); cy.append(ys*m); names.append(b+':S')
    cx.append(ux*RS/r); cy.append(uy*RS/r); names.append('L')
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), names


def solve(M, b):
    sc = np.sqrt((M**2).mean(0)); Mn = M/sc
    c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
    res = b - Mn@c; dof = len(b) - Mn.shape[1]; s2 = (res**2).sum()/dof
    cov = s2*np.linalg.pinv(Mn.T@Mn)/np.outer(sc, sc)
    return c/sc, cov, res, float(np.sqrt(s2))


def scale_columns(d, blocks):
    """The S_b columns alone, for freezing the scale at its fitted value."""
    n = len(d); Z = np.zeros(n)
    xs, ys = (d.px.values-NX/2)*PS, (d.py.values-NY/2)*PS
    cx, cy = [], []
    for b in blocks:
        m = (d.block.values == b).astype(float)
        cx.append(xs*m); cy.append(ys*m)
    return np.vstack([np.column_stack(cx), np.column_stack(cy)])


ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='quintic', choices=['quintic', 'septic', 'twopass'],
                help='quintic: the record (imported scale, 20" gate); septic: model-order sensitivity; '
                     'twopass: scale fitted at stage 2, 20" then 3" gates')
ap.add_argument('--min-blocks', type=int, default=1,
                help='a star enters with this many block observations or more; 1 = every observation')
ap.add_argument('--magcut', type=float, default=13.0, help='science-set magnitude limit, G')
ap.add_argument('--vet', type=int, default=1, help='vet passes (1 is the record; 0 = no vet)')
ap.add_argument('--vet-k', type=float, default=4.0, help='the vet cut: median + k * 1.4826 * MAD')
ap.add_argument('--tag', default='', help='output subdirectory suffix, for grids')
ap.add_argument('--boot', type=int, default=1000)
ap.add_argument('--seed', type=int, default=17)
a = ap.parse_args()
OUT = os.path.join(REC, 'pooled_fit', a.ref + ('_' + a.tag if a.tag else '')); os.makedirs(OUT, exist_ok=True)
PUBLISHED_SCALE = 1.847363   # Dittrich et al. 2025, +- 1.3e-5 "/px (7 ppm); L = 1.839 +- 0.239"
sub = {'quintic': 'stage2', 'septic': 'stage2_septic', 'twopass': 'stage2_twopass'}[a.ref]

# ---------------------------------------------------------------- the rows
parts, P2 = [], {}
for tag, tmid in BLOCKS:
    hit = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', tag, sub, '**', 'distortion_data*.zip'), recursive=True))
    if not hit:
        print('  %s: no %s' % (tag, sub)); continue
    d = table(hit[-1], tmid)
    # the plate scale stage 2 used for this block's positions: the reference's when it
    # was imported, the field's own under distortion_free_scale
    P2[tag] = float(json.load(zipfile.ZipFile(hit[-1]).open('distortion_results.txt'))['platescale (arcseconds/pixel)'])
    d = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= a.magcut)].copy()
    d['block'] = tag
    parts.append(d)
d = pd.concat(parts, ignore_index=True)
blocks = [t for t, _ in BLOCKS if t in set(d.block)]
seen = d.groupby('key').block.nunique()
d = d[d.key.map(seen) >= a.min_blocks].copy()
print('=== pooled Method-2 fit, %s reference, G <= %.0f, stars in >= %d of %d blocks, vet %d pass(es) at median + %.0f MAD ==='
      % (a.ref, a.magcut, a.min_blocks, len(blocks), a.vet, a.vet_k))
print('  %d observations of %d stars before the vet' % (len(d), d.key.nunique()))

# ---------------------------------------------------------------- the fit, with the vet
for k in range(a.vet + 1):
    M, names = design(d, blocks)
    b = np.concatenate([d.dx.values, d.dy.values])
    c, cov, res, rms = solve(M, b)
    n = len(d); per = np.hypot(res[:n], res[n:])
    if k == a.vet:
        break
    lim = max(np.median(per) + a.vet_k*1.4826*np.median(np.abs(per-np.median(per))), 0.6)
    drop = per >= lim
    print('  vet pass %d: %d observations beyond %.3f" removed' % (k+1, drop.sum(), lim))
    d = d[~drop].copy()
iL = names.index('L')
d['res_x'], d['res_y'], d['res'] = res[:n], res[n:], per
eL = float(np.sqrt(cov[iL, iL]))
print('  %d observations of %d stars; %d parameters' % (n, d.key.nunique(), len(names)))
print('  L = %.3f +- %.3f" (formal), per-observation residual %.3f"' % (c[iL], eL, rms))
for bname in blocks:
    iS, iT = names.index(bname+':S'), names.index(bname+':theta')
    rho = cov[iS, iL]/np.sqrt(cov[iS, iS]*cov[iL, iL])
    # the JOINT scale -- stage 2's scale corrected by the S_b fitted alongside L -- is the
    # one comparable to a published plate scale; a scale fitted without L has the
    # deflection absorbed into it (48 ppm here) and is not
    joint = P2[bname] - c[iS]*PS
    print('    %-11s %3d obs  S = %+7.1f +- %4.1f ppm  theta = %+6.1f"  rho(S, L) = %+.2f  joint scale %.7f "/px (%+.0f ppm from published)'
          % (bname, (d.block == bname).sum(), 1e6*c[iS], 1e6*np.sqrt(cov[iS, iS]), 206265*c[iT], rho,
             joint, 1e6*(joint/PUBLISHED_SCALE - 1)))

# ---------------------------------------------------------------- each block alone, same rows
print('  each block alone on the same rows:')
for bname in blocks:
    db = d[d.block == bname]
    Mb, nb = design(db, [bname]); cb, covb, _, rb = solve(Mb, np.concatenate([db.dx.values, db.dy.values]))
    print('    %-11s %3d stars  L = %.3f +- %.3f"  residual %.3f"' % (bname, len(db), cb[-1], np.sqrt(covb[-1, -1]), rb))

# ---------------------------------------------------------------- the scale share
Sfix = np.array([c[names.index(bname+':S')] for bname in blocks])
b_fixed = b - scale_columns(d, blocks)@Sfix
Mf, nf = design(d, blocks, free_scale=False)
cf, covf, _, rf = solve(Mf, b_fixed)
eL_fixed = float(np.sqrt(covf[nf.index('L'), nf.index('L')]))
share = float(np.sqrt(max(eL**2 - eL_fixed**2, 0.0)))
print('  scale share: sigma_L %.3f" with the four scales free, %.3f" with them frozen at their fitted'
      ' values -> %.3f" of the formal term is the scale degeneracy' % (eL, eL_fixed, share))

# ---------------------------------------------------------------- the star bootstrap
rng = np.random.default_rng(a.seed)
keys = d.key.unique(); groups = {k: np.flatnonzero(d.key.values == k) for k in keys}
Ls = []
for _ in range(a.boot):
    pick = rng.choice(keys, len(keys), replace=True)
    idx = np.concatenate([groups[k] for k in pick])
    dd = d.iloc[idx]
    Mb, nb = design(dd, blocks)
    cb, *_ = solve(Mb, np.concatenate([dd.dx.values, dd.dy.values]))
    Ls.append(cb[nb.index('L')])
Ls = np.array(Ls)
print('  star bootstrap (%d samples over %d stars): L = %.3f +- %.3f"  [16-84 %%: %.3f - %.3f]'
      % (a.boot, len(keys), c[iL], Ls.std(), *np.percentile(Ls, [16, 84])))

summary = dict(ref=a.ref, min_blocks=a.min_blocks, magcut=a.magcut, vet_passes=a.vet, vet_k=a.vet_k, observations=int(n),
               stars=int(d.key.nunique()), L=float(c[iL]), sigma_formal=eL, sigma_bootstrap=float(Ls.std()),
               sigma_L_scales_frozen=eL_fixed, scale_share=share, residual=rms,
               blocks={bname: dict(n=int((d.block == bname).sum()), S_ppm=float(1e6*c[names.index(bname+':S')]),
                                   joint_scale_arcsec_per_px=float(P2[bname] - c[names.index(bname+':S')]*PS),
                                   theta_arcsec=float(206265*c[names.index(bname+':theta')]),
                                   rho_S_L=float(cov[names.index(bname+':S'), iL]/np.sqrt(cov[names.index(bname+':S'), names.index(bname+':S')]*cov[iL, iL])))
                       for bname in blocks})
summary['GR'], summary['NEWTON'] = 1.7512, 0.8756
summary['L_over_Newton'] = float(c[iL]/0.8756)
json.dump(summary, open(os.path.join(OUT, 'pooled_summary.json'), 'w'), indent=1)
d.to_csv(os.path.join(OUT, 'pooled_rows.csv'), index=False)
pd.DataFrame(dict(L=Ls)).to_csv(os.path.join(OUT, 'bootstrap.csv'), index=False)
print('\n->', OUT)
