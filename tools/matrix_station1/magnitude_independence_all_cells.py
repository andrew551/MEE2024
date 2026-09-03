"""The magnitude-independence test across all three eclipse datasets: 2017, 2024, 2026.

Douglas, 2026-09-03: "The moment estimator carries half an arcsecond of apparent deflection
per magnitude on this field. The windowed carries none: can we show the comparison now for
all three eclipses data sets, 2017, 2024 and 2026?"

**A correction to the name first.** This test was called "the achromaticity test" in
`s1_estimator_arbiter.py` and in the commit that introduced it. That was wrong. Achromatic
means independent of *wavelength*, which gravitational deflection also is, but this test does
not vary wavelength -- it varies **apparent magnitude**. The physical fact it leans on is that
the deflection depends only on the ray's impact parameter, so a faint star and a bright star
at the same angular distance from the Sun are bent by the same angle. Brightness-independence,
not achromaticity. The tool name is corrected here and the old one is kept as an alias in the
docs so the earlier commit can be found.

**The test.** Fit the deflection term as `L + Lmag*(G - 10)` and read `Lmag`, in arcseconds of
apparent deflection per magnitude. It must be zero. It is not zero for an estimator whose
centroid error grows toward the faint end -- which is what a footprint moment does on a steep
coronal background, because the residual left by an annular subtraction scales inversely with
the star's own flux.

**What each cell can offer, honestly.** Only Station 1 has enough stars to constrain `Lmag`
tightly. Bruns' fields carry about 30 and Leon's about 35 per tier, so their `Lmag` errors are
several times larger and a null result there means "cannot tell", not "clean". That asymmetry
is itself part of the answer to whether Bruns could have chosen between his two programs.

Conventions available per cell (this is what exists on disk, not a designed matrix):

    2017 Bruns    moments + Gaussian   matrix_bruns2017_like2024  EA, EB
                  moments + annular    matrix_bruns2017_moment2   EA, EB
                  windowed             none -- would need a fresh eclipse reduction
    2024 Station1 windowed / moments   eclipse_tiers union_*.csv  (four blocks, same stacks)
    2026 Leon     windowed + Gaussian  step3_bg_ab/windowed_gaussian  four tiers
                  moments + annular    step3_bg_ab/moments_annular    four tiers

Fields are unioned within a cell (a star seen in several tiers is averaged) so every cell gets
the most stars its data allows, and the Sun's frame position is computed per field from the
ephemeris through that field's own affine.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

M = r"D:/MEE2024 output/MEE_output"
GREF = 10.0

CELLS = [
    dict(cell='2017 Bruns', conv='moments + Gaussian', nx=3296, ny=2472,
         fields=[(os.path.join(M, 'matrix_bruns2017_like2024', 'EA'), '2017-08-21T17:43:22'),
                 (os.path.join(M, 'matrix_bruns2017_like2024', 'EB'), '2017-08-21T17:44:13')],
         magcut=11.0, rmin=2.0, rmax=9.0),
    dict(cell='2017 Bruns', conv='moments + annular', nx=3296, ny=2472,
         fields=[(os.path.join(M, 'matrix_bruns2017_moment2', 'EA'), '2017-08-21T17:43:22'),
                 (os.path.join(M, 'matrix_bruns2017_moment2', 'EB'), '2017-08-21T17:44:13')],
         magcut=11.0, rmin=2.0, rmax=9.0),
    dict(cell='2026 Leon', conv='windowed + Gaussian', nx=4144, ny=2822,
         fields=[(os.path.join(M, 'step3_bg_ab', 'windowed_gaussian', t), '2026-08-12T18:28:33')
                 for t in ('0p1s', '0p3s', '0p6s', '1p2s')],
         magcut=11.0, rmin=2.0, rmax=9.0),
    dict(cell='2026 Leon', conv='moments + annular', nx=4144, ny=2822,
         fields=[(os.path.join(M, 'step3_bg_ab', 'moments_annular', t), '2026-08-12T18:28:33')
                 for t in ('0p1s', '0p3s', '0p6s', '1p2s')],
         magcut=11.0, rmin=2.0, rmax=9.0),
]


def field_table(root, tiso, nx, ny):
    zs = glob.glob(os.path.join(root, '**', 'distortion_data*.zip'), recursive=True)
    if not zs:
        return None
    zf = zipfile.ZipFile(zs[0])
    nm = [n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
    rn = [n for n in zf.namelist() if n.endswith('distortion_results.txt')]
    if not nm or not rn:
        return None
    j = json.load(zf.open(rn[0]))
    ps = j['platescale (arcseconds/pixel)']
    d = pd.read_csv(zf.open(nm[0]))
    d.columns = [c.strip() for c in d.columns]
    if 'flag_is_outlier' in d:
        d = d[d['flag_is_outlier'] == False].copy()
    if 'flag_is_double' in d:
        d = d[d['flag_is_double'] == False].copy()
    sun = get_sun(Time(tiso, scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cm = np.c_[X, Y, np.ones(len(d))]
    out = pd.DataFrame(dict(ID=d.ID.values, magV=d.magV.values, px=d.px.values, py=d.py.values,
                            dx=(ox@ax - cm@ax)*ps, dy=(ox@ay - cm@ay)*ps,
                            rx=(d.px.values-SPX)*ps, ry=(d.py.values-SPY)*ps))
    out['R'] = np.hypot(out.rx, out.ry); out['Rsun'] = out.R/RS
    out['RS'] = RS; out['ps'] = ps
    return out


def union(tables):
    ids = {}
    for k, t in enumerate(tables):
        for i in t.ID.values:
            ids.setdefault(i, []).append(k)
    rows = []
    for i, ks in ids.items():
        sub = [tables[k][tables[k].ID == i].iloc[0] for k in ks]
        rows.append(dict(ID=i, ntier=len(ks), magV=sub[0].magV,
                         **{c: float(np.mean([s[c] for s in sub]))
                            for c in ('px', 'py', 'dx', 'dy', 'rx', 'ry', 'R', 'Rsun', 'RS', 'ps')}))
    return pd.DataFrame(rows)


def fit(d, nx, ny, magterm=False, vet=True):
    d = d.copy()
    for _ in range(4 if vet else 1):
        p, q, r = d.px.values, d.py.values, d.R.values
        ux, uy = d.rx.values/r, d.ry.values/r
        RS, ps = d.RS.values, d.ps.values
        dx = d.dx.values-np.median(d.dx.values); dy = d.dy.values-np.median(d.dy.values)
        mag = d.magV.values
        n = len(d); Z = np.zeros(n)
        xs, ys = (p-nx/2)*ps, (q-ny/2)*ps
        cx = [np.ones(n), Z, -ys, xs, ux*RS/r]; cy = [Z, np.ones(n), xs, ys, uy*RS/r]
        lab = ['N1', 'N2', 'Th', 'S', 'L']
        if magterm:
            cx.append(ux*RS/r*(mag-GREF)); cy.append(uy*RS/r*(mag-GREF)); lab.append('Lmag')
        A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
        sc = np.sqrt((A**2).mean(0)); An = A/sc; b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(An, b, rcond=None)
        res = b - An@c; s2 = (res**2).sum()/(len(b)-An.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(An.T@An))))
        c, e = c/sc, e/sc
        per = np.hypot(res[:n], res[n:])
        if not vet:
            break
        lim = max(4*1.4826*np.median(np.abs(per-np.median(per))) + np.median(per), 0.6)
        if (per < lim).all():
            break
        d = d[per < lim]
    out = dict(n=len(d), rms=float(np.sqrt(s2)), magspan=float(mag.max()-mag.min()))
    for k in ('L', 'Lmag'):
        if k in lab:
            out[k] = float(c[lab.index(k)]); out['e'+k] = float(e[lab.index(k)])
    return out


rows = []
# the two Station 1 conventions come from the tier union already on disk
S1 = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_tiers"
for est in ('windowed', 'moments'):
    p = os.path.join(S1, 'union_%s.csv' % est)
    if os.path.exists(p):
        U = pd.read_csv(p); U['ps'] = 1.84847
        CELLS.append(dict(cell='2024 Station 1', conv='%s + annular' % est, nx=9576, ny=6388,
                          table=U, magcut=12.0, rmin=2.0, rmax=9.0))

print('%-15s %-21s %5s %6s %9s %9s %8s %10s %9s %7s'
      % ('dataset', 'convention', 'stars', 'G span', 'L', '+-', 'rms', 'Lmag "/mag', '+-', 'sigma'))
for c in CELLS:
    if 'table' in c:
        U = c['table']
    else:
        tabs = [t for t in (field_table(r, ti, c['nx'], c['ny']) for r, ti in c['fields']) if t is not None]
        if not tabs:
            print('%-15s %-21s  no reduction found on disk' % (c['cell'], c['conv'])); continue
        U = union(tabs)
    d = U[(U.Rsun > c['rmin']) & (U.Rsun < c['rmax']) & (U.magV <= c['magcut'])]
    if len(d) < 20:
        print('%-15s %-21s  only %d stars after cuts' % (c['cell'], c['conv'], len(d))); continue
    a = fit(d, c['nx'], c['ny'], magterm=False)
    b = fit(d, c['nx'], c['ny'], magterm=True)
    z = abs(b['Lmag']/b['eLmag']) if b['eLmag'] else float('nan')
    rows.append((c['cell'], c['conv'], a, b, z))
    print('%-15s %-21s %5d %6.1f %+9.3f %9.3f %8.3f %+10.3f %9.3f %7.1f'
          % (c['cell'], c['conv'], a['n'], a['magspan'], a['L'], a['eL'], a['rms'],
             b['Lmag'], b['eLmag'], z))

print('\nLmag is the apparent deflection per magnitude. Gravity gives zero: the deflection')
print('depends on the impact parameter, not on how bright the star looks.')
print('\n  %-15s %-21s %s' % ('dataset', 'convention', 'verdict'))
for cell, conv, a, b, z in rows:
    v = ('clean: consistent with zero' if z < 2 else 'BIASED: %.1f sigma from zero' % z)
    if b['eLmag'] > 0.30:
        v += '  (but +-%.2f "/mag -- too few stars to be decisive)' % b['eLmag']
    print('  %-15s %-21s %s' % (cell, conv, v))
