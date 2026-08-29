"""Two S2 refinements: (1) inverse-variance weighting across tiers for the full union;
(2) the joint CAL+science fit with a shared nuisance, done as a single lstsq."""
import glob, os, sys
import numpy as np, pandas as pd
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools"
exec(open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read().split("print()\nfor t in (")[0])

# ---------- (1) empirical per-tier noise, then weighted per-star combination
# per-tier variance from deviations about each star's cross-tier mean (stars in >=3 tiers)
per = {}
for t in ('0p1s','0p3s','0p6s','1p2s'):
    for _, r in tier_tabs[t].iterrows():
        per.setdefault(int(r.cat_i), {})[t] = (r.px, r.py, r.dx, r.dy)
dev = {t: [] for t in tier_tabs}
for i, tiers in per.items():
    if len(tiers) < 3: continue
    mdx = np.mean([v[2] for v in tiers.values()]); mdy = np.mean([v[3] for v in tiers.values()])
    for t, v in tiers.items():
        dev[t].append((v[2]-mdx)**2 + (v[3]-mdy)**2)
var = {t: np.mean(d) if d else 1.0 for t, d in dev.items()}
print('per-tier empirical noise (arcsec/axis):',
      {t: f'{np.sqrt(v/2):.3f}' for t, v in var.items()})
w = {t: 1.0/max(v, 1e-4) for t, v in var.items()}

recs = []
for i, tiers in per.items():
    W = sum(w[t] for t in tiers)
    px = sum(w[t]*v[0] for t, v in tiers.items())/W
    py = sum(w[t]*v[1] for t, v in tiers.items())/W
    dx = sum(w[t]*v[2] for t, v in tiers.items())/W
    dy = sum(w[t]*v[3] for t, v in tiers.items())/W
    spread = max(np.hypot(v[2]-dx, v[3]-dy) for v in tiers.values()) if len(tiers)>1 else 0
    recs.append((i, px, py, dx, dy, len(tiers), spread))
U = pd.DataFrame(recs, columns=['cat_i','px','py','dx','dy','ntier','spread'])
U['mag'] = cmag[U.cat_i.values]
sp = U.loc[U.ntier>=2, 'spread']
lim = 3*1.4826*np.median(np.abs(sp-sp.median())) + sp.median()
U = U[~((U.ntier>=2) & (U.spread > max(lim, 1.5)))]
rx, ry = (U.px.values-SUNPX)*PS, (U.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
ok = (R > 2.0*R_SUN_AS) & (U.mag.values <= 11.0)
U, rx, ry, R = U[ok], rx[ok], ry[ok], R[ok]
Lb = fit_L(U, rx, ry, R); Lv = fit_L(U, rx, ry, R, nuis_deg=2)
rng = np.random.default_rng(5); n = len(U)
boots = []
for _ in range(200):
    k = rng.integers(0, n, n)
    try: boots.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=2))
    except Exception: pass
print(f'FULL UNION, inverse-variance weighted: N={n}, '
      f'L base {Lb:+.3f}, L v-deg2 {Lv:+.3f} +- {np.std(boots, ddof=1):.3f} (stat)')

# ---------- (2) joint CAL + science, shared nuisance (single lstsq)
# CAL rows enter at their own model class: only nuisance DEGREE-3 terms are shared
# (CAL's own quadratic-free fit removed deg<=2 content from its residuals, so sharing
# deg<=2 would bias the science nuisance toward zero -- the honest joint model shares
# exactly the orders CAL legitimately measures).
calres = glob.glob(r'D:/MEE2024 output/MEE_output/cal_pileo_step2/canonical_16f_night2refs/**/TWOD_RESIDUALS.csv', recursive=True)[0]
dc = pd.read_csv(calres)
cdx = dc['dx_arcsec'].values - np.median(dc['dx_arcsec'])
cdy = dc['dy_arcsec'].values - np.median(dc['dy_arcsec'])
cxs, cys = (dc['px'].values-NX/2)/W_NORM, (dc['py'].values-NY/2)/W_NORM
sxs, sys_ = (U.px.values-NX/2)/W_NORM, (U.py.values-NY/2)/W_NORM
ux, uy = rx/R, ry/R
ns, nc = len(U), len(dc)
deg2 = [(i,j) for i in range(3) for j in range(3-i) if (i,j)!=(0,0)]
deg3 = [(i,j) for i in range(4) for j in range(4-i) if i+j==3]
Zs, Zc = np.zeros(ns), np.zeros(nc)
# science x-rows | science y-rows | cal x-rows | cal y-rows
cols = []
labels = []
def col(sx_, sy_, cx_, cy_, name):
    cols.append(np.concatenate([sx_, sy_, cx_, cy_])); labels.append(name)
col(np.ones(ns), Zs, Zc, Zc, 'N1s'); col(Zs, np.ones(ns), Zc, Zc, 'N2s')
col(-(U.py.values-NY/2)*PS, (U.px.values-NX/2)*PS, Zc, Zc, 'Ths')
col(ux*R_SUN_AS/R, uy*R_SUN_AS/R, Zc, Zc, 'L')
for i, j in deg2:                       # science-only deg-2 vertical nuisance
    col(Zs, sxs**i*sys_**j, Zc, Zc, f'v{i}{j}s')
for i, j in deg3:                       # SHARED deg-3 vertical nuisance
    col(Zs, sxs**i*sys_**j, Zc, cxs**i*cys**j, f'v{i}{j}sh')
A = np.column_stack(cols)
b = np.concatenate([U.dx.values, U.dy.values, cdx, cdy])
c, *_ = np.linalg.lstsq(A, b, rcond=None)
Ljoint = c[labels.index('L')]
print(f'JOINT CAL+science (shared deg-3 vertical, {nc} CAL stars + {n} science): '
      f'L = {Ljoint:+.3f} arcsec')
print(f'  (science-alone v-deg2 was {fit_L(U, rx, ry, R, nuis_deg=2):+.3f}; the shift is the '
      f'joint fit constraining the deg-3 terms with daytime CAL data)')
