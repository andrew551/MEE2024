"""S2 push: the two improvements the assessment said were still in the data.

(1) The deeper union: catalogue to G 12.0 (S2_LIMIT_MAG), stars fainter than 11.0
    admitted only with >= 2 tiers so the cross-tier vet applies to them. The mag <= 11
    subset is re-fit under the SAME deeper catalogue as a drift check -- a deeper
    catalogue tightens the doubles/blend filters on bright stars too, so the baseline is
    re-measured, not assumed.

(2) The GLS error model: the spatial covariance of the per-star residuals is MEASURED
    (exponential covariogram, amplitude per axis, one shared length scale rho fitted by
    scan over the unbinned pair products), then
      - the sandwich sigma: what statistical error the EXISTING OLS estimator really has
        under that measured correlation (the star-resampling bootstrap assumes
        independent stars; correlated patches make its sigma wrong in either direction);
      - the GLS fit itself: L and sigma_L from C^-1-weighted lstsq.
    The CAL_piLeo residual field (74 stars, zero deflection, same daytime sky 65 s
    before C2) is run through the same covariogram as an independent witness for rho.
    Caveat stated up front: amplitudes come from post-fit residuals, so smooth content
    absorbed by the v-deg2 nuisance is not in them -- both witnesses measure the
    SUB-POLYNOMIAL patch scale, which is exactly the scale left in the fit.

Last: the patch-nuisance feasibility arithmetic -- how many RBF bumps a nuisance at the
measured rho would need versus how many stars the union has. If the count fails, that IS
the answer to "have we reached the limits of the data".
"""
import glob, os, sys
import numpy as np, pandas as pd

os.environ['S2_LIMIT_MAG'] = '12.0'
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])          # build machinery + tier tables

rng = np.random.default_rng(7)

def boot_se(U, rx, ry, R, nd=2, nboot=200):
    n = len(U); boots = []
    for _ in range(nboot):
        k = rng.integers(0, n, n)
        try: boots.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=nd))
        except Exception: pass
    return float(np.std(boots, ddof=1))

def show(name, U, rx, ry, R):
    Lb = fit_L(U, rx, ry, R); Lv = fit_L(U, rx, ry, R, nuis_deg=2)
    se = boot_se(U, rx, ry, R)
    h = 1/np.mean((R_SUN_AS/R)**2)
    print(f'{name:30} N={len(U):3d} h={h:5.1f} Rsun^2  L base {Lb:+.3f}"  '
          f'L v-deg2 {Lv:+.3f} +- {se:.3f}" (stat, indep-star bootstrap)', flush=True)
    return Lv, se

print('\n=== (1) the deeper union: catalogue to G 12.0, faint stars need >= 2 tiers ===')
unions = {}
for tiers, nm in ((('0p6s','1p2s'), '0.6+1.2'), (('0p1s','0p3s','0p6s','1p2s'), 'FULL')):
    U, rx, ry, R = build_union(tiers)
    m11 = U.mag.values <= 11.0
    show(f'{nm} mag<=11 (drift check)', U[m11], rx[m11], ry[m11], R[m11])
    show(f'{nm} mag<=12 (deep)', U, rx, ry, R)
    unions[nm] = (U, rx, ry, R)
    new = U[~m11]
    if len(new):
        print('  new stars beyond G 11.0: ' + '; '.join(
            f'G {r.mag:.2f} ({int(r.ntier)} tiers, spread {r.spread:.2f}", '
            f'R {np.hypot((r.px-SUNPX)*PS, (r.py-SUNPY)*PS)/R_SUN_AS:.2f} Rsun)'
            for _, r in new.iterrows()), flush=True)

# ------------------------------------------------------- (2) measured covariance + GLS
RHOS = np.logspace(np.log10(150), np.log10(12000), 60)     # arcsec

def pairs(px, py, ex, ey):
    D = PS*np.hypot(px[:, None]-px[None, :], py[:, None]-py[None, :])
    iu = np.triu_indices(len(px), 1)
    return D, D[iu], (ex[:, None]*ex[None, :])[iu], (ey[:, None]*ey[None, :])[iu]

def fit_kernel(d, pxx, pyy):
    """Shared rho, per-axis amplitude, by scan over the unbinned pair products."""
    best = None
    for rho in RHOS:
        m = np.exp(-d/rho)
        s2x = max(0.0, float(pxx@m/(m@m)))
        s2y = max(0.0, float(pyy@m/(m@m)))
        sse = float(((pxx-s2x*m)**2).sum() + ((pyy-s2y*m)**2).sum())
        if best is None or sse < best[0]:
            best = (sse, rho, s2x, s2y)
    return best[1], best[2], best[3]

def binned(d, p, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (d >= a) & (d < b)
        out.append((a, b, k.sum(), float(np.mean(p[k])) if k.any() else np.nan))
    return out

def gls_block(U, rx, ry, R, tag):
    n = len(U)
    A, labels = design(U.px.values, U.py.values, rx, ry, R, 2)
    b = np.concatenate([U.dx.values, U.dy.values])
    c_ols, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A@c_ols
    ex, ey = r[:n], r[n:]
    D, d, pxx, pyy = pairs(U.px.values, U.py.values, ex, ey)
    rho, s2x, s2y = fit_kernel(d, pxx, pyy)
    vx, vy = float(ex.var(ddof=1)), float(ey.var(ddof=1))
    print(f'\n{tag}: per-axis residual rms x {np.sqrt(vx):.3f}" y {np.sqrt(vy):.3f}"')
    print(f'  fitted kernel: rho = {rho:.0f} arcsec ({rho/60:.1f} arcmin; frame is '
          f'{NX*PS/60:.0f} x {NY*PS/60:.0f} arcmin); correlated amplitude '
          f'x {np.sqrt(s2x):.3f}" y {np.sqrt(s2y):.3f}" '
          f'({100*s2x/max(vx,1e-9):.0f}% / {100*s2y/max(vy,1e-9):.0f}% of variance)')
    print('  binned covariogram (pair products, arcsec^2):')
    print(f'  {"bin (arcsec)":>16} {"npairs":>7} {"<ex ex>":>9} {"<ey ey>":>9}')
    edges = [0, 500, 1000, 1500, 2000, 3000, 4500, 6500, 9000, 13000]
    bx = binned(d, pxx, edges); by = binned(d, pyy, edges)
    for (a, bb, k, mx_), (_, _, _, my_) in zip(bx, by):
        print(f'  {a:6.0f} - {bb:6.0f} {k:7d} {mx_:9.4f} {my_:9.4f}')
    # covariance model: correlated kernel + white remainder (floored)
    wx = max(vx - s2x, 0.03**2); wy = max(vy - s2y, 0.03**2)
    K = np.exp(-D/rho)
    C = np.zeros((2*n, 2*n))
    C[:n, :n] = s2x*K + wx*np.eye(n)
    C[n:, n:] = s2y*K + wy*np.eye(n)
    Ci = np.linalg.inv(C)
    cov_g = np.linalg.inv(A.T@Ci@A)
    c_g = cov_g@(A.T@Ci@b)
    iL = labels.index('L')
    P = np.linalg.inv(A.T@A)@A.T                       # OLS operator
    sand = P@C@P.T                                     # its true covariance under C
    print(f'  OLS  L = {c_ols[iL]:+.3f}" ; sandwich sigma under measured C = '
          f'{np.sqrt(sand[iL, iL]):.3f}" (bootstrap said {boot_se(U, rx, ry, R):.3f}")')
    print(f'  GLS  L = {c_g[iL]:+.3f} +- {np.sqrt(cov_g[iL, iL]):.3f}" (stat)', flush=True)
    return rho, s2x, s2y

U, rx, ry, R = unions['0.6+1.2']
m11 = U.mag.values <= 11.0
gls_block(U[m11], rx[m11], ry[m11], R[m11], '=== (2) GLS, 0.6+1.2 mag<=11 ===')
rho12, s2x12, s2y12 = gls_block(U, rx, ry, R, '=== (2) GLS, 0.6+1.2 mag<=12 ===')
Uf, rxf, ryf, Rf = unions['FULL']
gls_block(Uf, rxf, ryf, Rf, '=== (2) GLS, FULL union mag<=12 ===')

# ------------------------------------------------------- CAL witness for rho
calres = glob.glob(r'D:/MEE2024 output/MEE_output/cal_pileo_step2/'
                   r'canonical_16f_night2refs/**/TWOD_RESIDUALS.csv', recursive=True)[0]
dc = pd.read_csv(calres)
cex = dc['dx_arcsec'].values - np.median(dc['dx_arcsec'])
cey = dc['dy_arcsec'].values - np.median(dc['dy_arcsec'])
_, dcal, pxxc, pyyc = pairs(dc['px'].values, dc['py'].values, cex, cey)
rhoc, s2xc, s2yc = fit_kernel(dcal, pxxc, pyyc)
print(f'\n=== CAL_piLeo witness ({len(dc)} stars, zero deflection, same daytime sky) ===')
print(f'  rho = {rhoc:.0f} arcsec ({rhoc/60:.1f} arcmin); correlated amplitude '
      f'x {np.sqrt(s2xc):.3f}" y {np.sqrt(s2yc):.3f}" '
      f'(residual rms x {cex.std(ddof=1):.3f}" y {cey.std(ddof=1):.3f}")')

# ------------------------------------------------------- patch-nuisance feasibility
print('\n=== patch-nuisance feasibility at the measured rho ===')
for rho_use, srcname in ((rho12, 'science'), (rhoc, 'CAL')):
    for spacing, k in ((rho_use, 1.0), (2*rho_use, 2.0)):
        nb = int(np.ceil(NX*PS/spacing)) * int(np.ceil(NY*PS/spacing))
        print(f'  rho from {srcname} ({rho_use:.0f}"), grid spacing {k:.0f} rho: '
              f'{nb} vertical RBF bumps + 9 base params vs N = {len(U)} stars '
              f'-> {"FEASIBLE" if nb + 9 < len(U)/1.5 else "NOT identifiable"}')
print('done', flush=True)
