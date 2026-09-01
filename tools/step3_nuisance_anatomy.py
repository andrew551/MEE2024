"""How the vertical nuisance cuts a fake L, and what it costs -- measured, not asserted.

Douglas: "it cut the fake-L nulls 0.77 -> 0.32, a 58 % reduction: how does it do this?"

The estimator is one least-squares over 2N components. Without the nuisance it has four
columns -- two pointing offsets, a rotation, and L (a radial pattern falling as 1/R). Any
structure in the data that resembles that radial pattern has nowhere else to go, so on a
field with ZERO true deflection the atmosphere's own structure is reported as L. That is
the 0.77 arcsec.

Adding the nuisance gives the fit somewhere else to put it: a low-order polynomial in
frame position whose direction is the local vertical. Least squares then splits the data
between L and the nuisance in proportion to how well each explains it, and because Leon's
atmosphere is vertically polarised (V/H 2.3) the nuisance takes most of it. What is left in
L is the part the nuisance cannot represent.

Three numbers decide whether that trade is worth making, and this script measures all
three on the real null fields:

  1. FAKE L per nuisance order -- how much bias is removed;
  2. VARIANCE INFLATION on L -- the price. The nuisance columns are not orthogonal to the
     L column, so adding them widens L's error bar by 1/sqrt(1 - r^2), where r is the
     correlation between the L column and the nuisance subspace. This is the reason more
     nuisance is not always better;
  3. SIGNAL RETENTION -- inject a known L = 1.7512 and refit. If the nuisance can mimic
     the deflection pattern it will eat some of it, and the estimator would be quietly
     biasing the real answer as well as the fake one.
"""
import glob, os, sys
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
RD = r"D:/MEE2024 output/MEE_output/refraction"
BRUNS = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3"
L_REF = 1.7512


def build(x_px, y_px, rx, ry, R, vx, vy, nx, ny, w_norm, ps, r_sun, nuis_deg=None):
    xs, ys = (x_px-nx/2)/w_norm, (y_px-ny/2)/w_norm
    ur, vr = rx/R, ry/R
    n = len(x_px)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-ny/2)*ps, ur*r_sun/R]
    cols_y = [Z, np.ones(n), (x_px-nx/2)*ps, vr*r_sun/R]
    labels = ['N1', 'N2', 'Th', 'L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(vx*xs**i*ys**j)
                cols_y.append(vy*xs**i*ys**j)
                labels.append('v%d%d' % (i, j))
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def anatomy(tag, px, py, dx, dy, rx, ry, R, vx, vy, nx, ny, w_norm, ps, r_sun):
    print('\n%s  (N = %d stars)' % (tag, len(px)))
    print('  %-6s %10s %12s %14s %12s' % ('nuis', 'fake L', 'var-inflation',
                                          'resid rms', 'injected L back'))
    inj_x = L_REF*(r_sun/R)*(rx/R)
    inj_y = L_REF*(r_sun/R)*(ry/R)
    for nd in (None, 1, 2, 3):
        A, labels = build(px, py, rx, ry, R, vx, vy, nx, ny, w_norm, ps, r_sun, nd)
        iL = labels.index('L')
        b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A@c
        # variance inflation on L: how much of the L column the OTHER columns can mimic
        other = np.delete(A, iL, axis=1)
        col = A[:, iL]
        proj, *_ = np.linalg.lstsq(other, col, rcond=None)
        explained = other@proj
        r2 = 1.0 - float(np.sum((col-explained)**2)/np.sum(col**2))
        vif = 1.0/np.sqrt(max(1e-12, 1.0-r2))
        # signal retention: inject a known deflection and see what comes back
        cinj, *_ = np.linalg.lstsq(A, b + np.concatenate([inj_x, inj_y]), rcond=None)
        back = cinj[iL] - c[iL]
        print('  %-6s %+10.3f %12.2fx %14.3f %12.4f' %
              (str(nd), c[iL], vif, float(np.sqrt(np.mean(resid**2))), back))


# ---------------------------------------------------------------- Leon M5 nulls
PS_L, NX_L, NY_L, W_L = 2.2054043, 6248, 4176, 3124.0
RSUN_L = 947.1
SUNPX_L, SUNPY_L = 3171.0, 3232.0
VX_L, VY_L = 0.0, 1.0            # Leon: sensor +y is the local vertical to 3.1 degrees
for w in ('N1', 'N2', 'N3'):
    files = sorted(glob.glob(os.path.join(RD, 'm5_rehearsal', w, 'f*', '**',
                                          'TWOD_RESIDUALS.csv'), recursive=True))
    if not files:
        continue
    acc = {}
    for f in files:
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec))
    ids = [k for k, v in acc.items() if len(v) >= 20]
    if len(ids) < 30:
        continue
    P = np.array([[np.median([q[c] for q in acc[i]]) for c in range(4)] for i in ids])
    px, py, dx, dy = P.T
    dx, dy = dx - np.median(dx), dy - np.median(dy)
    # the S1 gate's own outlier clip, reproduced exactly: without it a handful of wild
    # per-star medians dominate every fit and the residual rms runs to 3.6 arcsec, which
    # is not the field the +-0.33 was measured on
    mag = np.hypot(dx, dy)
    lim = max(3.0*1.4826*np.median(np.abs(mag - np.median(mag))) + np.median(mag), 2.5)
    good = mag < lim
    px, py, dx, dy = px[good], py[good], dx[good], dy[good]
    rx, ry = (px-SUNPX_L)*PS_L, (py-SUNPY_L)*PS_L
    R = np.hypot(rx, ry)
    keep = R > 2.0*RSUN_L
    if keep.sum() < 30:
        continue
    px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
    anatomy('LEON night null %s (alt ~10 deg, V/H 2.3)' % w, px, py, dx, dy, rx, ry, R,
            VX_L, VY_L, NX_L, NY_L, W_L, PS_L, RSUN_L)

# ---------------------------------------------------------------- Bruns nulls
PS_B, NX_B, NY_B, W_B = 2.0868004, 3296, 2472, 1648.0
RSUN_B = 948.7
SUNPX_B, SUNPY_B = 1645.0, 1741.0
VX_B, VY_B = 0.447, -0.895
done = 0
for f in sorted(glob.glob(os.path.join(BRUNS, '*', '**', 'TWOD_RESIDUALS.csv'),
                          recursive=True)):
    if done >= 3:
        break
    field = f.replace('\\', '/').split('/')[-4]
    d = pd.read_csv(f)
    d = d[d['magV'] <= 11.0]
    px, py = d['px'].values, d['py'].values
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
    dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    rx, ry = (px-SUNPX_B)*PS_B, (py-SUNPY_B)*PS_B
    R = np.hypot(rx, ry)
    keep = R > 2.0*RSUN_B
    if keep.sum() < 30:
        continue
    px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
    anatomy('BRUNS night null %s (alt ~54 deg, V/H 1.0)' % field, px, py, dx, dy, rx, ry, R,
            VX_B, VY_B, NX_B, NY_B, W_B, PS_B, RSUN_B)
    done += 1
