"""The inner-annulus refit: both inner stars at their per-frame-verified values.

b17_perframe2.py measured (raw frames, same-frame reference, quadratic local
background) and exonerated the preprocessing (path bias <= 0.014 arcsec on every
star). The stacked chain's G 7.52 reading (+0.04 arcsec radial) is therefore a
stack-path artifact -- the remaining suspects being the 0.09 s stack's alignment
quality and a stacked-centroid-on-structured-background realization -- and the
per-frame VECTOR medians supersede the stacked values for the two E2-only stars,
exactly as Leon's anchor carried its own verified measurement into the union.

This script: re-measures per-frame vectors, rebuilds the R > 1.45 union with the two
inner rows replaced, refits L (base and v-deg2) with the 200-resample bootstrap.
"""
import glob, os, sys
import numpy as np
from astropy.io import fits

sys.argv = ['x']
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
exec(src.split("def report(")[0])          # machinery through design/build_union/fit_L

W, SIG = 10, 2.0

def measure(img, x0, y0):
    xi, yi = int(round(x0)), int(round(y0))
    w = img[yi-W:yi+W+1, xi-W:xi+W+1].astype(np.float64)
    if w.shape != (2*W+1, 2*W+1): return None
    yy, xx = np.mgrid[-W:W+1, -W:W+1]
    ring = (np.maximum(np.abs(xx), np.abs(yy)) >= W-2) & (w < 65535)
    if ring.sum() < 20: return None
    A = np.c_[np.ones(ring.sum()), xx[ring], yy[ring],
              xx[ring]**2, xx[ring]*yy[ring], yy[ring]**2]
    cf, *_ = np.linalg.lstsq(A, w[ring], rcond=None)
    s = w - (cf[0] + cf[1]*xx + cf[2]*yy + cf[3]*xx**2 + cf[4]*xx*yy + cf[5]*yy**2)
    s[w >= 65535] = 0.0
    cx = cy = 0.0
    for _ in range(5):
        g = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*SIG**2))
        ws = np.clip(s, 0, None)*g
        tot = ws.sum()
        if tot <= 0: return None
        cx, cy = float((ws*xx).sum()/tot), float((ws*yy).sum()/tot)
    if max(abs(cx), abs(cy)) > W-4: return None
    return xi+cx, yi+cy, float(s[W-3:W+4, W-3:W+4].max())

tab = tier_tabs['E2']
craE, cdecE = corrs['E2']
srx, sry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
sR = np.hypot(srx, sry)
outer = sR > 2.0*R_SUN_AS
per_star = {int(r.cat_i): [] for _, r in tab.iterrows()}
for f in sorted(glob.glob(r'I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse/E2_*.fit')):
    img = fits.getdata(f)
    meas = {}
    for _, r in tab.iterrows():
        m = measure(img, r.px, r.py)
        if m is not None and m[2] > 250:
            sky = chain(np.array([[m[1], m[0]]]))
            i = int(r.cat_i)
            dxi = (sky[0,1]-craE[i])*np.cos(np.radians(cdecE[i]))*3600
            deta = (sky[0,0]-cdecE[i])*3600
            meas[i] = ((ax[0]*dxi/3600 + ax[1]*deta/3600)*PS,
                       (ay[0]*dxi/3600 + ay[1]*deta/3600)*PS)
    ref = [meas[int(r.cat_i)] for _, r in tab[outer].iterrows() if int(r.cat_i) in meas]
    if len(ref) < 5: continue
    ox = np.median([v[0] for v in ref]); oy = np.median([v[1] for v in ref])
    for i, (dx, dy) in meas.items():
        per_star[i].append((dx-ox, dy-oy))

U, rx, ry, R = build_union(('EA','E2','EB'), 1.45)
inner_ids = [int(r.cat_i) for _, r in U.iterrows()
             if np.hypot((r.px-SUNPX), (r.py-SUNPY))*PS < 2.0*R_SUN_AS]
for i in inner_ids:
    v = np.array(per_star[i])
    mdx, mdy = float(np.median(v[:,0])), float(np.median(v[:,1]))
    k = U.index[U.cat_i == i][0]
    kk = int(np.where(tab.cat_i.values == i)[0][0])
    print(f'replacing G {U.loc[k,"mag"]:.2f} ({sR[kk]/R_SUN_AS:.2f} Rsun): stacked '
          f'({U.loc[k,"dx"]:+.2f},{U.loc[k,"dy"]:+.2f}) -> per-frame ({mdx:+.2f},{mdy:+.2f}) '
          f'arcsec over {len(v)} frames', flush=True)
    U.loc[k, 'dx'] = mdx; U.loc[k, 'dy'] = mdy

rng = np.random.default_rng(3)
def boot(U_, rx_, ry_, R_, nd):
    n = len(U_); bs = []
    for _ in range(200):
        k = rng.integers(0, n, n)
        try: bs.append(fit_L(U_.iloc[k], rx_[k], ry_[k], R_[k], nuis_deg=nd))
        except Exception: pass
    return float(np.std(bs, ddof=1))

h = 1/np.mean((R_SUN_AS/R)**2)
Lb = fit_L(U, rx, ry, R); Lv = fit_L(U, rx, ry, R, nuis_deg=2)
print(f'\nINNER union, per-frame-verified: N={len(U)} h={h:.1f} Rsun^2')
print(f'  L base   {Lb:+.3f} +- {boot(U, rx, ry, R, None):.3f} (stat)')
print(f'  L v-deg2 {Lv:+.3f} +- {boot(U, rx, ry, R, 2):.3f} (stat)')
print(f'  eq-23 scale term: {h*R_SUN_AS*10.3e-6:.3f}" @10.3 ppm, '
      f'{h*R_SUN_AS*22.5e-6:.3f}" @22.5 ppm  [GR {L_REF}]', flush=True)
print('done', flush=True)
