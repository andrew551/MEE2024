"""The full anchor treatment for Bruns' inner annulus: per-frame, same-frame-referenced.

v1 (b17_perframe.py) measured the two inner stars per frame but referenced their
displacements to the STACKED tier offset -- and the stack lives in its alignment
frame's coordinates while the per-frame medians live in mid-series coordinates, an
~0.4-0.9 arcsec reference mismatch (the frames carry +-1 arcsec of common jitter at
0.09 s: the two stars' frame-by-frame positions visibly correlate). Superseded here.

v2 removes every external reference: on EACH raw E2 frame, every catalogue-matched E2
star is measured independently (window half-size 10 px, background = quadratic fit to
the window's border ring with saturated pixels excluded, Gaussian-weighted centroid
sigma 2.0, 5 iterations); each frame's own pointing offset is the median displacement
of its OUTER (R > 2 R_sun) stars; the inner stars' per-frame deflections are measured
against that same-frame reference. Median across frames is the measurement, MAD/sqrt(n)
the error. No stack, no alignment, no cross-frame reference anywhere.

The same machinery then runs on the PREPROCESSED (blur-subtracted, disk-painted)
frames: the difference IS the held-constant path's centroid bias in the inner annulus,
measured star by star.
"""
import glob, os, sys
import numpy as np
from astropy.io import fits

sys.argv = ['x']
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
exec(src.split("def design(")[0])         # host chain, catalogue, affines, tier tables

W, SIG = 10, 2.0

def measure(img, x0, y0):
    xi, yi = int(round(x0)), int(round(y0))
    w = img[yi-W:yi+W+1, xi-W:xi+W+1].astype(np.float64)
    if w.shape != (2*W+1, 2*W+1):
        return None
    yy, xx = np.mgrid[-W:W+1, -W:W+1]
    ring = (np.maximum(np.abs(xx), np.abs(yy)) >= W-2) & (w < 65535)
    if ring.sum() < 20:
        return None
    A = np.c_[np.ones(ring.sum()), xx[ring], yy[ring],
              xx[ring]**2, xx[ring]*yy[ring], yy[ring]**2]
    cf, *_ = np.linalg.lstsq(A, w[ring], rcond=None)
    bg = cf[0] + cf[1]*xx + cf[2]*yy + cf[3]*xx**2 + cf[4]*xx*yy + cf[5]*yy**2
    s = w - bg
    s[w >= 65535] = 0.0
    cx, cy = 0.0, 0.0
    for _ in range(5):
        g = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*SIG**2))
        ws = np.clip(s, 0, None)*g
        tot = ws.sum()
        if tot <= 0:
            return None
        cx, cy = float((ws*xx).sum()/tot), float((ws*yy).sum()/tot)
    if max(abs(cx), abs(cy)) > W-4:
        return None
    return xi+cx, yi+cy, float(s[W-3:W+4, W-3:W+4].max())

tab = tier_tabs['E2']
cra, cdec = corrs['E2']
srx, sry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
sR = np.hypot(srx, sry)
outer = sR > 2.0*R_SUN_AS
print(f'E2 matched stars: {len(tab)} ({int(outer.sum())} outer for the per-frame reference)')

def run(frames, tag):
    print(f'\n== {tag} ({len(frames)} frames)')
    per_star = {int(r.cat_i): [] for _, r in tab.iterrows()}
    for f in frames:
        img = fits.getdata(f)
        meas = {}
        for _, r in tab.iterrows():
            m = measure(img, r.px, r.py)
            if m is not None and m[2] > 250:
                sky = chain(np.array([[m[1], m[0]]]))
                i = int(r.cat_i)
                dxi = (sky[0,1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
                deta = (sky[0,0]-cdec[i])*3600
                meas[i] = (ax[0]*dxi/3600 + ax[1]*deta/3600)*PS, \
                          (ay[0]*dxi/3600 + ay[1]*deta/3600)*PS
        ref = [meas[int(r.cat_i)] for _, r in tab[outer].iterrows() if int(r.cat_i) in meas]
        if len(ref) < 5:
            print(f'  frame {os.path.basename(f)}: only {len(ref)} reference stars -- skipped')
            continue
        ox = np.median([v[0] for v in ref]); oy = np.median([v[1] for v in ref])
        for i, (dx, dy) in meas.items():
            per_star[i].append((dx-ox, dy-oy))
    print(f'{"G":>6} {"R (Rsun)":>8} {"nfr":>4} {"radial defl (as)":>16} {"+- se (as)":>10} {"GR (as)":>8}')
    out = {}
    for k, (_, r) in enumerate(tab.iterrows()):
        i = int(r.cat_i)
        v = np.array(per_star[i])
        if len(v) < 6:
            continue
        ux_, uy_ = srx[k]/sR[k], sry[k]/sR[k]
        dr = v[:,0]*ux_ + v[:,1]*uy_
        med = float(np.median(dr))
        se = 1.4826*np.median(np.abs(dr-med))/np.sqrt(len(dr))
        out[i] = (med, se)
        star_tag = ' <-- INNER' if sR[k] < 2.0*R_SUN_AS else ''
        print(f'{r.mag:6.2f} {sR[k]/R_SUN_AS:8.2f} {len(v):4d} {med:16.3f} {se:10.3f} '
              f'{L_REF*R_SUN_AS/sR[k]:8.3f}{star_tag}')
    return out

raw = run(sorted(glob.glob(r'I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse/E2_*.fit')),
          'RAW frames, quadratic local background')
pre = run(sorted(glob.glob(os.path.join(B, 'E2', 'preprocessed', '*.fits'))),
          'PREPROCESSED frames (blur-sub + disk), same measurement')

print('\n== the held-constant path bias, star by star (preprocessed minus raw, arcsec radial):')
for i in raw:
    if i in pre:
        k = int(np.where(tab.cat_i.values == i)[0][0])
        print(f'  G {tab.mag.values[k]:.2f} at {sR[k]/R_SUN_AS:.2f} Rsun: '
              f'{pre[i][0]-raw[i][0]:+.3f}')
print('done', flush=True)
