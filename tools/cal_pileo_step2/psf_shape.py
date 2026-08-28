"""Adaptive second moments of the stacked stars.

A plain moment over a fixed box measures the background, not the star: at BOX=12 with a
~2 px PSF the noise floor dominates and every star comes back at sigma ~ 5.4 px. This
iterates a Gaussian weight to the star's own size, which is what the pipeline's
centroid_refine_window does and what the moment is worth quoting from.
"""
import sys
import numpy as np, pandas as pd, zipfile
from astropy.io import fits

def adaptive(cut, n=30):
    """Gaussian-weighted moments, weight re-matched to the measured shape each pass."""
    k = cut.shape[0] // 2
    yy, xx = np.mgrid[-k:k+1, -k:k+1]
    r = np.hypot(xx, yy)
    w0 = cut - np.median(cut[(r > k-3) & (r <= k)])
    mx = my = 0.0
    vxx = vyy = 4.0
    vxy = 0.0
    for _ in range(n):
        det = vxx*vyy - vxy**2
        if det <= 0:
            return None
        dx, dy = xx - mx, yy - my
        chi2 = (vyy*dx**2 - 2*vxy*dx*dy + vxx*dy**2) / det
        g = np.exp(-0.5*chi2)
        w = w0 * g
        tot = w.sum()
        if tot <= 0:
            return None
        nmx, nmy = (w*xx).sum()/tot, (w*yy).sum()/tot
        # the Gaussian weight halves the measured moment, so undo it
        nxx = 2*(w*(xx-nmx)**2).sum()/tot
        nyy = 2*(w*(yy-nmy)**2).sum()/tot
        nxy = 2*(w*(xx-nmx)*(yy-nmy)).sum()/tot
        if nxx <= 0 or nyy <= 0 or nxx*nyy - nxy**2 <= 0:
            return None
        if abs(nxx-vxx) < 1e-4 and abs(nyy-vyy) < 1e-4 and abs(nmx-mx) < 1e-4:
            mx, my, vxx, vyy, vxy = nmx, nmy, nxx, nyy, nxy
            break
        mx, my, vxx, vyy, vxy = nmx, nmy, nxx, nyy, nxy
    return vxx, vyy, vxy

BOX = 10
img = fits.getdata(sys.argv[1]).astype(float)
c = pd.read_csv(zipfile.ZipFile(sys.argv[2]).open('STACKED_CENTROIDS_DATA.csv'))
ny, nx = img.shape
rows = []
for _, s in c.iterrows():
    x0, y0 = int(round(s.px)), int(round(s.py))
    if not (BOX < x0 < nx-BOX and BOX < y0 < ny-BOX):
        continue
    m = adaptive(img[y0-BOX:y0+BOX+1, x0-BOX:x0+BOX+1])
    if m is None:
        continue
    rows.append((s.px, s.py, s['flux (noise-normed)'], *m))

d = pd.DataFrame(rows, columns=['px', 'py', 'flux', 'vxx', 'vyy', 'vxy'])
d = d[(d.vxx.between(0.5, 25)) & (d.vyy.between(0.5, 25))]
sx, sy = np.sqrt(d.vxx), np.sqrt(d.vyy)
e = np.sqrt((d.vxx-d.vyy)**2 + 4*d.vxy**2)/(d.vxx+d.vyy)
ang = 0.5*np.degrees(np.arctan2(2*d.vxy, d.vxx-d.vyy))
print(f'{sys.argv[3] if len(sys.argv)>3 else ""}  {len(d)} stars')
print(f'  sigma_x {sx.median():.3f} px   sigma_y {sy.median():.3f} px   sy/sx {(sy/sx).median():.3f}')
print(f'  FWHM {2.355*np.sqrt(0.5*(d.vxx+d.vyy)).median()*2.2054:.2f}"   '
      f'elongation {e.median():.3f} at PA {ang.median():+.1f} deg (0=+px, 90=+py)')
