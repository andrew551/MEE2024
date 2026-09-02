"""Cell 2, Station 1 (Mexico 2024): the LEON 18.3 diagnostic on the existing zenith fits.

Douglas, 2026-09-02, opening cell 2: the NP101 + reducer + ASI6200MM has severe quintic
distortion, but "the residual distortion does not seem to have an increase with radial
distance as the Leon data did. What strategy is best in terms of annular/gaussian/moment/
windowed for this dataset?"

The convention question was settled per instrument on Leon and Bruns by ONE measurement:
does the optic carry a magnitude-dependent centroid bias that grows with field radius
(docs/LEON_2026-08-11.md 18.3)? That is the signature of an asymmetric off-axis PSF measured
by a footprint moment -- bright and faint stars are summed over different parts of the same
one-sided profile -- and the windowed estimator exists to remove it. Leon had it at +299 mas
(bright outward) in 12/12 fields; Bruns' optic did not to speak of, and there the background
mode was the lever instead.

This runs the same diagnostic on Station 1's seventeen 2024-era zenith fits (free quintic,
tol 0.1, G <= 15, footprint moments over an S/N threshold with sigma_subtract 3 -- the exact
centroider 18.3 diagnosed). It needs no re-reduction: the matched tables are in
D:/MEE2024 output/Station 1/zenith calibrations/. It reports:

  * residual rms by field radius (is there a radial increase at all);
  * the mean RADIAL and TANGENTIAL residual by magnitude and radius -- a coma bias is
    radial-only and grows with r, so the tangential table is the control;
  * the slope of the radial residual against magnitude per radius bin, in mas/mag, which is
    the number to compare with the windowed re-reduction when the raw zenith frames are found.

Also measures the PSF FWHM on a zenith stack, since the window sigma (2.0 px) is chosen
against it.
"""
import glob, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits
from scipy.ndimage import maximum_filter

Z = r"D:/MEE2024 output/Station 1/zenith calibrations"
F = r"D:/MEE2024 output/Station 1/zenith fields"
NX, NY, PS = 9576, 6388, 1.84847

rows = []
for z in sorted(glob.glob(os.path.join(Z, 'distortion_data*.zip'))):
    zf = zipfile.ZipFile(z)
    nm = [n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    d = pd.read_csv(zf.open(nm)); d.columns = [c.strip() for c in d.columns]
    d['dx'] = (d['RA(obs)'] - d['RA(catalog)'])*np.cos(np.radians(d['DEC(catalog)']))*3600
    d['dy'] = (d['DEC(obs)'] - d['DEC(catalog)'])*3600
    d['field'] = os.path.basename(z)[15:29]
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
D['r'] = np.hypot(D.px-NX/2, D.py-NY/2)
ux, uy = (D.px-NX/2)/D.r, (D.py-NY/2)/D.r
D['rad'] = (D.dx*ux + D.dy*uy)*1000
D['tan'] = (-D.dx*uy + D.dy*ux)*1000
print('%d zenith fits, %d matched stars, G %.1f-%.1f' % (D.field.nunique(), len(D), D.magV.min(), D.magV.max()))

print('\n=== residual rms by field radius, all magnitudes (mas) ===')
edges = [0, 1000, 2000, 3000, 4000, 5000, 6000]
print('%-16s' % 'radius (px)' + ''.join('%11s' % ('%d-%d' % (a, b)) for a, b in zip(edges[:-1], edges[1:])))
for lab, col in (('radial rms', 'rad'), ('tangential rms', 'tan')):
    print('%-16s' % lab + ''.join('%11.1f' % np.sqrt(np.mean(D[col][(D.r >= a) & (D.r < b)]**2))
                                  for a, b in zip(edges[:-1], edges[1:])))
print('%-16s' % 'stars' + ''.join('%11d' % ((D.r >= a) & (D.r < b)).sum() for a, b in zip(edges[:-1], edges[1:])))
print('(the innermost tangential bin is inflated by the decomposition near r = 0, not by the data)')

print('\n=== mean RADIAL and TANGENTIAL residual by magnitude and radius, all fields pooled (mas) ===')
mb = [(4, 8), (8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
rb = [(0, 1500), (1500, 2500), (2500, 3500), (3500, 4500), (4500, 6000)]
for lab, col in (('radial', 'rad'), ('tangential', 'tan')):
    print('%-12s' % lab + ''.join('%11s' % ('r%d-%d' % b) for b in rb))
    for a, b in mb:
        k = (D.magV >= a) & (D.magV < b)
        print('%-12s' % ('G%g-%g' % (a, b)) + ''.join('%11.0f' % D[col][k & (D.r >= lo) & (D.r < hi)].mean() for lo, hi in rb))
print('%-12s' % 'stars' + ''.join('%11d' % ((D.r >= lo) & (D.r < hi)).sum() for lo, hi in rb))

print('\n=== slope of the radial residual against magnitude, per radius bin (mas per mag) ===')
print('a coma bias is radial-only and grows with r; Leon 08-12 by the 18.3 split was +299 mas bright-minus-faint beyond 2500 px')
for lo, hi in rb:
    k = (D.r >= lo) & (D.r < hi)
    c = np.polyfit(D.magV[k], D.rad[k], 1)
    print('  r %4d-%4d px: %+6.1f mas/mag   (n=%d)' % (lo, hi, c[0], k.sum()))
outer = D[D.r > 2500]
bf = outer.rad[outer.magV < 11].mean() - outer.rad[(outer.magV >= 12) & (outer.magV < 13)].mean()
per = []
for f, sub in outer.groupby('field'):
    per.append(sub.rad[sub.magV < 11].mean() - sub.rad[(sub.magV >= 12) & (sub.magV < 13)].mean())
print('  18.3 statistic, r > 2500 px, G<11 minus G12-13: %+.0f mas pooled; per field %+.0f to %+.0f, '
      'same sign in %d/%d' % (bf, min(per), max(per), sum(np.sign(p) == np.sign(bf) for p in per), len(per)))

# ---- the PSF, for the window sigma
p = sorted(glob.glob(os.path.join(F, 'CENTROID_OUTPUT*', 'STACKED*.fit')))
if p:
    img = fits.getdata(p[0], ignore_missing_simple=True).astype(float)
    med = np.median(img); mad = 1.4826*np.median(np.abs(img[::7, ::7]-med))
    box = 8
    peaks = (img == maximum_filter(img, 21)) & (img > med + 25*mad)
    ys, xs = np.where(peaks); pk = img[ys, xs]
    sel = pk < 0.6*img.max(); ys, xs, pk = ys[sel], xs[sel], pk[sel]
    order = np.argsort(-pk)[:400]
    yy, xx = np.mgrid[-box:box+1, -box:box+1]
    fw, rr = [], []
    for y, x in zip(ys[order], xs[order]):
        if y < box or x < box or y >= img.shape[0]-box or x >= img.shape[1]-box:
            continue
        sub = np.clip(img[y-box:y+box+1, x-box:x+box+1] - med, 0, None); tot = sub.sum()
        cx, cy = (sub*xx).sum()/tot, (sub*yy).sum()/tot
        var = ((sub*((xx-cx)**2 + (yy-cy)**2)).sum()/tot)/2
        fw.append(2.355*np.sqrt(max(var, 1e-6))); rr.append(np.hypot(x-NX/2, y-NY/2))
    fw, rr = np.array(fw), np.array(rr)
    print('\n=== PSF on %s ===' % os.path.basename(p[0]))
    print('  FWHM median %.2f px = %.2f"  (10-90%%: %.2f-%.2f px; n=%d); sigma %.2f px against the 2.0 px window'
          % (np.median(fw), np.median(fw)*PS, np.percentile(fw, 10), np.percentile(fw, 90), len(fw), np.median(fw)/2.355))
    for lo, hi in ((0, 2000), (2000, 3500), (3500, 6000)):
        k = (rr >= lo) & (rr < hi)
        if k.sum():
            print('    r %4d-%4d px: FWHM %.2f px (n=%d)' % (lo, hi, np.median(fw[k]), k.sum()))
