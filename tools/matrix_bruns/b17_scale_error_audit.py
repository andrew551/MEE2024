"""Bruns' plate-scale uncertainty, his formula against ours, and whether either counts the
two calibration fields.

Douglas, 2026-09-02: "With regards to the stated Bruns plate scale error: is this also
reduced by the fact that he averaged over two fields? Is this taken into account in both
ours and his estimate?"

His formula, quoted from the paper: "the RMS error multiplied by the square root of the
number of stars, divided by the sum of their distances from the image center, gives the
aggregate plate scale uncertainty" --

    sigma_S / S  =  rms * sqrt(n) / sum_i |r_i|

which is the ordinary least-squares standard error of a scale fitted through stars at
distances r_i under independent, equal-variance errors. It is dimensionless when rms and r
share units, and it falls as 1/sqrt(n), so pooling two fields' stars would carry the
sqrt(2) of averaging automatically -- but only if n is the pooled count.

He reports rms 0.077 arcsec, n = 96, and sigma_S/S = 3.34 ppm. Our L and R8 reductions of
the same two fields have 105-119 stars EACH, so 96 is a single field's worth: his figure is
a one-field uncertainty and the averaging of L and R is NOT in it.

Ours is. The per-field HC3 on those fits is 14.5 and 14.7 ppm and the record's bracket
figure is 10.3 = 14.6/sqrt(2) (docs/STEPS12_LEON_VS_BRUNS2017.md step 2).

That leaves the factor of three between 3.34 and 10.3 to explain, and this tool tests the
two candidate explanations by running his own formula on our tables:

  * if his formula on OUR rms lands near our HC3, the estimator is not the issue and the
    whole difference is the INPUT -- his 0.077 arcsec is a centroid-precision figure while
    our 0.21-0.24 arcsec is the actual fit residual, which also contains catalogue error,
    distortion-model mismatch and atmosphere;
  * if his formula on our rms lands well below our HC3, the estimator matters too, and the
    gap is the spatial correlation of the residuals that an independent-errors formula
    cannot see.

Both are reported, with his own geometry check ("for a dense, uniform array of stars this is
equal to twice the mean centroid error divided by the widest separation ... the calculated
factor is closer to 1.5") recomputed on our star distribution.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd

CONV = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024"
WIND = r"D:/MEE2024 output/MEE_output/bruns2017_lr"
NX, NY = 3296, 2472
HIS_RMS, HIS_N, HIS_SIGMA = 0.077, 96, 3.34e-6


def load(tree, field):
    z = glob.glob(os.path.join(tree, field, 'stage2', 'distortion_data*.zip'))
    if not z:
        z = glob.glob(os.path.join(tree, field, 'stage2', '**', 'distortion_data*.zip'), recursive=True)
    if not z:
        return None
    zf = zipfile.ZipFile(z[0])
    name = [n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
    res = [n for n in zf.namelist() if n.endswith('distortion_results.txt')]
    if not name:
        return None
    d = pd.read_csv(zf.open(name[0]))
    d.columns = [c.strip() for c in d.columns]
    j = json.load(zf.open(res[0])) if res else {}
    return d, j


def his_formula(d, ps, rms):
    """sigma_S/S = rms * sqrt(n) / sum |r_i|, r measured from the image centre."""
    r = np.hypot(d.px.values - NX/2, d.py.values - NY/2)*ps        # arcsec
    n = len(d)
    return rms*np.sqrt(n)/np.sum(r), n, float(np.mean(r)), float(np.max(r))


print('%-26s %6s %8s %10s %12s %12s %12s' % ('field', 'stars', 'fit rms', 'mean |r|',
                                             'his formula', 'stored HC0', 'HC3 (x1.08)'))
rows = []
for tree, label, fields in ((CONV, 'convention of record', ('L', 'R8')),
                            (WIND, 'windowed (superseded)', ('L', 'R8'))):
    for f in fields:
        got = load(tree, f)
        if got is None:
            print('%-26s not found' % (label + ' ' + f)); continue
        d, j = got
        ps = j.get('platescale (arcseconds/pixel)', 2.0868)
        rms = j.get('final rms error (arcseconds)', np.nan)
        hc0 = j.get('platescale_relative_uncertainty', np.nan)
        sig, n, rbar, rmax = his_formula(d, ps, rms)
        rows.append(dict(tree=label, field=f, n=n, rms=rms, rbar=rbar, rmax=rmax,
                         his=sig, hc0=hc0))
        print('%-26s %6d %8.4f %10.0f %10.2f ppm %10.2f ppm %10.2f ppm'
              % (label + ' ' + f, n, rms, rbar, 1e6*sig, 1e6*hc0, 1e6*hc0*1.08))

T = pd.DataFrame(rows)
rec = T[T.tree == 'convention of record']
print()
print('=== his formula, fed his own centroid rms instead of our fit rms ===')
for _, r in rec.iterrows():
    scaled = r.his*HIS_RMS/r.rms
    print('  %-3s at rms %.3f" (his figure) instead of %.3f" (our fit): %.2f ppm'
          % (r.field, HIS_RMS, r.rms, 1e6*scaled))
print('  he reports %.2f ppm from rms %.3f" and n = %d' % (1e6*HIS_SIGMA, HIS_RMS, HIS_N))
print()
print('=== his own geometry check: 2 x mean centroid error / widest separation ===')
# "mean centroid error" has to be the standard error of the MEAN, rms/sqrt(n), and "widest
# separation" the sensor's long dimension rather than its diagonal. Read any other way the
# check does not reproduce his own factor, and read this way it does -- which is the
# independent confirmation that his formula has been transcribed correctly.
LONG_SIDE = NX*2.0868
for _, r in rec.iterrows():
    simple = 2*(HIS_RMS/np.sqrt(HIS_N))/LONG_SIDE
    detailed = r.his*HIS_RMS/r.rms
    print('  %-3s long side %.0f arcsec, se of the mean %.4f" -> simple %.2f ppm; '
          'detailed/simple = %.2f (he quotes "closer to 1.5")'
          % (r.field, LONG_SIDE, HIS_RMS/np.sqrt(HIS_N), 1e6*simple, detailed/simple))
print()
one = float(rec.hc0.mean()*1.08)
print('=== does each estimate carry the two-field averaging? ===')
print('  OURS: per-field HC3 ~%.1f ppm; the record quotes the BRACKET at 10.3 ppm, which is'
      % (1e6*one))
print('        14.6/sqrt(2) from the windowed pair (docs/STEPS12 step 2). The sqrt(2) IS in.')
print('  HIS : one rms (0.077", the mean of the two fields) and one star count (96). Our own')
print('        reductions of those fields find %d and %d stars EACH, so 96 is one field\'s'
      % (int(rec.n.iloc[0]), int(rec.n.iloc[1])))
print('        worth and the sqrt(2) is NOT in. Had he applied it: %.2f ppm, and his scale'
      % (1e6*HIS_SIGMA/np.sqrt(2)))
print('        term would fall from 1.23 %% to %.2f %% of L, his total from 3.4 to %.2f %%.'
      % (1.23/np.sqrt(2), np.hypot(3.1, 1.23/np.sqrt(2))))
print()
print('=== so what is the factor of three between 3.34 and 10.3 ppm? ===')
his_on_our_rms = float(rec.his.mean())
print('  his formula on OUR fit rms, per field      : %.2f ppm' % (1e6*his_on_our_rms))
print('  our HC3, per field                          : %.2f ppm' % (1e6*one))
print('  ratio (estimator effect)                    : %.2f' % (one/his_on_our_rms))
print('  his formula on HIS rms, per field           : %.2f ppm'
      % (1e6*his_on_our_rms*HIS_RMS/float(rec.rms.mean())))
print('  ratio our-rms / his-rms (input effect)      : %.2f' % (float(rec.rms.mean())/HIS_RMS))


# ---------------------------------------------------------------------------------------
# HC0 / HC3 on the convention of record, by the recipe that produced the published table
#
# docs/STEPS12_LEON_VS_BRUNS2017.md step 2 quotes HC0 13.4/13.6 and HC3 14.5/14.7 ppm for
# Bruns L and R, and the bracket at ~10.3 = 14.6/sqrt(2). Those were computed on the
# WINDOWED pair. The reduction of record is the Gaussian + moments pair, which has more
# stars and a lower rms, so carrying 10.3 into cell 1's budget mixes two conventions.
# This recomputes both trees with the identical estimator (tools/cal_pileo_step2/
# estimator_audit.py): a quadratic design in normalised sensor coordinates, White's
# sandwich on the two linear coefficients, HC3 leverage-corrected by 1/(1-h)^2.
def hc_scale(field_dir):
    hit = glob.glob(os.path.join(field_dir, 'stage2', '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        return None
    d = pd.read_csv(hit[0])
    W, CX, CY = NX/2.0, NX/2.0, NY/2.0
    x, y = (d.px.values-CX)/W, (d.py.values-CY)/W
    X = np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])
    ex, ey = d.dx_px.values, d.dy_px.values
    XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    out = {}
    for k, f in (('HC0', np.ones(len(d))), ('HC3', 1/(1-h)**2)):
        cov = lambda e: XtXi @ ((X*(e**2*f)[:, None]).T @ X) @ XtXi
        out[k] = float(np.hypot(cov(ex)[1, 1]**.5, cov(ey)[2, 2]**.5)/W*1e6)
    out['n'] = len(d)
    return out


print()
print('=== HC0/HC3 by the published recipe, both conventions ===')
print('%-34s %6s %10s %10s' % ('field', 'stars', 'HC0 (ppm)', 'HC3 (ppm)'))
res = {}
for tree, label in ((WIND, 'windowed (the published 10.3)'), (CONV, 'convention of record')):
    vals = []
    for f in ('L', 'R8'):
        got = hc_scale(os.path.join(tree, f))
        if got is None:
            print('%-34s no residuals' % (label + ' ' + f)); continue
        vals.append(got)
        print('%-34s %6d %10.2f %10.2f' % (label + ' ' + f, got['n'], got['HC0'], got['HC3']))
    if len(vals) == 2:
        combined = float(np.mean([v['HC3'] for v in vals])/np.sqrt(2))
        res[label] = combined
        print('%-34s %6s %10s %10.2f  <- the bracket mean, HC3/sqrt(2)'
              % ('  ' + label + ' BRACKET', '', '', combined))

if len(res) == 2:
    a, b = res['windowed (the published 10.3)'], res['convention of record']
    h_cell1, RS = 8.58, 948.7
    print()
    print('  published (windowed)      %.2f ppm -> scale term %.4f" at h = %.2f' % (a, a*1e-6*h_cell1*RS, h_cell1))
    print('  convention of record      %.2f ppm -> scale term %.4f"' % (b, b*1e-6*h_cell1*RS))
    stat, atm = 0.060, 0.059
    for nm, v in (('with the published 10.3', a), ('with the record\'s own', b)):
        sc = v*1e-6*h_cell1*RS
        tot = float(np.hypot(np.hypot(stat, sc), atm))
        print('  %-26s total %.4f"  GR at %.2f sigma  Newton at %.1f sigma'
              % (nm, tot, abs(1.764-1.7512)/tot, abs(1.764-0.8756)/tot))
