"""Cell 1 against Bruns 2018, number for number -- and the answers to Douglas'
2026-09-01 chart critique that need computation rather than redrawing.

Covers: (a) the full per-star value table, with tangential components, so the arrows can
be checked against numbers; (b) the fit with and without the two close-in stars, both
methods, mirroring Bruns' own published variants; (c) the fit without the rotation
parameter, since Bruns minimised "only the deflection coefficient and simple RA and Dec
offsets" -- no roll; (d) the link-count scan: the 7-brightest-common-stars offset was
Bruns' choice, and this measures whether more (or fewer) common stars would place the
close-in pair better, and where dimmer link stars start to hurt; (e) forensics on the
star near r = 4.7 R_sun with the anomalously small deflection.
"""
import glob, os, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
CONV = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024"
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
GR = 1.7512

tab = pd.read_csv(os.path.join(OUT, 'bruns_method_star_table.csv'))
rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
linked = (tab.src == 'E2-linked').values
n = len(tab)


def solve(sel=None, with_scale=False, with_rotation=True):
    s = np.ones(n, bool) if sel is None else sel
    m = int(s.sum())
    px_, py_ = tab.px.values[s], tab.py.values[s]
    dx_, dy_ = tab.dx.values[s], tab.dy.values[s]
    rxs, rys, Rs = rx[s], ry[s], R[s]
    ur, vr = rxs/Rs, rys/Rs
    Z = np.zeros(m)
    cols_x = [np.ones(m), Z]
    cols_y = [Z, np.ones(m)]
    labels = ['N1', 'N2']
    if with_rotation:
        cols_x.append(-(py_-NY/2)*PS)
        cols_y.append((px_-NX/2)*PS)
        labels.append('Th')
    if with_scale:
        cols_x.append((px_-NX/2)*PS)
        cols_y.append((py_-NY/2)*PS)
        labels.append('S')
    cols_x.append(ur*R_SUN_AS/Rs)
    cols_y.append(vr*R_SUN_AS/Rs)
    labels.append('L')
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([dx_, dy_])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    iL = labels.index('L')
    return c[iL], float(np.sqrt(cov[iL, iL])), c, labels


def boot(sel=None, with_scale=False, with_rotation=True, nb=400):
    s = np.arange(n) if sel is None else np.where(sel)[0]
    rng = np.random.default_rng(3)
    out = []
    for _ in range(nb):
        k = rng.choice(s, len(s))
        mask = np.zeros(n, bool)
        # bootstrap with repetition needs index lists, not masks; rebuild directly
        try:
            tb = tab.iloc[k]
            rxs, rys = (tb.px.values-SUNPX)*PS, (tb.py.values-SUNPY)*PS
            Rs = np.hypot(rxs, rys)
            m = len(tb)
            ur, vr = rxs/Rs, rys/Rs
            Z = np.zeros(m)
            cols_x = [np.ones(m), Z]
            cols_y = [Z, np.ones(m)]
            if with_rotation:
                cols_x.append(-(tb.py.values-NY/2)*PS)
                cols_y.append((tb.px.values-NX/2)*PS)
            if with_scale:
                cols_x.append((tb.px.values-NX/2)*PS)
                cols_y.append((tb.py.values-NY/2)*PS)
            cols_x.append(ur*R_SUN_AS/Rs)
            cols_y.append(vr*R_SUN_AS/Rs)
            A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
            c, *_ = np.linalg.lstsq(A, np.concatenate([tb.dx.values, tb.dy.values]),
                                    rcond=None)
            out.append(c[-1])
        except Exception:
            pass
    return float(np.std(out, ddof=1))


# ---- (a) the per-star value table, radial AND tangential
L1, sL1, c1, l1 = solve()
# the L-view displacement: data minus everything the fit attributes to non-L columns
ur, vr = rx/R, ry/R


def clean_vectors(c, labels, with_scale=False, with_rotation=True):
    dxc = tab.dx.values - c[labels.index('N1')]
    dyc = tab.dy.values - c[labels.index('N2')]
    if with_rotation:
        dxc -= c[labels.index('Th')]*(-(tab.py.values-NY/2)*PS)
        dyc -= c[labels.index('Th')]*((tab.px.values-NX/2)*PS)
    if with_scale:
        dxc -= c[labels.index('S')]*((tab.px.values-NX/2)*PS)
        dyc -= c[labels.index('S')]*((tab.py.values-NY/2)*PS)
    return dxc, dyc


dxc, dyc = clean_vectors(c1, l1)
rad = dxc*ur + dyc*vr
tan = -dxc*vr + dyc*ur
print('=== per-star table (Method 1 view: constants+rotation removed) ===')
print('%6s %6s %6s %8s %9s %9s %9s %8s' % ('G', 'px', 'py', 'R/Rsun', 'rad (")',
                                           'tan (")', 'GR (")', 'src'))
for k in np.argsort(R):
    print('%6.2f %6.0f %6.0f %8.2f %+9.3f %+9.3f %9.3f %8s'
          % (tab.mag.values[k], tab.px.values[k], tab.py.values[k], R[k]/R_SUN_AS,
             rad[k], tan[k], GR*R_SUN_AS/R[k], tab.src.values[k][:8]))
print('tangential rms %.3f arcsec (radial-about-fit rms %.3f) -- the arrows are NOT '
      'purely radial' % (float(np.sqrt(np.mean(tan**2))),
                         float(np.sqrt(np.mean((rad - L1*R_SUN_AS/R)**2)))))

# ---- (b)+(c) the Bruns-variant grid
print('\n=== the fit grid (bootstrap errors, 400 resamples) ===')
grid = []
for name, sel, ws, wr in (
        ('M1, all 27, with rotation', None, False, True),
        ('M1, all 27, NO rotation (Bruns: offsets+L only)', None, False, False),
        ('M1, sans close-in (25), with rotation', ~linked, False, True),
        ('M1, sans close-in, NO rotation', ~linked, False, False),
        ('M2 (scale free), all 27', None, True, True),
        ('M2 (scale free), sans close-in', ~linked, True, True)):
    L, sL, _, _ = solve(sel, ws, wr)
    se = boot(sel, ws, wr)
    m = n if sel is None else int(sel.sum())
    hh = 1/np.mean((R_SUN_AS/(R if sel is None else R[sel]))**2)
    grid.append((name, m, L, se, hh))
    print('%-48s N=%2d  L = %+.3f +- %.3f   h = %.1f' % (name, m, L, se, hh))

# ---- (d) the link-count scan
print('\n=== the offset link vs number of common stars ===')
det = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(OUT, 'master062',
                                                         'centroid_data*.zip'))[0])
                  .open('STACKED_CENTROIDS_DATA.csv'))
detE2 = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(CONV, 'E2',
                                                           'centroid_data*.zip'))[0])
                    .open('STACKED_CENTROIDS_DATA.csv'))
detE2 = detE2.sort_values('flux (noise-normed)', ascending=False).reset_index(drop=True)
pairs = []
for _, r in detE2.iterrows():
    dd = np.hypot(det['px'].values - r.px, det['py'].values - r.py)
    k = int(np.argmin(dd))
    if dd[k] < 25.0:
        pairs.append((det['px'].values[k] - r.px, det['py'].values[k] - r.py,
                      r['flux (noise-normed)']))
P = np.array(pairs)
print('common stars found: %d (E2 has %d detections)' % (len(P), len(detE2)))
print('%4s %12s %12s %14s %11s' % ('N', 'offx (px)', 'offy (px)', 'se of link (")',
                                   'faintest flux'))
off7 = None
for N in range(3, len(P)+1):
    ox, oy = P[:N, 0].mean(), P[:N, 1].mean()
    se = float(np.hypot(P[:N, 0].std(ddof=1), P[:N, 1].std(ddof=1))/np.sqrt(N))*PS
    print('%4d %12.3f %12.3f %14.3f %11.0f' % (N, ox, oy, se, P[N-1, 2]))
    if N == 7:
        off7 = (ox, oy)

# ---- (e) forensics on the r ~ 4.7 outlier
print('\n=== the r ~ 4.7 R_sun outlier ===')
cand = np.argmin(np.abs(R/R_SUN_AS - 4.7) + np.abs(rad - 0.2))
k = int(np.argmin(np.where((R/R_SUN_AS > 4.3) & (R/R_SUN_AS < 5.0), rad, 1e9)))
print('star: G %.2f at px (%.0f, %.0f), R = %.2f R_sun, radial %+.3f" (GR %.3f), '
      'tangential %+.3f"' % (tab.mag.values[k], tab.px.values[k], tab.py.values[k],
                             R[k]/R_SUN_AS, rad[k], GR*R_SUN_AS/R[k], tan[k]))
edge = min(tab.px.values[k], tab.py.values[k], NX-tab.px.values[k], NY-tab.py.values[k])
print('distance from nearest frame edge: %.0f px' % edge)
dd = np.hypot(det['px'].values - tab.px.values[k], det['py'].values - tab.py.values[k])
near_det = np.sort(dd)[1:4]
print('nearest OTHER detections in the master: %s px' % np.round(near_det, 1))
row = det.iloc[int(np.argmin(dd))]
print('detection: area %.0f px, flux %.0f, peak %s ADU'
      % (row['area (pixels)'], row['flux (noise-normed)'], row.get('peak (adu)', '?')))
# catalogue neighbours to mag 13 within 20 arcsec
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float
prov = providers.GaiaOfflineProvider.from_installed()
zh = zipfile.ZipFile(glob.glob(os.path.join(OUT, 'master062', 'stage2',
                                            'distortion_data*.zip'))[0])
import json
jres = json.load(zh.open([m for m in zh.namelist() if m.endswith('distortion_results.txt')][0]))
dh = pd.read_csv(zh.open([m for m in zh.namelist()
                          if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
dm = np.hypot(dh['px']-tab.px.values[k], dh['py']-tab.py.values[k])
i = int(np.argmin(dm))
ra_s, dec_s = dh['RA(catalog)'].iloc[i], dh['DEC(catalog)'].iloc[i]
cat13 = prov.lookup((ra_s-0.02, ra_s+0.02), (dec_s-0.02, dec_s+0.02), 13.0,
                    epoch=date_string_to_float('2017-08-21'))
cra13 = np.degrees(cat13.get_ra()); cdec13 = np.degrees(cat13.get_dec())
mag13 = np.asarray(cat13.get_mags())
sep = np.hypot((cra13-ra_s)*np.cos(np.radians(dec_s)), cdec13-dec_s)*3600
order = np.argsort(sep)
print('catalogue neighbours to G 13 within 30 arcsec:')
for j in order[:5]:
    if sep[j] < 30:
        print('   G %.2f at %.1f arcsec' % (mag13[j], sep[j]))

# ---- (f) the Bruns-vs-us table inputs
print('\n=== our error terms in Bruns\u2019 own currencies ===')
h_all = 1/np.mean((R_SUN_AS/R)**2)
print('scale: his moment formula gave 3.34 ppm -> 1.23%% of L; his rms/sqrt(2N) route.')
print('       ours (HC3, honest): 10.3 ppm -> h*Rsun*dS = %.3f arcsec = %.1f%% of L'
      % (h_all*R_SUN_AS*10.3e-6, 100*h_all*R_SUN_AS*10.3e-6/GR))
print('       ours at HIS 3.34 ppm: %.3f arcsec = %.1f%% of L'
      % (h_all*R_SUN_AS*3.34e-6, 100*h_all*R_SUN_AS*3.34e-6/GR))
print('stars: his eq(20) gave 0.088 arcsec = 3.1%%; our bootstrap %.3f = %.1f%%; our '
      'analytic (fit cov) %.3f = %.1f%%'
      % (0.065, 100*0.065/GR, sL1, 100*sL1/GR))
