"""M3-style residual maps for Bruns' night calibrations -- the atmosphere at the eclipse
geometry, and why Leon was limited.

Douglas' observation, and it is the key to the whole comparison: Bruns rehearsed the
**identical three pointings** on the two nights before the eclipse. Measured from the
fits themselves:

    EC  alt 54.56 az 143.4   <->  ECLIPSE field  alt 54.35 az 142.71
    LC  alt 53.56 az 131.0   <->  LEFT  cal      alt 53.47 az 130.63
    RC  alt 54.30 az 156.2   <->  RIGHT cal      alt 54.20 az 155.58

0.1-0.2 degrees in altitude and under a degree in azimuth. Same airmass, same optics,
same site -- a night-time replica of the eclipse-day geometry. Leon had nothing of the
kind: its rehearsal fields were at the zenith while the eclipse sat at 9.7 degrees, which
is why its atmospheric term had to be transported across a factor of six in airmass.

Construction, matched to `tools/refraction/m3_maps.py` so the numbers are comparable: each
night field is re-fitted with the **cubic and above frozen** (from the same 15-field
average the L/R calibration used) and **quadratic and below free** -- exactly how a
calibration field is reduced. The residual is therefore the structure a calibration fit
cannot absorb: cubic-and-above model error plus quasi-static atmosphere. Vectors are
rotated into the local alt-az frame through an affine fitted from the field's own matched
table, so "vertical" means vertical on the sky rather than on the sensor.

Outputs a quiver figure and a stats table alongside Leon's, in
D:/MEE2024 output/MEE_output/matrix_bruns2017_m3/.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_m3"
NX, NY, PS = 3296, 2472, 2.0868004
os.makedirs(OUT, exist_ok=True)
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']
# the same 15-field cubic average the L/R calibration froze
NIGHTREFS = json.load(open(glob.glob(
    r'D:/MEE2024 output/MEE_output/bruns2017_lr/L/stage2/DISTORTION_OUTPUT*/distortion/'
    r'distortion_results.txt')[0], encoding='utf-8'))['fixed distortion reference files'].split(';')


def original_epoch(field):
    """(date, UTC time) as the field's own first reduction recorded them.

    Load-bearing, not bookkeeping: the refraction correction is applied at the altitude
    implied by this time. A first version of this tool passed a placeholder 08:00 to every
    field, which put the correction at the wrong altitude and left a per-pointing
    systematic in the residuals -- visible as an alt/az split that made no physical sense
    (it reported alt 44 deg for a field the original fit puts at 54.6).
    """
    p = glob.glob(os.path.join(NIGHTS, field, 'stage2', '**', 'distortion_results.txt'),
                  recursive=True)
    if not p:
        return None, None
    j = json.load(open(p[0], encoding='utf-8'))
    return j.get('observation_date'), (j.get('observation_time (UTC)') or '08:00:00')


def quadfree(field):
    """Re-fit one night field with the cubic frozen and quadratic free; return residuals."""
    date, tm = original_epoch(field)
    if date is None:
        return None, None
    d = os.path.join(OUT, field)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        cz = glob.glob(os.path.join(NIGHTS, field, 'centroid_data*.zip'))
        if not cz:
            return None, None
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run([PY,'-m','mee2024.cli','distortion',cz[0],'--order','cubic',
                            '--date-from-header','--fix-distortion',*NIGHTREFS,
                            '--set','distortion_fixed_coefficients=quadratic',
                            '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
                            '--set','rough_match_threshhold=36',
                            '--set','enable_corrections=True',
                            '--set','enable_corrections_ref=True',*SITE,
                            '--set','observation_time=' + tm,'--no-display','--quiet','-o',d],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        return None, None
    res = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return hit[0], (res[0] if res else None)


def altaz_basis(resfile, csvfile):
    """Unit vectors of increasing altitude and azimuth, in sensor pixel axes.

    Built from the field's own matched table: an affine maps (RA, DEC) to (px, py), and
    the sky direction of increasing altitude is pushed through it. Same approach as
    m3_maps.py, which took it from a mid-block solve.
    """
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    j = json.load(open(resfile, encoding='utf-8'))
    d = pd.read_csv(csvfile)
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0))
    Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d['px'].values, rcond=None)
    ay, *_ = np.linalg.lstsq(A, d['py'].values, rcond=None)
    loc = EarthLocation(lat=42.7363889*u.deg, lon=-106.3180556*u.deg, height=2400*u.m)
    t = Time(j['observation_date'] + 'T' + (j.get('observation_time (UTC)') or '08:00:00'),
             scale='utc')
    fc = SkyCoord(j['RA']*u.deg, j['DEC']*u.deg)
    aa = fc.transform_to(AltAz(obstime=t, location=loc))
    out = {}
    for key, off in (('alt', dict(alt=aa.alt+0.05*u.deg, az=aa.az)),
                     ('az', dict(alt=aa.alt, az=aa.az+0.05*u.deg/np.cos(aa.alt)))):
        p = SkyCoord(AltAz(obstime=t, location=loc, **off)).icrs
        dv = np.array([(p.ra.deg-fc.ra.deg)*np.cos(np.radians(de0)), p.dec.deg-fc.dec.deg])
        v = np.array([dv @ ax[:2], dv @ ay[:2]])
        out[key] = v/np.linalg.norm(v)
    return out['alt'], out['az'], float(aa.alt.deg), float(aa.az.deg)


rows, maps = [], []
for group in ('EC', 'LC', 'RC'):
    for i in range(1, 11):
        field = '%s%02d' % (group, i)
        csvpath, respath = quadfree(field)
        if csvpath is None or respath is None:
            continue
        d = pd.read_csv(csvpath)
        d = d[d['magV'] <= 11.0]
        if len(d) < 40:
            continue
        matched = glob.glob(os.path.join(os.path.dirname(csvpath),
                                         'CATALOGUE_MATCHED_ERRORS.csv'))
        if not matched:
            continue
        e_alt, e_az, alt, az = altaz_basis(respath, matched[0])
        dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
        dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
        v_alt = dx*e_alt[0] + dy*e_alt[1]
        v_az = dx*e_az[0] + dy*e_az[1]
        j = json.load(open(respath, encoding='utf-8'))
        rows.append(dict(group=group, field=field, date=j['observation_date'], n=len(d),
                         alt=alt, az=az,
                         qs_rms=float(np.sqrt(np.mean(dx**2 + dy**2))),
                         qs_alt=float(np.sqrt(np.mean(v_alt**2))),
                         qs_az=float(np.sqrt(np.mean(v_az**2)))))
        maps.append((field, d['px'].values, d['py'].values, v_alt, v_az, alt))
        print('%s: N=%4d alt %.2f az %.2f  rms %.3f (alt %.3f, az %.3f) arcsec'
              % (field, len(d), alt, az, rows[-1]['qs_rms'], rows[-1]['qs_alt'],
                 rows[-1]['qs_az']), flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'b17_m3_stats.csv'), index=False)

if maps:
    # ALL fields (Douglas 2026-09-01: the first version sampled nine of the 28 and the
    # RECORD copy looked like fields had gone missing)
    sel = maps
    ncol = 6
    nrow = int(np.ceil(len(sel)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.2*nrow))
    # Arrow scale IDENTICAL to Leon's m3_quiver_maps (tools/refraction/m3_maps.py:
    # scale=0.0018, i.e. one arcsec of residual draws as ~556 px), with the same crimson
    # 1-arcsec reference arrow and green increasing-altitude arrow in every panel, so the
    # two figures compare side by side at a glance -- Douglas' request. The old
    # 'arrow length x250' annotation is gone with the old scale.
    LSCALE = 0.0018
    for ax, (field, px, py, va, vz, alt) in zip(axes.ravel(), sel):
        ax.quiver(px, py, vz, va, angles='xy', scale_units='xy', scale=LSCALE,
                  width=0.004, color='tab:blue')
        ax.quiver([260], [2150], [1.0], [0.0], angles='xy', scale_units='xy',
                  scale=LSCALE, width=0.006, color='crimson')
        ax.annotate('1"', (300, 2260), fontsize=8, color='crimson')
        ax.annotate('', xy=(180, 2000), xytext=(180, 1720),
                    arrowprops=dict(arrowstyle='->', color='green'))
        ax.annotate('up', (60, 1830), fontsize=8, color='green', rotation=90)
        ax.set_title('%s  (alt %.1f deg, %d stars)' % (field, alt, len(px)), fontsize=10)
        ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect(1)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(sel):]:
        ax.axis('off')
    fig.suptitle('Bruns 2017 night calibrations at the eclipse-day pointings: residual '
                 'structure a calibration fit cannot absorb\n'
                 '(cubic frozen, quadratic free; each vector decomposed into azimuth (x) and altitude (y) components; '
                 'arrow scale identical to the Leon m3 maps)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'b17_m3_quiver_maps.png'), dpi=120)
    plt.close(fig)

if len(R):
    print()
    print('Bruns night calibrations, per group (quasi-static residual, arcsec):')
    for g, sub in R.groupby('group'):
        print('  %s (alt %.1f): rms %.3f   alt-component %.3f   az-component %.3f   V/H %.1f'
              % (g, sub.alt.mean(), sub.qs_rms.mean(), sub.qs_alt.mean(), sub.qs_az.mean(),
                 sub.qs_alt.mean()/max(sub.qs_az.mean(), 1e-9)))
    print('  ALL   : rms %.3f   alt %.3f   az %.3f' % (R.qs_rms.mean(), R.qs_alt.mean(),
                                                       R.qs_az.mean()))
    print()
    print('Leon M3 for comparison (alt 8.5-12.4 deg, same construction):')
    print('  qs rms 0.167-0.349 (mean 0.261), alt-component 0.153-0.323, az 0.066-0.134, V/H ~2.3')
    print()
    print('ratio Leon/Bruns on the quasi-static residual: %.1fx'
          % (0.261/max(R.qs_rms.mean(), 1e-9)))
    print('maps + stats ->', OUT)
