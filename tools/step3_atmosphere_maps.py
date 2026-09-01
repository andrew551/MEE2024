"""Leon 2026's atmosphere night maps, in the construction and style of cell 1's.

Cell 1's `atmosphere_night_maps.png` (tools/matrix_bruns/b17_m3_maps.py) draws every
night calibration field re-fitted with the cubic FROZEN (the same multi-field average the
calibration used) and the quadratic FREE -- exactly how a calibration field is reduced, so
the residual is what a calibration fit cannot absorb: cubic-and-above model error plus
quasi-static atmosphere. Positions and arrows both in sensor axes, one arrow scale
(0.0018: one arcsec draws as ~556 px), a crimson 1-arcsec reference and a green
increasing-altitude arrow on every panel.

Leon's night fields come in two kinds, and both are drawn:

  * the nine HORIZON field-windows (H1 = the eclipse alt/az, H2 = +2 deg, H3 = the
    calibration-field sightline; nights N1/N2/N3), alt 8.5-12.4 deg -- the eclipse-geometry
    rehearsal, the direct analogue of Bruns' EC/LC/RC. Their per-frame quadratic-free
    corrections-ON fits already exist (the M2 ladder, refraction/perframe/); the map is the
    per-star MEDIAN over the ~45 frames, the M3 construction, which is what a stack keeps;
  * the twelve ZENITH fields (six per night), the fields that supplied the frozen cubic --
    re-fitted here with that night's six-field average frozen and the quadratic free, the
    exact cell-1 construction (Bruns' EC06 is a member of the 15-field average that is
    frozen onto it, as each zenith field is a member of its night's six). At alt 85-89 deg
    the altitude direction is nearly meaningless, so those panels carry no green arrow and
    no alt/az split.

Science cut applied to what is drawn: G <= 11, as on the Bruns maps.

Outputs: step3_record/atmosphere_night_maps.png (+ versioned copy), atmosphere_maps_stats.csv,
and the zenith re-fits under step3_record/zenith_quadfree/<field>/.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RD = r"D:/MEE2024 output/MEE_output/refraction"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
VER = os.path.join(OUT, 'chart_versions')
os.makedirs(VER, exist_ok=True)
REV = os.environ.get('L26_REV', 'rev01')
NX, NY, PS = 6248, 4176, 2.2054043
LSCALE = 0.0018
MAGCUT = 11.0
SITE = EarthLocation(lat=42.740470*u.deg, lon=-5.613780*u.deg, height=1101*u.m)
DATE = {'N1': '2026-08-11', 'N2': '2026-08-12', 'N3': '2026-08-13'}
ZEN_REFS = {n: sorted(glob.glob(os.path.join(REPO, 'calibration', 'zenith_cubic', n + '_Z*.txt')))
            for n in ('08-11', '08-12')}
assert all(len(v) == 6 for v in ZEN_REFS.values())


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=120)
    fig.savefig(os.path.join(VER, REV + '_' + name), dpi=120)
    plt.close(fig)


def clip_medians(acc, min_frames=20):
    ids = [k for k, v in acc.items() if len(v) >= min_frames]
    P = np.array([[np.median([q[c] for q in acc[i]]) for c in range(4)] + [acc[i][0][4]]
                  for i in ids])
    px, py, qx, qy, mag = P.T
    qx, qy = qx - np.median(qx), qy - np.median(qy)
    m = np.hypot(qx, qy)
    lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
    good = m < lim
    return px[good], py[good], qx[good], qy[good], mag[good], int((~good).sum())


def horizon_altaz_basis(w, f):
    """Unit vectors (sensor px basis) of increasing altitude and azimuth, from one
    mid-block solved frame -- the m3_maps.py construction."""
    cal = sorted(glob.glob(os.path.join(RD, 'perframe', w, f, 'f20', 'corr_on', '**',
                                        'CATALOGUE_MATCHED_ERRORS.csv'), recursive=True))
    resf = sorted(glob.glob(os.path.join(RD, 'perframe', w, f, 'f20', 'corr_on', '**',
                                         'distortion_results.txt'), recursive=True))
    d = pd.read_csv(cal[0]); d.columns = [c.strip() for c in d.columns]
    j = json.load(open(resf[0], encoding='utf-8'))
    t = Time(f"{DATE[w]}T{j['observation_time (UTC)']}", scale='utc')
    aa = SkyCoord(d['RA(catalog)'].values*u.deg, d['DEC(catalog)'].values*u.deg
                  ).transform_to(AltAz(obstime=t, location=SITE))
    alt = aa.alt.deg
    azc = aa.az.deg*np.cos(np.radians(alt.mean()))
    A = np.column_stack([azc - azc.mean(), alt - alt.mean(), np.ones(len(d))])
    cx, *_ = np.linalg.lstsq(A, d.px.values, rcond=None)
    cy, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    v_az = np.array([cx[0], cy[0]]); v_az /= np.linalg.norm(v_az)
    v_alt = np.array([cx[1], cy[1]]); v_alt /= np.linalg.norm(v_alt)
    return v_alt, v_az, float(alt.mean())


rows, panels = [], []

# ---- the nine horizon field-windows: per-star medians of the M2 per-frame fits
for w in ('N1', 'N2', 'N3'):
    for f in ('H1', 'H2', 'H3'):
        files = sorted(glob.glob(os.path.join(RD, 'perframe', w, f, 'f*', 'corr_on', '**',
                                              'TWOD_RESIDUALS.csv'), recursive=True))
        acc = {}
        for fp in files:
            d = pd.read_csv(fp)
            d = d[d['magV'] <= MAGCUT]
            for _, r in d.iterrows():
                acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec, r.magV))
        px, py, qx, qy, mag, nclip = clip_medians(acc)
        v_alt, v_az, alt = horizon_altaz_basis(w, f)
        q_alt = qx*v_alt[0] + qy*v_alt[1]
        q_az = qx*v_az[0] + qy*v_az[1]
        name = f'{w}/{f}'
        rows.append(dict(field=name, kind='horizon', date=DATE[w], n=len(px), n_frames=len(files),
                         alt=alt, qs_rms=float(np.sqrt(np.mean(qx**2 + qy**2))),
                         qs_alt=float(np.sqrt(np.mean(q_alt**2))),
                         qs_az=float(np.sqrt(np.mean(q_az**2))), clipped=nclip))
        panels.append((name, px, py, qx, qy, alt, v_alt))
        print('%-6s alt %5.1f  N=%3d (of %d frames)  rms %.3f (alt %.3f, az %.3f) arcsec'
              % (name, alt, len(px), len(files), rows[-1]['qs_rms'], rows[-1]['qs_alt'],
                 rows[-1]['qs_az']), flush=True)


# ---- the twelve zenith fields: re-fit quadratic-free against the night's frozen six
def zenith_quadfree(field):
    night = field[:5]
    src = os.path.join(RD, 'zenith12', field)
    cz = glob.glob(os.path.join(src, 'centroid_data*.zip'))
    orig = glob.glob(os.path.join(src, 'stage2', '**', 'distortion_results.txt'), recursive=True)
    if not cz or not orig:
        return None
    j0 = json.load(open(orig[0], encoding='utf-8'))
    d = os.path.join(OUT, 'zenith_quadfree', field)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        # the same tolerance and magnitude the field's own free-cubic fit used, corrections
        # off as at zenith (docs/LEON_2026-08-11.md section 18.1)
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', cz[0], '--order', 'cubic',
                            '--date-from-header', '--fix-distortion', *ZEN_REFS[night],
                            '--set', 'distortion_fixed_coefficients=quadratic',
                            '--set', 'distortion_fit_tol=%s' % j0.get('error tolerance (as)', 0.5),
                            '--set', 'max_star_mag_dist=%s' % j0.get('star max magnitude', 13),
                            '--set', 'rough_match_threshhold=36',
                            '--set', 'enable_corrections=False',
                            '--set', 'enable_corrections_ref=False',
                            '--no-display', '--quiet', '-o', d],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return (hit[0], j0) if hit else None


for night in ('08-11', '08-12'):
    for z in ('Z1_base', 'Z2_mid_left', 'Z3_top_left', 'Z4_top_right', 'Z5_mid_right',
              'Z6_bottom_right'):
        field = f'{night}_{z}'
        got = zenith_quadfree(field)
        if got is None:
            print(f'{field}: no quadratic-free residuals', flush=True)
            continue
        csvpath, j0 = got
        d = pd.read_csv(csvpath)
        d = d[d['magV'] <= MAGCUT]
        px, py = d['px'].values, d['py'].values
        qx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
        qy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
        m = np.hypot(qx, qy)
        lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
        good = m < lim
        px, py, qx, qy = px[good], py[good], qx[good], qy[good]
        rows.append(dict(field=field, kind='zenith', date='2026-' + night, n=len(px),
                         n_frames=np.nan, alt=np.nan,
                         qs_rms=float(np.sqrt(np.mean(qx**2 + qy**2))),
                         qs_alt=np.nan, qs_az=np.nan, clipped=int((~good).sum())))
        panels.append((field, px, py, qx, qy, np.nan, None))
        print('%-18s zenith  N=%4d  rms %.3f arcsec (quadratic-free, night six frozen)'
              % (field, len(px), rows[-1]['qs_rms']), flush=True)

S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, 'atmosphere_maps_stats.csv'), index=False)

# ---- the figure: sensor axes for positions AND arrows, one scale, y down
ncol = 6
nrow = int(np.ceil(len(panels)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.2*nrow))
for ax, (name, px, py, qx, qy, alt, v_alt) in zip(axes.ravel(), panels):
    ax.quiver(px, py, qx, qy, angles='xy', scale_units='xy', scale=LSCALE, width=0.004,
              color='tab:blue')
    ax.quiver([300], [300], [1.0], [0.0], angles='xy', scale_units='xy', scale=LSCALE,
              width=0.006, color='crimson')
    ax.annotate('1"', (330, 480), fontsize=8, color='crimson')
    if v_alt is not None:
        va = v_alt*600
        ax.annotate('', xy=(5600 + va[0], 3700 + va[1]), xytext=(5600, 3700),
                    arrowprops=dict(arrowstyle='->', color='green'))
        ax.text(5250, 3350, 'up', color='green', fontsize=8)
        ax.set_title('%s  (alt %.1f deg, %d stars)' % (name, alt, len(px)), fontsize=10)
    else:
        ax.set_title('%s  (zenith, %d stars)' % (name, len(px)), fontsize=10)
    rms = float(np.sqrt(np.mean(qx**2 + qy**2)))
    ax.text(0.02, 0.04, 'rms %.2f"' % rms, transform=ax.transAxes, fontsize=8)
    ax.set_xlim(0, NX); ax.set_ylim(NY, 0); ax.set_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[len(panels):]:
    ax.axis('off')
fig.suptitle('Leon 2026 night fields: residual structure a calibration fit cannot absorb \u2014 '
             'the nine horizon windows at the eclipse geometry (per-star medians of ~45 '
             'per-frame quadratic-free fits, corrections ON)\nand the twelve zenith fields '
             '(one stack each, that night\u2019s six-field cubic frozen, quadratic free); '
             'G \u2264 11; arrows and positions both in SENSOR axes; arrow scale identical to the '
             'Bruns 2017 maps', fontsize=12)
fig.tight_layout()
save(fig, 'atmosphere_night_maps.png')

H = S[S.kind == 'horizon']; Z = S[S.kind == 'zenith']
print('\nhorizon windows (alt %.1f-%.1f): qs rms %.3f (%.3f-%.3f), alt-component %.3f, '
      'az-component %.3f, V/H %.1f'
      % (H.alt.min(), H.alt.max(), H.qs_rms.mean(), H.qs_rms.min(), H.qs_rms.max(),
         H.qs_alt.mean(), H.qs_az.mean(), H.qs_alt.mean()/max(H.qs_az.mean(), 1e-9)))
print('zenith fields: qs rms %.3f (%.3f-%.3f) over %d fields'
      % (Z.qs_rms.mean(), Z.qs_rms.min(), Z.qs_rms.max(), len(Z)))
print('Bruns 2017 night fields at the eclipse geometry, same construction: rms 0.102 '
      '(alt 0.074, az 0.069), V/H 1.0')
print('maps + stats ->', OUT)
