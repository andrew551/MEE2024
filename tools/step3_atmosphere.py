"""Leon 2026's atmospheric systematic, re-derived by cell 1's corrected method.

Leon's +-0.33 arcsec came from the S1 gate (tools/step3_s1_estimator.py, gate v2): the
M5 rehearsal's per-star median residuals (H1 frames fitted constant-only against the H3
reference, the night analogue of the science tier against CAL_piLeo), handed to the
estimator with the eclipse Sun imposed, worst |null| over the three windows with the
vertical-deg-2 nuisance = 0.32. Cell 1's number (+-0.15) was then built the same way in
spirit but with three details fixed along the way (tools/matrix_bruns/b17_atmosphere2.py,
docs/STEP3_CHARTS_AND_SETTINGS.md section 2), and the comparison should rest on one
construction. This re-derives Leon's term with those details applied:

  * each field's OWN observation time -- M5 already passed the per-frame time (the
    placeholder-time trap that manufactured +-1.40 on Bruns never applied here; checked
    below by reading the time back out of every rehearsal fit);
  * pairs INSIDE one night, minutes apart -- H1 and H3 are 10-13 minutes apart on each
    night (H1 block 22:31, H3 block 22:41 on 08-12), which is longer than Bruns' 6-7
    minute pairs and much longer than the science chain's ~2 minutes; stated, not hidden;
  * the SCIENCE cuts: G <= 11 (the gate used every matched star to G 13), R > 2 R_sun
    about the imposed Sun, the union's Sun pixel (3171, 3232), the union's design matrix;
  * the summary statistic is the rms over fields (cell 1's), reported alongside the max
    (Leon's original).

Two constructions are reported, because Leon's night data allows both and they bound
the science case from either side:

  A. per-star MEDIAN over the 45 frames (the quasi-static field; what a stack inherits
     and what the S1 gate used);
  B. every FRAME as its own null (quasi-static + that frame's 6 s jitter; an upper bound
     for a 7 s science stack, which averages a dozen frames but over only 40 s).

Neither is the eclipse-day atmosphere; both are the field-to-field differential over
minutes at alt 8.5-12.4 deg, inherited through a frozen distortion, measured at night.

Output: step3_record/atmosphere_nulls.csv and the numbers printed.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

RD = r"D:/MEE2024 output/MEE_output/refraction"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
os.makedirs(OUT, exist_ok=True)
PS, NX, NY, W_NORM = 2.2054043, 6248, 4176, 3124.0
SUNPX, SUNPY = 3171.0, 3232.0
R_SUN_AS, L_REF = 947.1, 1.7512
RCUT, MAGCUT = 2.0, 11.0


def design(x_px, y_px, rx, ry, R, nuis_deg=None):
    """The union's design matrix (tools/step3_s2_union.py), verbatim: N1, N2, Theta, L,
    then the vertical (sensor-y) polynomial nuisance."""
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ux, uy = rx/R, ry/R
    n = len(x_px)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ux*R_SUN_AS/R]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, uy*R_SUN_AS/R]
    labels = ['N1', 'N2', 'Th', 'L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, x_px, y_px, rx, ry, R, nuis_deg=None):
    A, labels = design(x_px, y_px, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]


def science_cut(px, py, mag):
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
    R = np.hypot(rx, ry)
    keep = (R > RCUT*R_SUN_AS) & (mag <= MAGCUT)
    return keep, rx, ry, R


def null_pair(px, py, dx, dy, mag, err, rng):
    keep, rx, ry, R = science_cut(px, py, mag)
    if keep.sum() < 25:
        return None
    px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
    dx = dx - np.median(dx); dy = dy - np.median(dy)
    Lb = fit_L(dx, dy, px, py, rx, ry, R)
    Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
    boots = [fit_L(dx + rng.normal(0, err/np.sqrt(2)), dy + rng.normal(0, err/np.sqrt(2)),
                   px, py, rx, ry, R, nuis_deg=2) for _ in range(60)]
    return dict(n=int(keep.sum()), Lb=Lb, Lv=Lv, floor=float(np.std(boots, ddof=1)),
                rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2)),
                h=float(1/np.mean((R_SUN_AS/R)**2)))


rng = np.random.default_rng(11)
rows = []
for w in ('N1', 'N2', 'N3'):
    files = sorted(glob.glob(os.path.join(RD, 'm5_rehearsal', w, 'f*', '**',
                                          'TWOD_RESIDUALS.csv'), recursive=True))
    # the time trap, checked: every rehearsal fit must carry its own frame time
    times = []
    for f in files:
        r = glob.glob(os.path.join(os.path.dirname(f), 'distortion_results.txt'))
        if r:
            times.append(json.load(open(r[0], encoding='utf-8')).get('observation_time (UTC)'))
    ref = glob.glob(os.path.join(RD, 'm5_rehearsal', w, 'H3_reference', 'stage2', '**',
                                 'distortion_results.txt'), recursive=True)
    tref = json.load(open(ref[0], encoding='utf-8'))['observation_time (UTC)'] if ref else '?'
    print(f'{w}: {len(files)} rehearsal frames, frame times {min(times)}..{max(times)} '
          f'({len(set(times))} distinct), H3 reference at {tref}', flush=True)
    assert len(set(times)) > len(files)//2, 'placeholder observation time detected'

    acc = {}
    per_frame = []
    for k, f in enumerate(files):
        d = pd.read_csv(f)
        # B: this frame as its own null field
        got = null_pair(d['px'].values, d['py'].values, d['dx_arcsec'].values,
                        d['dy_arcsec'].values, d['magV'].values, d['error_arcsec'].values,
                        rng)
        if got:
            per_frame.append(got)
            rows.append(dict(window=w, kind='frame', field=os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(f)))), **got))
        for _, r in d.iterrows():
            acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec, r.magV))
    # A: per-star medians over the frames (>= 20 frames each), MAD-clipped as the gate did
    ids = [i for i, v in acc.items() if len(v) >= 20]
    P = np.array([[np.median([q[0] for q in acc[i]]), np.median([q[1] for q in acc[i]]),
                   np.median([q[2] for q in acc[i]]), np.median([q[3] for q in acc[i]]),
                   acc[i][0][4],
                   np.std([q[2] for q in acc[i]], ddof=1)/np.sqrt(len(acc[i])),
                   np.std([q[3] for q in acc[i]], ddof=1)/np.sqrt(len(acc[i]))]
                  for i in ids])
    mpx, mpy, mdx, mdy, mmag, sdx, sdy = P.T
    mdx, mdy = mdx - np.median(mdx), mdy - np.median(mdy)
    mag_ = np.hypot(mdx, mdy)
    lim = max(3.0*1.4826*np.median(np.abs(mag_ - np.median(mag_))) + np.median(mag_), 2.5)
    good = mag_ < lim
    got = null_pair(mpx[good], mpy[good], mdx[good], mdy[good], mmag[good],
                    np.hypot(sdx, sdy)[good], rng)
    rows.append(dict(window=w, kind='median', field='per-star median', **got))
    pf = pd.DataFrame(per_frame)
    print(f'  A (per-star median, {got["n"]} stars G<=11 outside 2 R_sun, h={got["h"]:.1f}): '
          f'L base {got["Lb"]:+.3f}  L v-deg2 {got["Lv"]:+.3f}  (floor {got["floor"]:.3f}, '
          f'rms {got["rms"]:.3f} as/axis)', flush=True)
    print(f'  B (per frame, {len(pf)} frames, N {pf.n.min()}-{pf.n.max()}): '
          f'L v-deg2 mean {pf.Lv.mean():+.3f} rms {np.sqrt((pf.Lv**2).mean()):.3f} '
          f'max |{np.abs(pf.Lv).max():.3f}|; L base rms {np.sqrt((pf.Lb**2).mean()):.3f}; '
          f'residual rms {pf.rms.mean():.3f} as/axis', flush=True)

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'atmosphere_nulls.csv'), index=False)
med = T[T.kind == 'median']
frm = T[T.kind == 'frame']
print('\n=== Leon 2026 atmospheric systematic, cell-1 construction ===')
for tag, col in (('L base', 'Lb'), ('L v-deg2', 'Lv')):
    v = med[col].values
    print(f'  A per-star medians, 3 windows, {tag:8}: values {np.round(v, 3)}  '
          f'rms {np.sqrt((v**2).mean()):.3f}  max {np.abs(v).max():.3f}')
for tag, col in (('L base', 'Lb'), ('L v-deg2', 'Lv')):
    v = frm[col].values
    print(f'  B per frame, {len(v)} nulls,      {tag:8}: rms {np.sqrt((v**2).mean()):.3f}  '
          f'max {np.abs(v).max():.3f}  mean {v.mean():+.3f}')
print(f'  bootstrap floor (median over fields): {T.floor.median():.3f}')
v = med['Lv'].values
print(f'\nLEON ATMOSPHERIC SYSTEMATIC (rms of the v-deg2 per-star-median nulls, cell-1 '
      f'statistic) = +-{np.sqrt((v**2).mean()):.2f} arcsec; the S1 gate quoted the max, '
      f'+-{np.abs(v).max():.2f}. Per-frame upper bound +-{np.sqrt((frm.Lv.values**2).mean()):.2f}.')
print('table ->', os.path.join(OUT, 'atmosphere_nulls.csv'))
