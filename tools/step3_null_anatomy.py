"""The anatomy of the null-test L, part 1: does it track the plate-scale step between the
paired fields? For every null pair (Leon zenith, Bruns night one-sided) the free-fit plate
scales of the field and its reference, the step in ppm, the field geometry's own scale
leverage by injection, and the measured null. Written 2026-09-02 for docs/STEP3_2026.md
"What the null-test L is made of". Output: step3_record/null_vs_scale_step.csv."""
import glob, json, os, numpy as np, pandas as pd

def leverage(px, py, PS, NX, NY, W, SUNPX, SUNPY, RS, nuis=2):
    xs, ys = (px-NX/2)/W, (py-NY/2)/W
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry)
    ux, uy = rx/R, ry/R; n = len(px); Z = np.zeros(n)
    cx = [np.ones(n), Z, -(py-NY/2)*PS, ux*RS/R]; cy = [Z, np.ones(n), (px-NX/2)*PS, uy*RS/R]
    for i in range(nuis+1):
        for j in range(nuis+1-i):
            if i == 0 and j == 0: continue
            cx.append(Z); cy.append(xs**i*ys**j)
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    dx = 1e-6*(px-NX/2)*PS; dy = 1e-6*(py-NY/2)*PS
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return float(c[3])

def ps_of(resglob):
    p = glob.glob(resglob, recursive=True)
    return json.load(open(p[0], encoding='utf-8'))['platescale (arcseconds/pixel)'] if p else np.nan

rows = []
# ---- Leon zenith
Z = pd.read_csv(r"D:/MEE2024 output/MEE_output/step3_record/zenith_nulls.csv")
for _, r in Z.iterrows():
    ps_f = ps_of(rf"D:/MEE2024 output/MEE_output/refraction/zenith12/{r.field}/stage2/**/distortion_results.txt")
    ps_r = ps_of(rf"D:/MEE2024 output/MEE_output/refraction/zenith12/{r.ref}/stage2/**/distortion_results.txt")
    res = glob.glob(rf"D:/MEE2024 output/MEE_output/step3_record/zenith_nulls/{r.field}/**/TWOD_RESIDUALS.csv", recursive=True)[0]
    d = pd.read_csv(res); d = d[d['magV'] <= 11]
    px, py = d.px.values, d.py.values
    R = np.hypot((px-3171)*2.2054043, (py-3232)*2.2054043); k = R > 2*947.1
    g = leverage(px[k], py[k], 2.2054043, 6248, 4176, 3124.0, 3171.0, 3232.0, 947.1)
    step = 1e6*(ps_f-ps_r)/ps_r
    rows.append(dict(set='Leon zenith', pair=f'{r.field} vs {r.ref}', step_ppm=step, lev=g, pred=g*step, null=r.Lv))
# ---- Bruns nights (atmosphere3): field vs previous same-night field; free-fit scales from bruns2017_nights
B = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3"
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
for group in ('EC', 'LC', 'RC'):
    ep = []
    for i in range(1, 11):
        f = f'{group}{i:02d}'
        p = glob.glob(os.path.join(NIGHTS, f, 'stage2', '**', 'distortion_results.txt'), recursive=True)
        if not p: continue
        j = json.load(open(p[0], encoding='utf-8'))
        tm = (j.get('observation_time (UTC)') or '0:0:0').split(':')
        ep.append((j.get('observation_date'), int(tm[0])*60+int(tm[1]), f, j['platescale (arcseconds/pixel)']))
    by = {}
    for date, m, f, ps in ep: by.setdefault(date, []).append((m, f, ps))
    for date in by:
        seq = sorted(by[date])
        for (m0, ref, ps_r), (m1, f, ps_f) in zip(seq, seq[1:]):
            res = glob.glob(os.path.join(B, f, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
            if not res: continue
            d = pd.read_csv(res[0]); d = d[d['magV'] <= 11]
            px, py = d.px.values, d.py.values
            R = np.hypot((px-1645)*2.0868004, (py-1741)*2.0868004); k = R > 2*948.7
            if k.sum() < 25: continue
            g = leverage(px[k], py[k], 2.0868004, 3296, 2472, 1648.0, 1645.0, 1741.0, 948.7)
            # the measured null for this field, recomputed the b17 way (v-deg2 along the local vertical)
            VX, VY = 0.447, -0.895
            dx = d.dx_arcsec.values[k]-np.median(d.dx_arcsec.values[k]); dy = d.dy_arcsec.values[k]-np.median(d.dy_arcsec.values[k])
            xs, ys = (px[k]-1648)/1648.0, (py[k]-1236)/1648.0
            rx, ry = (px[k]-1645)*2.0868004, (py[k]-1741)*2.0868004; Rk = np.hypot(rx, ry)
            n = k.sum(); Zc = np.zeros(n)
            cx = [np.ones(n), Zc, -(py[k]-1236)*2.0868004, rx/Rk*948.7/Rk]; cy = [Zc, np.ones(n), (px[k]-1648)*2.0868004, ry/Rk*948.7/Rk]
            for i in range(3):
                for jj in range(3-i):
                    if i == 0 and jj == 0: continue
                    cx.append(VX*xs**i*ys**jj); cy.append(VY*xs**i*ys**jj)
            A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
            c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
            step = 1e6*(ps_f-ps_r)/ps_r
            rows.append(dict(set='Bruns night', pair=f'{f} vs {ref}', step_ppm=step, lev=g, pred=g*step, null=c[3]))
T = pd.DataFrame(rows)
for s, sub in T.groupby('set', sort=False):
    print(f'\n=== {s}: {len(sub)} pairs ===')
    print(f'{"pair":34} {"scale step":>10} {"leverage":>9} {"predicted":>10} {"measured":>9}')
    for _, r in sub.iterrows():
        print(f'{r.pair:34} {r.step_ppm:+8.1f} ppm {r.lev:8.4f} {r.pred:+9.3f}" {r.null:+8.3f}"')
    x, y = sub.pred.values, sub.null.values
    slope = float(x@y/(x@x)); rr = float(np.corrcoef(x, y)[0, 1])
    resid = y - slope*x
    print(f'  measured vs predicted-from-scale-step: correlation r = {rr:+.2f}, slope {slope:.2f}')
    print(f'  null rms {np.sqrt(np.mean(y**2)):.3f}"; rms after removing the scale-step part {np.sqrt(np.mean(resid**2)):.3f}"; '
          f'rms of the scale-step part alone {np.sqrt(np.mean(x**2)):.3f}"')
    print(f'  scale steps: rms {np.sqrt(np.mean(sub.step_ppm.values**2)):.1f} ppm, max {np.abs(sub.step_ppm.values).max():.1f} ppm')
T.to_csv(r"D:/MEE2024 output/MEE_output/step3_record/null_vs_scale_step.csv", index=False)
