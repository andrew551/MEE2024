"""The anatomy of the null-test L, part 2: refit every null with the plate scale FREE
(Method 2). What collapses was scale-like (drift or the reference's own scale estimation
error) and overlaps the budget's scale term; what survives is shape and overlaps nothing.
Written 2026-09-02 for docs/STEP3_2026.md "What the null-test L is made of"."""
import glob, os, numpy as np, pandas as pd

def nulls(files, PS, NX, NY, W, SUNPX, SUNPY, RS, VX=None, VY=None, magcut=11.0, rcut=2.0):
    out = []
    for f in files:
        d = pd.read_csv(f); d = d[d['magV'] <= magcut]
        px, py = d.px.values, d.py.values
        dx = d.dx_arcsec.values - np.median(d.dx_arcsec); dy = d.dy_arcsec.values - np.median(d.dy_arcsec)
        rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry); k = R > rcut*RS
        if k.sum() < 25: continue
        px, py, dx, dy, rx, ry, R = (a[k] for a in (px, py, dx, dy, rx, ry, R))
        xs, ys = (px-NX/2)/W, (py-NY/2)/W; n = len(px); Z = np.zeros(n)
        res = {}
        for tag, with_S in (('M1', False), ('M2', True)):
            cx = [np.ones(n), Z, -(py-NY/2)*PS]; cy = [Z, np.ones(n), (px-NX/2)*PS]; lab = ['N1','N2','Th']
            if with_S: cx.append((px-NX/2)*PS); cy.append((py-NY/2)*PS); lab.append('S')
            cx.append(rx/R*RS/R); cy.append(ry/R*RS/R); lab.append('L')
            for i in range(3):
                for j in range(3-i):
                    if i == 0 and j == 0: continue
                    if VX is None: cx.append(Z); cy.append(xs**i*ys**j)
                    else: cx.append(VX*xs**i*ys**j); cy.append(VY*xs**i*ys**j)
            A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
            c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
            res[tag] = c[lab.index('L')]
            if with_S: res['S_ppm'] = 1e6*c[lab.index('S')]
        out.append(dict(field=os.path.basename(f.split('DISTORTION')[0].rstrip('\/')), **res))
    return pd.DataFrame(out)

def report(name, T):
    m1, m2 = T.M1.values, T.M2.values
    print(f'{name:48} n={len(T):2d}  M1 null rms {np.sqrt((m1**2).mean()):.3f}"  M2 (scale free) null rms {np.sqrt((m2**2).mean()):.3f}"  '
          f'fitted scale step rms {np.sqrt((T.S_ppm.values**2).mean()):5.1f} ppm')

B1 = sorted(glob.glob(r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3/[ELR]C*/**/TWOD_RESIDUALS.csv", recursive=True))
B2 = sorted(glob.glob(r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3/bracket/*/**/TWOD_RESIDUALS.csv", recursive=True))
bruns = dict(PS=2.0868004, NX=3296, NY=2472, W=1648.0, SUNPX=1645.0, SUNPY=1741.0, RS=948.7, VX=0.447, VY=-0.895)
report('Bruns one-sided (field vs previous, 6-7 min)', nulls(B1, **bruns))
report('Bruns bracketed (field vs mean of before/after)', nulls(B2, **bruns))
Z = sorted(glob.glob(r"D:/MEE2024 output/MEE_output/step3_record/zenith_nulls/*/**/TWOD_RESIDUALS.csv", recursive=True))
leon = dict(PS=2.2054043, NX=6248, NY=4176, W=3124.0, SUNPX=3171.0, SUNPY=3232.0, RS=947.1)
report('Leon zenith consecutive (2.6 min)', nulls(Z, **leon))
# Leon M5: per-star medians per window (the gate construction), then M1/M2
rows = []
for w in ('N1','N2','N3'):
    acc = {}
    for f in sorted(glob.glob(rf"D:/MEE2024 output/MEE_output/refraction/m5_rehearsal/{w}/f*/**/TWOD_RESIDUALS.csv", recursive=True)):
        d = pd.read_csv(f)
        for _, r in d.iterrows(): acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec, r.magV))
    P = np.array([[np.median([q[c] for q in v]) for c in range(4)] + [v[0][4]] for v in acc.values() if len(v) >= 20])
    tmp = pd.DataFrame(dict(px=P[:,0], py=P[:,1], dx_arcsec=P[:,2], dy_arcsec=P[:,3], magV=P[:,4]))
    p = rf"C:/Users/dpesm/AppData/Local/Temp/claude/C--Users-dpesm-OneDrive-Documents-GitHub-MEE2024/b07fa5c2-0863-4163-b6a6-76b2bc21bff9/scratchpad/m5_{w}.csv"
    tmp.to_csv(p, index=False); rows.append(p)
T = nulls(rows, **leon)
print(f'{"Leon M5 horizon windows (per-star medians)":48} n={len(T):2d}  M1 null values {np.round(T.M1.values,3)}  M2 (scale free) {np.round(T.M2.values,3)}  fitted steps {np.round(T.S_ppm.values,1)} ppm')
print('   (Leon\'s one-sided field: L and S are near-degenerate under Method 2 -- the record measured -0.72 to +3.42" on these same windows -- so the M2 column here is a diagnostic, not a number)')
