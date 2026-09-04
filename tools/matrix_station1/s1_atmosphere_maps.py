"""Station 1's atmosphere night maps, in the construction and style of cells 1 and 3.

Douglas, 2026-09-04: "this is not what I was looking for. I wanted a chart like
atmosphere_night_maps.png in D:\\MEE2024 output\\MEE_output\\RECORD\\bruns2017".

The bar-and-run chart of `s1_atmosphere_chart.py` summarised the null test. This is the other
thing, and the one the other two cells have: **the residual structure a calibration fit cannot
absorb**, one panel per field, drawn as vectors in sensor axes.

Same construction as `matrix_bruns/b17_m3_maps.py` and `step3_atmosphere_maps.py`: each zenith
field re-fitted with the quintic-and-above FROZEN from the seventeen-field average and the
quadratic FREE -- exactly how a calibration field is reduced -- so what is left is quintic-
and-above model error plus quasi-static atmosphere. Those re-fits already exist, in
`station1_record/zenith_quadfree_corr/`, built by `s1_zenith_floor.py` with the 2024
corrections flags matched.

Held identical to the other cells so the three can be read side by side:

  * positions AND arrows both in sensor axes, y down;
  * one arrow scale for every panel, and the SAME scale as the Bruns and Leon maps
    (LSCALE 0.0018: one arcsec draws as ~556 px) -- so a Station 1 panel can be laid beside a
    Bruns panel and compared by eye;
  * a crimson one-arcsec reference in every panel;
  * science cut G <= 11, as on the Bruns and Leon maps.

The seventeen fields sit at zenith distance 1-3 deg, so as on Leon's zenith panels there is no
green altitude arrow and no alt/az split: the altitude direction is not meaningful there.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REC = r"D:/MEE2024 output/MEE_output/station1_record"
QF = os.path.join(REC, 'zenith_quadfree_corr')
OUT = os.path.join(REC, 'atmosphere_night_maps.png')
NX, NY = 9576, 6388
LSCALE = 0.0018          # identical to the Bruns 2017 and Leon 2026 maps
MAGCUT = 11.0

panels = []
for d in sorted(glob.glob(os.path.join(QF, '*'))):
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        continue
    t = pd.read_csv(hit[0])
    t = t[t.magV <= MAGCUT]
    if len(t) < 30:
        continue
    qx = t.dx_arcsec.values - np.median(t.dx_arcsec.values)
    qy = t.dy_arcsec.values - np.median(t.dy_arcsec.values)
    panels.append((os.path.basename(d), t.px.values, t.py.values, qx, qy))

if not panels:
    raise SystemExit('no quadratic-free refits found in %s' % QF)

ncol = 6
nrow = int(np.ceil(len(panels)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.2*nrow))
rows = []
for ax, (name, px, py, qx, qy) in zip(np.atleast_1d(axes).ravel(), panels):
    ax.quiver(px, py, qx, qy, angles='xy', scale_units='xy', scale=LSCALE, width=0.004,
              color='tab:blue')
    ax.quiver([300], [300], [1.0], [0.0], angles='xy', scale_units='xy', scale=LSCALE,
              width=0.006, color='crimson')
    ax.annotate('1"', (330, 480), fontsize=8, color='crimson')
    rms = float(np.sqrt(np.mean(qx**2 + qy**2)))
    ax.set_title('%s  (zenith, %d stars)' % (name[-6:], len(px)), fontsize=10)
    ax.text(0.02, 0.04, 'rms %.3f"' % rms, transform=ax.transAxes, fontsize=8)
    ax.set_xlim(0, NX); ax.set_ylim(NY, 0); ax.set_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])
    rows.append(dict(field=name, n=len(px), qs_rms=rms,
                     qs_x=float(np.sqrt((qx**2).mean())), qs_y=float(np.sqrt((qy**2).mean()))))
for ax in np.atleast_1d(axes).ravel()[len(panels):]:
    ax.axis('off')
S = pd.DataFrame(rows)
S.to_csv(os.path.join(REC, 'atmosphere_maps_stats.csv'), index=False)
fig.suptitle('Station 1, Mexico 2024: residual structure a calibration fit cannot absorb \u2014 '
             'the seventeen zenith calibration fields, 2024-04-08 05:32\u201306:15 UTC, z = 1\u20133\u00b0\n'
             '(one stack each, the seventeen-field quintic frozen, quadratic free, 2024 '
             'corrections flags matched); G \u2264 %g; arrows and positions both in SENSOR axes; '
             'arrow scale identical to the Bruns 2017 and Leon 2026 maps' % MAGCUT, fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print('%d panels; quasi-static residual %.3f" (%.3f-%.3f), sensor y/x %.2f'
      % (len(S), S.qs_rms.mean(), S.qs_rms.min(), S.qs_rms.max(), S.qs_y.mean()/S.qs_x.mean()))
print('  Bruns 2017 night 0.100", Leon 2026 zenith 0.067", Leon horizon 0.260"')
print('->', OUT)
