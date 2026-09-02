"""The atmospheric error floor, one table over three geometries.

Douglas, 2026-09-02, on seeing the horizon and zenith panels of `atmosphere_night_maps.png`
together: put the floors in a table. Three sets, all reduced by the SAME construction --
each field re-fitted with the cubic-and-above frozen from a multi-field average and the
quadratic free, which is exactly how a calibration field is reduced, so what remains is
the structure a calibration fit cannot absorb (cubic-and-above model error plus the
quasi-static atmosphere), G <= 11:

  * **Leon zenith**, twelve fields, alt ~85-89 deg -- the instrument-and-model floor, with
    almost no atmosphere above it. One stack per field, that night's own six-field cubic
    frozen (`step3_record/zenith_quadfree/`);
  * **Leon horizon**, nine field-windows, alt 8.5-12.4 deg -- the eclipse geometry. Per-star
    medians over ~45 per-frame fits, corrections ON (the M2 ladder, M3 construction);
  * **Bruns 2017 night**, 28 fields, alt 53-55 deg -- his EC/LC/RC rehearsals, which repeat
    the eclipse-day pointings on the two preceding nights, so they are the eclipse geometry
    and NOT a zenith set. One stack per field, the same 15-field cubic average frozen that
    his L/R calibration used (`tools/matrix_bruns/b17_m3_maps.py`).

The vertical/horizontal split is the load-bearing column: the horizontal component is
common to all three (instrument plus model), while the vertical is what grows toward the
horizon, and the vertical is the component that couples to a radial deflection signal.

Also tabulated, for each campaign, the null-test L systematic -- the estimator run on real
fields with zero true deflection -- since that is what an error budget actually carries.

Writes `step3_record/atmosphere_floor_table.csv` and prints the markdown.
"""
import glob, os
import numpy as np, pandas as pd

LEON = r"D:/MEE2024 output/MEE_output/step3_record/atmosphere_maps_stats.csv"
BRUNS = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_m3/b17_m3_stats.csv"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
ZQF = os.path.join(OUT, 'zenith_quadfree')

leon = pd.read_csv(LEON)
bruns = pd.read_csv(BRUNS)
h = leon[leon.kind == 'horizon']
z = leon[leon.kind == 'zenith']

# the zenith fields carry no meaningful alt/az split (the vertical direction is degenerate
# overhead), so their anisotropy is measured in sensor axes instead -- it answers the
# question the V/H column asks of the others: is the floor isotropic?
zx, zy = [], []
for f in sorted(glob.glob(os.path.join(ZQF, '*'))):
    hit = glob.glob(os.path.join(f, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not hit:
        continue
    d = pd.read_csv(hit[0])
    d = d[d['magV'] <= 11.0]
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
    dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    m = np.hypot(dx, dy)
    lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
    g = m < lim
    zx.append(float(np.sqrt(np.mean(dx[g]**2))))
    zy.append(float(np.sqrt(np.mean(dy[g]**2))))
zx, zy = np.array(zx), np.array(zy)

rows = [
    dict(set='Leon 2026 zenith', geometry='zenith rehearsal (alt 85-89 deg)', n_fields=len(z),
         stars=int(z.n.sum()), rms=z.qs_rms.mean(), rms_lo=z.qs_rms.min(), rms_hi=z.qs_rms.max(),
         vertical=np.nan, horizontal=np.nan, VH=np.nan, null_L=np.nan,
         note='instrument + model floor; sensor-axis anisotropy y/x = %.2f' % (zy.mean()/zx.mean())),
    dict(set='Bruns 2017 night', geometry='the eclipse-day pointings (alt 53-55 deg)',
         n_fields=len(bruns), stars=int(bruns.n.sum()), rms=bruns.qs_rms.mean(),
         rms_lo=bruns.qs_rms.min(), rms_hi=bruns.qs_rms.max(), vertical=bruns.qs_alt.mean(),
         horizontal=bruns.qs_az.mean(), VH=bruns.qs_alt.mean()/bruns.qs_az.mean(), null_L=0.150,
         note='22 constant-only same-night nulls -> +-0.15 arcsec'),
    dict(set='Leon 2026 horizon', geometry='the eclipse geometry (alt 8.5-12.4 deg)',
         n_fields=len(h), stars=int(h.n.sum()), rms=h.qs_rms.mean(), rms_lo=h.qs_rms.min(),
         rms_hi=h.qs_rms.max(), vertical=h.qs_alt.mean(), horizontal=h.qs_az.mean(),
         VH=h.qs_alt.mean()/h.qs_az.mean(), null_L=0.33,
         note='3 M5 night-window nulls -> max 0.31, rms 0.22; +-0.33 quoted'),
]
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'atmosphere_floor_table.csv'), index=False)


def f(x, d=3):
    return '--' if x != x else f'{x:.{d}f}'


print('| set | geometry | fields | quasi-static rms (arcsec) | vertical (arcsec) | '
      'horizontal (arcsec) | V/H | null-test L systematic (arcsec) |')
print('|---|---|---|---|---|---|---|---|')
for _, r in T.iterrows():
    print(f'| **{r["set"]}** | {r.geometry} | {r.n_fields} | **{r.rms:.3f}** '
          f'({r.rms_lo:.3f}-{r.rms_hi:.3f}) | {f(r.vertical)} | {f(r.horizontal)} | '
          f'{f(r.VH, 1)} | {f(r.null_L, 2)} |')
print()
for _, r in T.iterrows():
    print(f'* {r["set"]}: {r.note}')
print()
print(f'Leon horizon / Leon zenith = {h.qs_rms.mean()/z.qs_rms.mean():.1f}x   '
      f'Leon horizon / Bruns night = {h.qs_rms.mean()/bruns.qs_rms.mean():.1f}x   '
      f'Bruns night / Leon zenith = {bruns.qs_rms.mean()/z.qs_rms.mean():.1f}x')
print(f'horizontal components: Bruns {bruns.qs_az.mean():.3f}, Leon horizon '
      f'{h.qs_az.mean():.3f} arcsec  ({h.qs_az.mean()/bruns.qs_az.mean():.1f}x)')
print(f'vertical components:   Bruns {bruns.qs_alt.mean():.3f}, Leon horizon '
      f'{h.qs_alt.mean():.3f} arcsec  ({h.qs_alt.mean()/bruns.qs_alt.mean():.1f}x)')
print(f'zenith sensor axes: x {zx.mean():.3f}, y {zy.mean():.3f} arcsec over {len(zx)} fields')
print('table ->', os.path.join(OUT, 'atmosphere_floor_table.csv'))
