"""The Leon 2026 star table of record, written to disk -- the analogue of cell 1's
`bruns_method_star_table.csv`.

Until now the Leon union existed only inside a running process: `step3_s2_union.py`
builds it and prints L, and `step3_s2_plots.py` re-executes that build to draw. Nothing
on disk listed which stars carried the headline, with which displacement, from which
tiers. Cell 1 keeps its table beside its charts, and the comparison needs the same here.

This executes the union machinery unchanged (the headline's own build, catalogue G <= 11,
the 0.6+1.2 s tiers, anchor in) and writes:

  leon_union_star_table.csv             the headline set (0.6+1.2 s, anchor in)
  leon_union_star_table_sans_anchor.csv the same without the below-Sun anchor
  leon_union_star_table_full4.csv       all four tiers (the cross-check, not the record)
  leon_union_meta.json                  everything a chart needs without re-running the
                                        union: plate scale, Sun pixel, R_sun, the
                                        sky->sensor affine and its centre, tier mid-times

Columns: cat_i (index into the G <= 11 lookup, session-stable only), gaia_id, ra_cat /
dec_cat (Gaia, proper motion to 2026.61, NOT refraction-corrected -- for drawing), px / py
(median sensor position over the contributing tiers), dx / dy (median displacement,
obs - corrected catalogue, arcsec in SENSOR axes, per-tier median offset removed), mag,
ntier, tiers, spread (cross-tier disagreement, arcsec), R_rsun, is_anchor.

Two things the table makes checkable that the process did not: the per-star values the
arrows are drawn from, and the exact membership of the 42.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
os.makedirs(OUT, exist_ok=True)

src = open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])          # build machinery + tier tables, skip reports

ANCHOR_PX, ANCHOR_PY = 3161.0, 4163.0

# which tiers each catalogue star was seen in (the union keeps only the count)
seen_in = {}
for t, tab in tier_tabs.items():
    for i in tab.cat_i.values:
        seen_in.setdefault(int(i), []).append(t)

cra0 = np.degrees(cat.get_ra()); cdec0 = np.degrees(cat.get_dec())
try:
    gaia_ids = np.asarray(cat.get_ids())
except Exception:
    gaia_ids = np.array([''] * len(cra0))


def write(name, tiers, drop_anchor=False):
    U, rx_, ry_, R_ = build_union(tiers)
    U = U.copy()
    U['R_rsun'] = R_ / R_SUN_AS
    U['is_anchor'] = np.hypot(U.px.values - ANCHOR_PX, U.py.values - ANCHOR_PY) < 6
    if drop_anchor:
        U = U[~U.is_anchor]
    U['ra_cat'] = cra0[U.cat_i.values]
    U['dec_cat'] = cdec0[U.cat_i.values]
    U['gaia_id'] = [gaia_ids[i] if i < len(gaia_ids) else '' for i in U.cat_i.values]
    U['tiers'] = ['+'.join(t for t in tiers if t in seen_in.get(int(i), []))
                  for i in U.cat_i.values]
    cols = ['cat_i', 'gaia_id', 'ra_cat', 'dec_cat', 'px', 'py', 'dx', 'dy', 'mag',
            'ntier', 'tiers', 'spread', 'R_rsun', 'is_anchor']
    U = U[cols].sort_values('R_rsun').reset_index(drop=True)
    U.to_csv(os.path.join(OUT, name), index=False)
    print(f'{name}: {len(U)} stars, G {U.mag.min():.2f}-{U.mag.max():.2f}, '
          f'R {U.R_rsun.min():.2f}-{U.R_rsun.max():.2f} R_sun, anchor '
          f'{"in" if U.is_anchor.any() else "out"}', flush=True)
    return U


write('leon_union_star_table.csv', ('0p6s', '1p2s'))
write('leon_union_star_table_sans_anchor.csv', ('0p6s', '1p2s'), drop_anchor=True)
write('leon_union_star_table_full4.csv', ('0p1s', '0p3s', '0p6s', '1p2s'))

meta = dict(PS=PS, NX=NX, NY=NY, W_NORM=W_NORM, SUNPX=SUNPX, SUNPY=SUNPY, R_SUN_AS=R_SUN_AS,
            L_REF=L_REF, LIMIT_MAG=LIMIT_MAG, MIDT=MIDT, OPTS=OPTS,
            affine_ax=[float(v) for v in ax], affine_ay=[float(v) for v in ay],
            affine_ra0=float(ra0), affine_de0=float(de0),
            host_model=src_zip if 'src_zip' in dir() else None,
            note='affine: px = ax.[ (ra-ra0)cos(de0), dec-de0, 1 ], fitted from the 0.6 s '
                 'matched table; dx,dy = sensor-axis displacement in arcsec = PS * '
                 '(affine linear part applied to the sky displacement in degrees)')
json.dump(meta, open(os.path.join(OUT, 'leon_union_meta.json'), 'w'), indent=1)
print('meta ->', os.path.join(OUT, 'leon_union_meta.json'), flush=True)
