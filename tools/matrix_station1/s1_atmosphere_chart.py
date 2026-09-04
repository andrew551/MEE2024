"""Cell 2's atmosphere term, drawn: the zenith null test and the quasi-static floor.

Douglas, 2026-09-04: "Atmospheric term: already done, the same way as Bruns and Leon: is
there a chart for this?"

There were numbers but no chart. This draws them, from `s1_zenith_floor.py`'s two outputs:

  * `zenith_nulls.csv` -- sixteen consecutive zenith pairs, each field fitted constant-only
    against the one before it with the ECLIPSE Sun imposed at px (4309, 2730) and the science
    cuts applied, so the fitted L is a pure null: its true value is zero and whatever comes out
    is the floor the atmosphere and the transfer leave behind;
  * `zenith_floor.csv` -- each field refitted with the quintic-and-above frozen from the
    seventeen-field average and the quadratic free, which is the quasi-static residual.

Two panels, one row:

  LEFT   the sixteen nulls in capture order, Method 1 with the vertical-deg-2 nuisance and
         Method 2 with the scale free, with their rms bands. Method 1 is shown because it is
         what the other cells quote; Method 2 because it is how Station 1 is actually reduced.
         The pair spanning the 05:51 field is marked -- that field's sensor ran 12.7 C warm and
         its plate scale sits +68 ppm off, so its neighbours are the two worst nulls and the
         reason the summary also quotes the figure without them.

  RIGHT  the quasi-static residual per field against the campaign's other floors, so cell 2's
         number can be read against Leon's and Bruns'.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REC = r"D:/MEE2024 output/MEE_output/station1_record"
OUT = os.path.join(REC, 'atmosphere_chart.png')
N = pd.read_csv(os.path.join(REC, 'zenith_nulls.csv'))
F = pd.read_csv(os.path.join(REC, 'zenith_floor.csv'))

warm = N.index[N.field.astype(str).str.contains('203916') | N.ref.astype(str).str.contains('203916')]
rms_m1 = float(np.sqrt((N.Lv.values**2).mean()))
rms_m2 = float(np.sqrt((N.L_m2.values**2).mean()))
keep = ~N.index.isin(warm)
rms_m2_clean = float(np.sqrt((N.L_m2.values[keep]**2).mean()))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={'width_ratios': [1.55, 1]})
x = np.arange(len(N))
ax.axhline(0, color='black', lw=1)
ax.axhspan(-rms_m1, rms_m1, color='tab:orange', alpha=0.10)
ax.axhspan(-rms_m2, rms_m2, color='tab:blue', alpha=0.14)
ax.plot(x, N.Lv, 'o-', color='tab:orange', ms=5, lw=1.2,
        label='Method 1, vertical-deg-2 nuisance   rms %.3f$^{\\prime\\prime}$' % rms_m1)
ax.plot(x, N.L_m2, 's-', color='tab:blue', ms=5, lw=1.4,
        label='Method 2, scale free   rms %.3f$^{\\prime\\prime}$   (%.3f without the warm pair)'
              % (rms_m2, rms_m2_clean))
for i in warm:
    ax.axvline(i, color='crimson', ls=':', lw=1.2)
if len(warm):
    ax.annotate('the 05:51 field:\nsensor 12.7 $^\\circ$C warm,\nplate scale +68 ppm',
                (warm[0], N.L_m2.min()), textcoords='offset points', xytext=(6, -4),
                fontsize=8, color='crimson', va='top')
ax.set_xticks(x)
ax.set_xticklabels([('%s' % f)[-6:-2] for f in N.field], rotation=90, fontsize=7)
ax.set_xlabel('consecutive zenith pair (field, capture order 05:32$-$06:15 UTC)', fontsize=10)
ax.set_ylabel('fitted L on a field whose true L is zero  ($^{\\prime\\prime}$)', fontsize=11)
ax.set_title('The null test: cell 2\'s atmosphere term\n'
             'each zenith field fitted against the previous one, eclipse Sun imposed, science cuts',
             fontsize=11)
ax.legend(fontsize=8.5, loc='upper left')
ax.grid(alpha=0.25)

# Station 1 has NO Method 1 floor to quote, and that is not an oversight. Method 1 imports a
# plate scale, and Station 1's only calibration sits -640 ppm from the eclipse because the
# telescope was refocused between them, so there is no same-day scale that can legitimately be
# imported and Method 1 is not available for this cell at all. The Method 1 null number is a
# diagnostic of that impossibility, not a floor, and is kept out of this comparison.
floors = [('Bruns 2017 night\n(bracketed)', 0.059, 'tab:green'),
          ('Station 1 2024\nMethod 2', rms_m2, 'tab:blue'),
          ('Leon 2026 zenith\none-sided', 0.12, 'tab:purple'),
          ('Leakey 2024 zenith', 0.12, 'tab:grey'),
          ('Leon 2026 horizon', 0.33, 'tab:red')]
floors.sort(key=lambda t: t[1])
bx.barh([f[0] for f in floors], [f[1] for f in floors], color=[f[2] for f in floors], alpha=0.85)
for i, (nm, v, _) in enumerate(floors):
    bx.text(v + 0.006, i, '%.3f$^{\\prime\\prime}$' % v, va='center', fontsize=9)
bx.set_xlabel('null-test floor ($^{\\prime\\prime}$)', fontsize=10)
bx.set_xlim(0, 0.40)
bx.set_title('against the other cells\n(Station 1 sits between Bruns bracketed and Leon)', fontsize=11)
bx.grid(axis='x', alpha=0.25)
bx.tick_params(labelsize=8.5)

fig.text(0.012, 0.015,
         'Station 1: %d consecutive pairs, %.1f$-$%.1f min apart, %d$-$%d stars each, h = %.0f R$_\\odot^2$; '
         'photon floor %.3f$^{\\prime\\prime}$, quasi-static residual %.3f$^{\\prime\\prime}$ over %d fields '
         '(%.3f$-$%.3f).  Station 1 has NO Method 1 floor and that is not an oversight: Method 1 imports '
         'a plate scale, and the refocus put the only calibration $-$640 ppm from the eclipse, so there is '
         'no same-day scale that can legitimately be imported. The orange trace on the left sizes that '
         'impossibility; it is not a floor for this cell, and it is kept out of the comparison on the right.'
         % (len(N), N.gap_min.min(), N.gap_min.max(), int(N.n.min()), int(N.n.max()), N.h.mean(),
            N.floor.median(), F.qs_rms.mean(), len(F), F.qs_rms.min(), F.qs_rms.max()),
         fontsize=7.6)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(OUT, dpi=190)
print('%d nulls: Method 1 rms %.3f", Method 2 rms %.3f" (%.3f without the warm pair)'
      % (len(N), rms_m1, rms_m2, rms_m2_clean))
print('quasi-static floor %.3f" over %d fields (%.3f-%.3f)' % (F.qs_rms.mean(), len(F), F.qs_rms.min(), F.qs_rms.max()))
print('->', OUT)
